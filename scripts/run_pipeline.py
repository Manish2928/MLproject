"""
Master pipeline script for DementiaVoiceAI.

Orchestrates the complete ML workflow:
1. Load configuration and set seed
2. Load dataset (recordings with paths and labels)
3. Extract acoustic features (or load from cache)
4. Split into train / validation / test (speaker-level)
5. Train all baseline models
6. Evaluate best model on test set
7. Generate plots and metrics
8. Compute permutation importance
9. Save best model and feature schema

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --no-cache   # force re-extraction
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import set_global_seed
from src.data.dataset_adapter import CelebritySpeechAdapter
from src.features.audio_features import AudioFeatureExtractor
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DementiaVoiceAI — Run the full ML pipeline"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction of features (ignore cached features)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # ----------------------------------------------------------------
    # 1. Load config and set seed
    # ----------------------------------------------------------------
    setup_logging()
    cfg = load_config(args.config)
    seed = cfg.project.seed
    set_global_seed(seed)
    logger.info(f"{'='*60}")
    logger.info(f"DementiaVoiceAI Pipeline — Starting")
    logger.info(f"{'='*60}")
    logger.info(f"Seed: {seed}")

    # ----------------------------------------------------------------
    # 2. Load dataset
    # ----------------------------------------------------------------
    logger.info("\n--- Step 1: Loading dataset ---")
    adapter = CelebritySpeechAdapter(cfg, anonymize=False)

    if not adapter.validate_source():
        logger.error("Dataset validation failed! Check data paths.")
        sys.exit(1)

    recordings = adapter.load_recordings()

    # Filter to files that exist on disk
    recordings = recordings[recordings["file_found"]].copy()
    logger.info(f"Recordings with audio on disk: {len(recordings)}")

    # Map labels to binary: dementia=1, control=0
    recordings["label_binary"] = (
        recordings["group"].map({"dementia": 1, "control": 0})
    )

    # Extract speaker_id from participant_id
    recordings["speaker_id"] = recordings["participant_id"]

    # ----------------------------------------------------------------
    # 3. Extract features (or load cache)
    # ----------------------------------------------------------------
    logger.info("\n--- Step 2: Feature extraction ---")
    cache_path = resolve_path(cfg.paths.data_root) / "features_cache.csv"

    if cache_path.exists() and not args.no_cache:
        logger.info(f"Loading cached features from {cache_path}")
        feature_df = pd.read_csv(cache_path)
        logger.info(f"Cached features loaded: {feature_df.shape}")
    else:
        extractor = AudioFeatureExtractor(cfg)
        feature_df = extractor.extract_batch(
            audio_paths=recordings["audio_path"].tolist(),
            labels=recordings["label_binary"].tolist(),
            speaker_ids=recordings["speaker_id"].tolist(),
        )
        feature_df.to_csv(cache_path, index=False)
        logger.info(f"Feature cache saved to {cache_path}")

    # Ensure labels and speaker_ids are present
    if "label" not in feature_df.columns:
        # Merge from recordings
        path_to_label = recordings.set_index("audio_path")[
            ["label_binary", "speaker_id"]
        ].to_dict()
        feature_df["label"] = feature_df["audio_path"].map(
            path_to_label["label_binary"]
        )
        feature_df["speaker_id"] = feature_df["audio_path"].map(
            path_to_label["speaker_id"]
        )

    feature_df = feature_df.dropna(subset=["label", "speaker_id"])
    feature_df["label"] = feature_df["label"].astype(int)

    # Get feature column names
    feature_cols = AudioFeatureExtractor.get_feature_columns(feature_df)
    logger.info(f"Feature matrix: {feature_df.shape[0]} rows × {len(feature_cols)} features")

    # ----------------------------------------------------------------
    # 4. Speaker-level train / val / test split
    # ----------------------------------------------------------------
    logger.info("\n--- Step 3: Speaker-level splitting ---")
    from sklearn.model_selection import StratifiedGroupKFold

    model_df = feature_df.dropna(subset=["label", "speaker_id"]).copy()

    speakers = model_df.groupby("speaker_id")["label"].first()
    speaker_ids_arr = speakers.index.values
    speaker_labels = speakers.values

    # First split: 80% train+val, 20% test
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(
        sgkf_outer.split(speaker_ids_arr, speaker_labels, speaker_ids_arr)
    )

    trainval_speakers = set(speaker_ids_arr[trainval_idx])
    test_speakers = set(speaker_ids_arr[test_idx])

    # Second split: from trainval, take ~20% for val (=> ~64% train, ~16% val, ~20% test)
    trainval_labels = speaker_labels[trainval_idx]
    trainval_ids = speaker_ids_arr[trainval_idx]
    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_sub_idx, val_sub_idx = next(
        sgkf_inner.split(trainval_ids, trainval_labels, trainval_ids)
    )
    train_speakers = set(trainval_ids[train_sub_idx])
    val_speakers = set(trainval_ids[val_sub_idx])

    # Verify no overlap
    assert len(train_speakers & val_speakers) == 0, "Train-val speaker overlap!"
    assert len(train_speakers & test_speakers) == 0, "Train-test speaker overlap!"
    assert len(val_speakers & test_speakers) == 0, "Val-test speaker overlap!"

    # Assign splits
    model_df["split"] = "unknown"
    model_df.loc[model_df["speaker_id"].isin(train_speakers), "split"] = "train"
    model_df.loc[model_df["speaker_id"].isin(val_speakers), "split"] = "val"
    model_df.loc[model_df["speaker_id"].isin(test_speakers), "split"] = "test"

    train_df = model_df[model_df["split"] == "train"]
    val_df = model_df[model_df["split"] == "val"]
    test_df = model_df[model_df["split"] == "test"]

    logger.info(
        f"Split sizes — Train: {len(train_df)} recordings "
        f"({train_df['speaker_id'].nunique()} speakers), "
        f"Val: {len(val_df)} ({val_df['speaker_id'].nunique()} speakers), "
        f"Test: {len(test_df)} ({test_df['speaker_id'].nunique()} speakers)"
    )
    logger.info(
        f"Train labels: {dict(train_df['label'].value_counts().sort_index())}"
    )
    logger.info(
        f"Val labels:   {dict(val_df['label'].value_counts().sort_index())}"
    )
    logger.info(
        f"Test labels:  {dict(test_df['label'].value_counts().sort_index())}"
    )

    # Prepare matrices
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # ----------------------------------------------------------------
    # 5. Train all models
    # ----------------------------------------------------------------
    logger.info("\n--- Step 4: Training models ---")
    trainer = ModelTrainer(cfg)
    val_results = trainer.train(X_train, y_train, X_val, y_val)
    logger.info("\nValidation Results:")
    logger.info(
        val_results[
            ["model", "balanced_accuracy", "roc_auc", "f1_dementia",
             "sensitivity_recall_dementia", "specificity"]
        ].to_string(index=False)
    )

    # ----------------------------------------------------------------
    # 6. Evaluate best model on test set
    # ----------------------------------------------------------------
    logger.info("\n--- Step 5: Evaluating best model on test set ---")
    best_name, best_model = trainer.get_best_model(val_results)
    logger.info(f"Selected model: {best_name}")

    evaluator = ModelEvaluator(cfg)
    test_metrics = evaluator.evaluate_test(
        best_model, X_test, y_test, model_name=best_name
    )

    # ----------------------------------------------------------------
    # 7. Generate comparison plots
    # ----------------------------------------------------------------
    logger.info("\n--- Step 6: Generating plots ---")
    evaluator.plot_model_comparison(val_results)

    # ----------------------------------------------------------------
    # 8. Permutation importance
    # ----------------------------------------------------------------
    logger.info("\n--- Step 7: Permutation importance ---")
    importance_df = evaluator.compute_permutation_importance(
        best_model, X_val, y_val, feature_cols, top_n=20
    )

    # ----------------------------------------------------------------
    # 9. Save model and feature schema
    # ----------------------------------------------------------------
    logger.info("\n--- Step 8: Saving artifacts ---")
    import joblib

    models_dir = resolve_path(cfg.paths.saved_models)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Feature columns
    feat_cols_path = models_dir / "feature_columns.json"
    with open(feat_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature columns to {feat_cols_path}")

    # Run config
    import sklearn
    import librosa

    run_config = {
        "selected_model": best_name,
        "positive_class": 1,
        "negative_class": 0,
        "positive_label_name": "dementia",
        "negative_label_name": "control",
        "sample_rate": cfg.audio.target_sample_rate,
        "seed": seed,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_speakers": train_df["speaker_id"].nunique(),
        "n_val_speakers": val_df["speaker_id"].nunique(),
        "n_test_speakers": test_df["speaker_id"].nunique(),
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_f1": test_metrics["f1_dementia"],
        "sklearn_version": sklearn.__version__,
        "librosa_version": librosa.__version__,
    }
    config_path = models_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    logger.info(f"Best model: {best_name}")
    logger.info(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1 (dementia): {test_metrics['f1_dementia']:.4f}")
    logger.info(f"\nSaved artifacts:")
    logger.info(f"  Model:            {model_path}")
    logger.info(f"  Feature columns:  {feat_cols_path}")
    logger.info(f"  Run config:       {config_path}")
    logger.info(f"  Metrics:          {evaluator.metrics_dir}")
    logger.info(f"  Plots:            {evaluator.plots_dir}")


if __name__ == "__main__":
    main()
