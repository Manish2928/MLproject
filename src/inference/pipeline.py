"""
End-to-end inference pipeline for DementiaVoiceAI.

Takes a raw audio file and produces a prediction with confidence score.
Wraps: audio loading → feature extraction → model prediction.

Supports loading a saved model from disk or using a model in memory.

Usage:
    from src.inference.pipeline import InferencePipeline
    pipeline = InferencePipeline.from_saved(cfg, "saved_models/best_model.joblib")
    result = pipeline.predict("path/to/audio.wav")
    print(result)
    # {'file': 'audio.wav', 'predicted_class': 'Control',
    #  'confidence': 0.87, 'confidence_band': 'High', ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import DotDict, resolve_path
from src.utils.logger import get_logger
from src.features.audio_features import AudioFeatureExtractor

logger = get_logger(__name__)


class InferencePipeline:
    """End-to-end inference: audio file → prediction.

    Parameters
    ----------
    cfg : DotDict
        Project configuration.
    model : sklearn Pipeline or estimator
        Trained classification model.
    feature_columns : list[str]
        Ordered list of feature columns the model expects.
    threshold : float
        Classification threshold.
    """

    def __init__(
        self,
        cfg: DotDict,
        model: Any,
        feature_columns: List[str],
        threshold: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.extractor = AudioFeatureExtractor(cfg)

        # Confidence bands from config
        bands = cfg.inference.get("confidence_bands", {})
        self.high_threshold = bands.get("high", 0.85)
        self.medium_threshold = bands.get("medium", 0.65)
        self.disclaimer = cfg.inference.get(
            "disclaimer",
            "This is an academic AI/ML model and is not a medical diagnostic tool."
        )

    @classmethod
    def from_saved(
        cls,
        cfg: DotDict,
        model_path: str | Path,
        feature_columns_path: Optional[str | Path] = None,
    ) -> "InferencePipeline":
        """Load a saved model and feature schema from disk.

        Parameters
        ----------
        cfg : DotDict
            Project configuration.
        model_path : str | Path
            Path to the saved model (.joblib file).
        feature_columns_path : str | Path | None
            Path to feature_columns.json. If None, looks next to model file.

        Returns
        -------
        InferencePipeline
        """
        import joblib

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = resolve_path(model_path)

        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load feature columns
        if feature_columns_path is None:
            feature_columns_path = model_path.parent / "feature_columns.json"
        else:
            feature_columns_path = Path(feature_columns_path)
            if not feature_columns_path.is_absolute():
                feature_columns_path = resolve_path(feature_columns_path)

        with open(feature_columns_path, "r") as f:
            feature_columns = json.load(f)
        logger.info(f"Loaded {len(feature_columns)} feature columns")

        threshold = cfg.evaluation.get("threshold", 0.5)

        return cls(cfg, model, feature_columns, threshold)

    def predict(self, audio_path: str | Path) -> Dict[str, Any]:
        """Run inference on a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        dict
            Prediction result containing:
            - file: filename
            - predicted_class: 'Dementia' or 'Control'
            - confidence: probability score (0-1)
            - confidence_band: 'High', 'Medium', or 'Low'
            - raw_probability: model's raw dementia-class probability
            - threshold: classification threshold used
            - disclaimer: research warning
        """
        audio_path = Path(audio_path).resolve()

        # 1. Extract features
        raw_features = self.extractor.extract(str(audio_path))
        feature_dict = {f"feat_{k}": v for k, v in raw_features.items()}

        # 2. Create a DataFrame with the exact feature columns the model expects
        row = pd.DataFrame([feature_dict]).reindex(
            columns=self.feature_columns
        )

        # 3. Get prediction probability
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(row)[:, 1][0])
        else:
            score = float(self.model.decision_function(row)[0])
            prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid

        # 4. Apply threshold
        pred = int(prob >= self.threshold)
        predicted_class = "Dementia" if pred == 1 else "Control"

        # 5. Confidence band
        confidence = prob if pred == 1 else (1.0 - prob)
        if confidence >= self.high_threshold:
            band = "High"
        elif confidence >= self.medium_threshold:
            band = "Medium"
        else:
            band = "Low"

        return {
            "file": audio_path.name,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_band": band,
            "raw_probability_dementia": round(prob, 4),
            "threshold": self.threshold,
            "disclaimer": self.disclaimer,
        }

    def predict_batch(
        self, audio_paths: List[str | Path]
    ) -> pd.DataFrame:
        """Run inference on multiple audio files.

        Parameters
        ----------
        audio_paths : list
            Paths to audio files.

        Returns
        -------
        pd.DataFrame
            One row per prediction.
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file": Path(path).name,
                    "predicted_class": "Error",
                    "confidence": 0.0,
                    "confidence_band": "N/A",
                    "raw_probability_dementia": 0.0,
                    "threshold": self.threshold,
                    "disclaimer": str(e)[:200],
                })
        return pd.DataFrame(results)
