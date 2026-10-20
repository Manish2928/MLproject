"""
Model training pipeline for DementiaVoiceAI.

Trains and compares multiple classical ML models for binary classification
(dementia vs. control) using extracted acoustic features.

Models include: Logistic Regression, SVM (linear and RBF), Random Forest,
Extra Trees, Gradient Boosting (Histogram and optionally XGBoost).
Each model is wrapped in a sklearn Pipeline with imputation and scaling.

Inspired by demo.ipynb cell 33, adapted for config-driven training.

Usage:
    from src.training.trainer import ModelTrainer
    trainer = ModelTrainer(cfg)
    results = trainer.train(X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.utils.config import DotDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_probabilities(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    prob : array-like
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names to values.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_dementia": precision_score(y_true, pred, zero_division=0),
        "sensitivity_recall_dementia": recall_score(
            y_true, pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1_dementia": f1_score(y_true, pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true, prob)
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "pr_auc": (
            average_precision_score(y_true, prob)
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _get_scaler(scaler_name: str):
    """Return a scaler instance based on config name."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    cls = scalers.get(scaler_name)
    return cls() if cls else None


def build_model_zoo(cfg: DotDict, seed: int = 42) -> Dict[str, Pipeline]:
    """Build a dictionary of named sklearn Pipelines.

    Each pipeline: Imputer → Scaler → Classifier.

    Parameters
    ----------
    cfg : DotDict
        Project configuration (uses ``cfg.training.baseline.*``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Pipeline]
        Mapping of model name → fitted pipeline.
    """
    scaler_name = cfg.training.baseline.get("scaler", "standard")
    class_weight = cfg.training.baseline.get("class_weight", "balanced")

    def _pipe(model) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        scaler = _get_scaler(scaler_name)
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", model))
        return Pipeline(steps)

    models = {
        "Logistic Regression": _pipe(
            LogisticRegression(
                max_iter=2000,
                class_weight=class_weight,
                random_state=seed,
                solver="lbfgs",
            )
        ),
        "Linear SVM": _pipe(
            SVC(
                kernel="linear",
                class_weight=class_weight,
                probability=True,
                random_state=seed,
            )
        ),
        "RBF SVM": _pipe(
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight=class_weight,
                probability=True,
                random_state=seed,
            )
        ),
        "Random Forest": _pipe(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "Extra Trees": _pipe(
            ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": _pipe(
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.06,
                random_state=seed,
            )
        ),
        "Hist Gradient Boosting": _pipe(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.06,
                l2_regularization=1.0,
                random_state=seed,
            )
        ),
    }

    # Optional: XGBoost if installed
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = _pipe(
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )
        )
    except ImportError:
        logger.info("XGBoost not installed — skipping XGBoost model.")

    return models


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Trains and compares multiple ML models.

    Parameters
    ----------
    cfg : DotDict
        Project configuration.
    """

    def __init__(self, cfg: DotDict) -> None:
        self.cfg = cfg
        self.seed = cfg.project.seed
        self.threshold = cfg.evaluation.get("threshold", 0.5)
        self.models = build_model_zoo(cfg, self.seed)
        self.fitted_models: Dict[str, Pipeline] = {}
        self.val_probabilities: Dict[str, np.ndarray] = {}

    def train(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pd.DataFrame,
        y_val: np.ndarray,
    ) -> pd.DataFrame:
        """Train all models and evaluate on validation set.

        Parameters
        ----------
        X_train, y_train : array-like
            Training features and labels.
        X_val, y_val : array-like
            Validation features and labels.

        Returns
        -------
        pd.DataFrame
            Validation metrics per model, sorted by balanced_accuracy.
        """
        val_rows = []

        for name, estimator in self.models.items():
            logger.info(f"Training: {name}")
            estimator.fit(X_train, y_train)

            # Get probabilities
            if hasattr(estimator, "predict_proba"):
                prob = estimator.predict_proba(X_val)[:, 1]
            else:
                score = estimator.decision_function(X_val)
                prob = (score - score.min()) / (
                    score.max() - score.min() + 1e-9
                )

            metrics = evaluate_probabilities(y_val, prob, self.threshold)
            metrics["model"] = name
            val_rows.append(metrics)
            self.fitted_models[name] = estimator
            self.val_probabilities[name] = prob

            logger.info(
                f"  {name}: balanced_acc={metrics['balanced_accuracy']:.4f}, "
                f"roc_auc={metrics['roc_auc']:.4f}, "
                f"f1={metrics['f1_dementia']:.4f}"
            )

        results = (
            pd.DataFrame(val_rows)
            .sort_values(
                ["balanced_accuracy", "roc_auc"], ascending=False
            )
            .reset_index(drop=True)
        )

        logger.info(
            f"\nBest model on validation: "
            f"{results.iloc[0]['model']} "
            f"(balanced_acc={results.iloc[0]['balanced_accuracy']:.4f})"
        )

        return results

    def get_best_model(
        self, val_results: Optional[pd.DataFrame] = None
    ) -> Tuple[str, Pipeline]:
        """Return the best model based on validation results.

        Parameters
        ----------
        val_results : pd.DataFrame | None
            Validation results from train(). If None, selects first fitted.

        Returns
        -------
        tuple[str, Pipeline]
            (model_name, fitted_pipeline)
        """
        if val_results is not None:
            best_name = val_results.iloc[0]["model"]
        else:
            best_name = next(iter(self.fitted_models))
        return best_name, self.fitted_models[best_name]
