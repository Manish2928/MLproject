"""
Model evaluation and reporting for DementiaVoiceAI.

Generates:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Model comparison bar charts
- Permutation importance plots
- CSV metrics files

Inspired by demo.ipynb cells 35 and 37, adapted for the project's
results/ directory structure.

Usage:
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(cfg)
    evaluator.evaluate_test(best_model, X_test, y_test, model_name="RBF SVM")
    evaluator.plot_model_comparison(val_results)
    evaluator.compute_permutation_importance(model, X_val, y_val, feature_cols)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.inspection import permutation_importance

from src.utils.config import DotDict, resolve_path
from src.utils.logger import get_logger
from src.training.trainer import evaluate_probabilities

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates trained models and generates reports and plots.

    Parameters
    ----------
    cfg : DotDict
        Project configuration.
    """

    def __init__(self, cfg: DotDict) -> None:
        self.cfg = cfg
        self.seed = cfg.project.seed
        self.threshold = cfg.evaluation.get("threshold", 0.5)

        # Output directories
        self.metrics_dir = resolve_path(cfg.paths.reports) / "metrics"
        self.plots_dir = resolve_path(cfg.paths.reports) / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_test(
        self,
        model,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray,
        model_name: str = "Best Model",
    ) -> Dict[str, float]:
        """Evaluate a trained model on the held-out test set.

        Parameters
        ----------
        model : sklearn Pipeline or estimator
            The trained model.
        X_test, y_test : array-like
            Test features and labels.
        model_name : str
            Name of the model (for titles and filenames).

        Returns
        -------
        dict[str, float]
            Test metrics.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get probabilities
        if hasattr(model, "predict_proba"):
            test_prob = model.predict_proba(X_test)[:, 1]
        else:
            test_score = model.decision_function(X_test)
            test_prob = (test_score - test_score.min()) / (
                test_score.max() - test_score.min() + 1e-9
            )

        test_metrics = evaluate_probabilities(
            y_test, test_prob, self.threshold
        )
        test_pred = (test_prob >= self.threshold).astype(int)

        # Print classification report
        report = classification_report(
            y_test,
            test_pred,
            target_names=["No dementia / control", "Dementia"],
            zero_division=0,
        )
        logger.info(f"\nTest metrics for {model_name}:\n{report}")

        # Save metrics CSV
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df["model"] = model_name
        metrics_path = self.metrics_dir / "test_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved test metrics to {metrics_path}")

        # ----- Confusion Matrix -----
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Control", "Predicted Dementia"],
            yticklabels=["Actual Control", "Actual Dementia"],
            ax=ax,
        )
        ax.set_title(f"Test Confusion Matrix — {model_name}")
        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "confusion_matrix_test.png", dpi=150
        )
        plt.close(fig)

        # ----- ROC Curve -----
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, test_prob, ax=ax)
        ax.set_title(f"Test ROC Curve — {model_name}")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "roc_curve_test.png", dpi=150)
        plt.close(fig)

        # ----- Precision-Recall Curve -----
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, test_prob, ax=ax)
        ax.set_title(f"Test Precision-Recall Curve — {model_name}")
        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "precision_recall_curve_test.png", dpi=150
        )
        plt.close(fig)

        logger.info(f"Saved test plots to {self.plots_dir}")
        return test_metrics

    def plot_model_comparison(
        self, val_results: pd.DataFrame
    ) -> None:
        """Generate a bar chart comparing validation metrics across models.

        Parameters
        ----------
        val_results : pd.DataFrame
            Validation results from ModelTrainer.train().
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Save CSV
        csv_path = self.metrics_dir / "validation_model_comparison.csv"
        val_results.to_csv(csv_path, index=False)

        # Bar chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, metric, title in zip(
            axes,
            ["balanced_accuracy", "roc_auc", "f1_dementia"],
            ["Balanced Accuracy", "ROC AUC", "F1 Score (Dementia)"],
        ):
            data = val_results.sort_values(metric, ascending=True)
            sns.barplot(
                data=data, x=metric, y="model", ax=ax, palette="viridis"
            )
            ax.set_xlim(0, 1)
            ax.set_title(f"Validation {title}")
            ax.set_xlabel(title)
            ax.set_ylabel("")

        fig.suptitle("Model Comparison — Validation Set", fontsize=14)
        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "model_comparison_validation.png", dpi=150
        )
        plt.close(fig)
        logger.info(f"Saved model comparison plot to {self.plots_dir}")

    def compute_permutation_importance(
        self,
        model,
        X_val: np.ndarray | pd.DataFrame,
        y_val: np.ndarray,
        feature_names: List[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compute and plot permutation importance.

        Parameters
        ----------
        model : sklearn estimator
            Trained model.
        X_val, y_val : array-like
            Validation data.
        feature_names : list[str]
            Names of the features.
        top_n : int
            Number of top features to plot.

        Returns
        -------
        pd.DataFrame
            Permutation importance results.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("Computing permutation importance (this may take a moment)...")
        perm = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=self.seed,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        # Save CSV
        imp_path = self.metrics_dir / "permutation_importance.csv"
        importance_df.to_csv(imp_path, index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_imp = importance_df.head(top_n).sort_values("importance_mean")
        ax.barh(
            top_imp["feature"],
            top_imp["importance_mean"],
            xerr=top_imp["importance_std"],
            color="steelblue",
        )
        ax.set_title(f"Top {top_n} Feature Importances (Permutation)")
        ax.set_xlabel("Decrease in Balanced Accuracy When Shuffled")
        fig.tight_layout()
        fig.savefig(
            self.plots_dir / "permutation_importance.png", dpi=150
        )
        plt.close(fig)

        logger.info(f"Saved permutation importance to {imp_path}")
        return importance_df
