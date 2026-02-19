"""
Comprehensive credit model evaluation and validation.

Implements standard credit risk model evaluation metrics including Gini,
KS statistic, PSI, calibration analysis, and decile-level performance.
Generates SR 11-7 aligned validation reports.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


class CreditModelEvaluator:
    """Evaluator for credit risk models with regulatory-grade metrics and reporting.

    Computes standard credit risk validation metrics (Gini, KS, PSI),
    generates visualization charts, and produces comprehensive
    validation reports aligned with SR 11-7 expectations.
    """

    def __init__(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.predictions = model.predict(X_test)
        self.metrics: dict[str, float] = {}

    def compute_gini(self, y_true: Optional[pd.Series] = None, y_pred: Optional[np.ndarray] = None) -> float:
        """Compute the Gini coefficient (2 * AUC - 1).

        The Gini coefficient measures the model's ability to discriminate
        between defaults and non-defaults. Values range from 0 (random)
        to 1 (perfect separation).

        Args:
            y_true: True labels. Defaults to test set.
            y_pred: Predicted probabilities. Defaults to test predictions.

        Returns:
            Gini coefficient.
        """
        y_true = y_true if y_true is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.predictions

        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        self.metrics["gini"] = gini
        self.metrics["auc"] = auc
        logger.info("Gini: %.4f (AUC: %.4f)", gini, auc)
        return gini

    def compute_ks_statistic(self, y_true: Optional[pd.Series] = None, y_pred: Optional[np.ndarray] = None) -> float:
        """Compute the Kolmogorov-Smirnov statistic.

        KS measures the maximum separation between the cumulative
        distributions of defaults and non-defaults. Higher values
        indicate better model discrimination.

        Args:
            y_true: True labels.
            y_pred: Predicted probabilities.

        Returns:
            KS statistic.
        """
        y_true = y_true if y_true is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.predictions

        df = pd.DataFrame({"pred": y_pred, "actual": y_true})
        df = df.sort_values("pred", ascending=False).reset_index(drop=True)

        total_bads = df["actual"].sum()
        total_goods = len(df) - total_bads

        df["cum_bads"] = df["actual"].cumsum() / max(total_bads, 1)
        df["cum_goods"] = (1 - df["actual"]).cumsum() / max(total_goods, 1)

        ks = (df["cum_bads"] - df["cum_goods"]).abs().max()
        self.metrics["ks_statistic"] = ks
        logger.info("KS Statistic: %.4f", ks)
        return ks

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute the Population Stability Index.

        PSI measures how much the score distribution has shifted between
        two populations (typically development vs validation). PSI < 0.1
        indicates no shift, 0.1-0.25 moderate shift, > 0.25 significant shift.

        Args:
            expected: Score distribution from the development/training set.
            actual: Score distribution from the validation/monitoring set.
            n_bins: Number of bins for distribution comparison.

        Returns:
            PSI value.
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_pct = (expected_counts / max(expected_counts.sum(), 1)).clip(min=0.0001)
        actual_pct = (actual_counts / max(actual_counts.sum(), 1)).clip(min=0.0001)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        self.metrics["psi"] = psi
        logger.info("PSI: %.4f", psi)
        return psi

    def plot_roc_auc(self, output_path: Optional[str] = None) -> str:
        """Generate ROC curve with AUC annotation.

        Args:
            output_path: Path to save the chart. Defaults to 'roc_curve.png'.

        Returns:
            Path to the saved chart.
        """
        output_path = output_path or "roc_curve.png"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        fpr, tpr, _ = roc_curve(self.y_test, self.predictions)
        auc = roc_auc_score(self.y_test, self.predictions)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#1976D2", linewidth=2, label=f"Model (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#757575", linewidth=1, linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("ROC curve saved to %s", output_path)
        return output_path

    def plot_ks_chart(self, output_path: Optional[str] = None) -> str:
        """Generate KS chart showing cumulative distribution separation.

        Args:
            output_path: Path to save the chart.

        Returns:
            Path to the saved chart.
        """
        output_path = output_path or "ks_chart.png"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        df = pd.DataFrame({"pred": self.predictions, "actual": self.y_test.values})
        df = df.sort_values("pred", ascending=False).reset_index(drop=True)

        total_bads = df["actual"].sum()
        total_goods = len(df) - total_bads

        df["cum_bads"] = df["actual"].cumsum() / max(total_bads, 1)
        df["cum_goods"] = (1 - df["actual"]).cumsum() / max(total_goods, 1)
        df["pct_population"] = np.arange(1, len(df) + 1) / len(df)

        ks_idx = (df["cum_bads"] - df["cum_goods"]).abs().idxmax()
        ks_value = (df["cum_bads"] - df["cum_goods"]).abs().max()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df["pct_population"], df["cum_bads"], color="#D32F2F", linewidth=2, label="Cumulative Bad Rate")
        ax.plot(df["pct_population"], df["cum_goods"], color="#1976D2", linewidth=2, label="Cumulative Good Rate")
        ax.axvline(x=df["pct_population"].iloc[ks_idx], color="#388E3C", linestyle="--", alpha=0.7)
        ax.annotate(f"KS = {ks_value:.4f}", xy=(df["pct_population"].iloc[ks_idx], 0.5),
                    fontsize=12, fontweight="bold", color="#388E3C")
        ax.set_xlabel("Population Proportion")
        ax.set_ylabel("Cumulative Rate")
        ax.set_title("KS Chart")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("KS chart saved to %s", output_path)
        return output_path

    def plot_calibration(self, output_path: Optional[str] = None, n_bins: int = 10) -> str:
        """Generate calibration plot comparing predicted vs actual default rates.

        Args:
            output_path: Path to save the chart.
            n_bins: Number of calibration bins.

        Returns:
            Path to the saved chart.
        """
        output_path = output_path or "calibration_curve.png"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.predictions, n_bins=n_bins, strategy="quantile",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, marker="o",
                linewidth=2, color="#1976D2", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#757575", label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Calibration curve saved to %s", output_path)
        return output_path

    def plot_score_distribution(self, output_path: Optional[str] = None) -> str:
        """Generate score distribution plot by default status.

        Args:
            output_path: Path to save the chart.

        Returns:
            Path to the saved chart.
        """
        output_path = output_path or "score_distribution.png"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        defaults = self.predictions[self.y_test == 1]
        non_defaults = self.predictions[self.y_test == 0]

        ax.hist(non_defaults, bins=50, alpha=0.6, color="#1976D2", label="Non-Default", density=True)
        ax.hist(defaults, bins=50, alpha=0.6, color="#D32F2F", label="Default", density=True)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution by Default Status")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Score distribution saved to %s", output_path)
        return output_path

    def compute_decile_analysis(self) -> pd.DataFrame:
        """Compute decile-wise default rates and lift.

        Ranks predictions into deciles and computes the actual default
        rate, cumulative capture rate, and lift for each decile.

        Returns:
            DataFrame with decile-level performance metrics.
        """
        df = pd.DataFrame({
            "predicted": self.predictions,
            "actual": self.y_test.values,
        })
        df["decile"] = pd.qcut(df["predicted"], q=10, labels=False, duplicates="drop") + 1

        decile_stats = df.groupby("decile").agg(
            count=("actual", "count"),
            num_defaults=("actual", "sum"),
            avg_predicted=("predicted", "mean"),
        ).reset_index()

        decile_stats["default_rate"] = decile_stats["num_defaults"] / decile_stats["count"]
        total_defaults = decile_stats["num_defaults"].sum()
        decile_stats["capture_rate"] = decile_stats["num_defaults"] / max(total_defaults, 1)
        decile_stats["cum_capture"] = decile_stats["capture_rate"].cumsum()

        overall_default_rate = df["actual"].mean()
        decile_stats["lift"] = decile_stats["default_rate"] / max(overall_default_rate, 0.001)

        decile_stats = decile_stats.sort_values("decile", ascending=False)
        self.metrics["decile_analysis"] = decile_stats.to_dict("records")

        logger.info("Decile analysis complete:\n%s", decile_stats.to_string())
        return decile_stats

    def generate_validation_report(
        self,
        output_dir: str = "reports",
    ) -> dict[str, Any]:
        """Generate comprehensive model validation report aligned with SR 11-7.

        Computes all standard validation metrics, generates visualization
        charts, and compiles a structured report suitable for model risk
        management review.

        Args:
            output_dir: Directory to save report artifacts.

        Returns:
            Validation report as a dictionary.
        """
        os.makedirs(output_dir, exist_ok=True)

        gini = self.compute_gini()
        ks = self.compute_ks_statistic()

        if self.X_train is not None and self.y_train is not None:
            train_preds = self.model.predict(self.X_train)
            psi = self.compute_psi(train_preds, self.predictions)
        else:
            psi = None

        decile_df = self.compute_decile_analysis()

        self.plot_roc_auc(os.path.join(output_dir, "roc_curve.png"))
        self.plot_ks_chart(os.path.join(output_dir, "ks_chart.png"))
        self.plot_calibration(os.path.join(output_dir, "calibration_curve.png"))
        self.plot_score_distribution(os.path.join(output_dir, "score_distribution.png"))

        eval_config = self.config_thresholds()
        ks_pass = ks >= eval_config["ks_threshold"]
        gini_pass = gini >= eval_config["gini_threshold"]
        psi_pass = psi is None or psi <= eval_config["psi_threshold"]

        report = {
            "validation_summary": {
                "overall_result": "PASS" if (ks_pass and gini_pass and psi_pass) else "REQUIRES REVIEW",
                "validation_date": pd.Timestamp.now().isoformat(),
            },
            "discriminatory_power": {
                "auc_roc": self.metrics.get("auc"),
                "gini": gini,
                "gini_threshold": eval_config["gini_threshold"],
                "gini_result": "PASS" if gini_pass else "FAIL",
                "ks_statistic": ks,
                "ks_threshold": eval_config["ks_threshold"],
                "ks_result": "PASS" if ks_pass else "FAIL",
            },
            "stability": {
                "psi": psi,
                "psi_threshold": eval_config["psi_threshold"],
                "psi_result": "PASS" if psi_pass else "FAIL",
            },
            "decile_analysis": decile_df.to_dict("records"),
            "charts": {
                "roc_curve": os.path.join(output_dir, "roc_curve.png"),
                "ks_chart": os.path.join(output_dir, "ks_chart.png"),
                "calibration_curve": os.path.join(output_dir, "calibration_curve.png"),
                "score_distribution": os.path.join(output_dir, "score_distribution.png"),
            },
        }

        logger.info("Validation report generated: %s", report["validation_summary"]["overall_result"])
        return report

    def config_thresholds(self) -> dict[str, float]:
        """Return evaluation thresholds from config or defaults."""
        try:
            with open("configs/config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
        except (FileNotFoundError, Exception):
            config = {}

        eval_config = config.get("evaluation", {})
        return {
            "ks_threshold": eval_config.get("ks_threshold", 0.3),
            "gini_threshold": eval_config.get("gini_threshold", 0.4),
            "psi_threshold": eval_config.get("psi_threshold", 0.25),
        }
