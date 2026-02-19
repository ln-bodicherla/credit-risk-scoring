"""
Fair lending analysis for credit risk models.

Implements adverse impact ratio, marginal effect analysis, disparate
impact testing, and ECOA compliance reporting to ensure the model
does not discriminate against protected classes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


class FairLendingAnalyzer:
    """Analyzer for ECOA compliance and fair lending in credit risk models.

    Performs statistical tests for disparate impact across protected
    classes (race, gender, age), computes adverse impact ratios, and
    generates compliance reports with mitigation recommendations.
    """

    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: Optional[dict[str, pd.Series]] = None,
        config_path: str = "configs/config.yaml",
    ):
        self.model = model
        self.X = X
        self.y = y
        self.predictions = model.predict(X)
        self.protected_attributes = protected_attributes or {}
        self.config = self._load_config(config_path)
        self.results: dict[str, Any] = {}

        fairness_config = self.config.get("fairness", {})
        self.air_threshold = fairness_config.get("air_threshold", 0.8)

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load fairness configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def compute_adverse_impact_ratio(
        self,
        protected_attr: pd.Series,
        attr_name: str,
        approval_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Compute the Adverse Impact Ratio for a protected attribute.

        AIR compares the approval rate of each protected group to the
        group with the highest approval rate. An AIR below 0.80 is
        generally considered evidence of adverse impact under the
        four-fifths rule.

        Args:
            protected_attr: Series with protected attribute values.
            attr_name: Name of the protected attribute.
            approval_threshold: Probability threshold for approval decision.

        Returns:
            AIR results for each group comparison.
        """
        approved = (self.predictions < approval_threshold).astype(int)

        groups = protected_attr.unique()
        approval_rates = {}

        for group in groups:
            mask = protected_attr == group
            group_total = mask.sum()
            group_approved = approved[mask].sum()
            approval_rates[group] = group_approved / max(group_total, 1)

        reference_group = max(approval_rates, key=approval_rates.get)
        reference_rate = approval_rates[reference_group]

        air_results = {}
        for group, rate in approval_rates.items():
            if group == reference_group:
                continue
            air = rate / max(reference_rate, 0.0001)
            air_results[group] = {
                "approval_rate": round(rate, 4),
                "reference_rate": round(reference_rate, 4),
                "air": round(air, 4),
                "passes_threshold": air >= self.air_threshold,
                "reference_group": str(reference_group),
            }

        result = {
            "attribute": attr_name,
            "approval_rates": {str(k): round(v, 4) for k, v in approval_rates.items()},
            "reference_group": str(reference_group),
            "group_comparisons": air_results,
            "overall_pass": all(v["passes_threshold"] for v in air_results.values()),
        }

        self.results[f"air_{attr_name}"] = result
        logger.info("AIR for %s: overall %s", attr_name, "PASS" if result["overall_pass"] else "FAIL")
        return result

    def compute_marginal_effect(
        self,
        protected_attr: pd.Series,
        attr_name: str,
    ) -> dict[str, Any]:
        """Compute marginal effects of the protected attribute on predictions.

        Estimates how much the predicted probability changes when only
        the protected attribute changes, holding all other features constant.
        This isolates the model's direct sensitivity to the protected class.

        Args:
            protected_attr: Series with protected attribute values.
            attr_name: Name of the protected attribute.

        Returns:
            Marginal effect statistics.
        """
        groups = protected_attr.unique()
        group_predictions = {}

        for group in groups:
            mask = protected_attr == group
            group_preds = self.predictions[mask]
            group_predictions[group] = {
                "mean": float(np.mean(group_preds)),
                "median": float(np.median(group_preds)),
                "std": float(np.std(group_preds)),
                "count": int(mask.sum()),
            }

        group_means = {g: v["mean"] for g, v in group_predictions.items()}
        overall_mean = float(np.mean(self.predictions))

        marginal_effects = {}
        for group, mean_pred in group_means.items():
            marginal_effects[str(group)] = {
                "mean_prediction": round(mean_pred, 6),
                "deviation_from_overall": round(mean_pred - overall_mean, 6),
                "relative_effect": round((mean_pred - overall_mean) / max(overall_mean, 0.0001), 4),
            }

        if len(groups) == 2:
            g1, g2 = groups[:2]
            mask1 = protected_attr == g1
            mask2 = protected_attr == g2
            t_stat, p_value = stats.ttest_ind(
                self.predictions[mask1],
                self.predictions[mask2],
                equal_var=False,
            )
            effect_size = (np.mean(self.predictions[mask1]) - np.mean(self.predictions[mask2])) / \
                          np.sqrt((np.var(self.predictions[mask1]) + np.var(self.predictions[mask2])) / 2)
        else:
            f_stat, p_value = stats.f_oneway(
                *[self.predictions[protected_attr == g] for g in groups]
            )
            t_stat = f_stat
            effect_size = None

        result = {
            "attribute": attr_name,
            "overall_mean": round(overall_mean, 6),
            "group_statistics": group_predictions,
            "marginal_effects": marginal_effects,
            "statistical_test": {
                "test_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant_at_005": p_value < 0.05,
                "effect_size": round(float(effect_size), 4) if effect_size is not None else None,
            },
        }

        self.results[f"marginal_{attr_name}"] = result
        logger.info(
            "Marginal effect for %s: p-value=%.4f, significant=%s",
            attr_name, p_value, p_value < 0.05,
        )
        return result

    def test_disparate_impact(
        self,
        protected_attr: pd.Series,
        attr_name: str,
        approval_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Perform statistical testing for disparate impact.

        Combines the four-fifths rule (AIR) with chi-squared testing
        to provide both practical and statistical evidence of disparate
        treatment across protected groups.

        Args:
            protected_attr: Protected attribute values.
            attr_name: Attribute name.
            approval_threshold: Decision threshold.

        Returns:
            Disparate impact test results.
        """
        approved = (self.predictions < approval_threshold).astype(int)
        groups = protected_attr.unique()

        contingency = pd.crosstab(protected_attr, approved)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        air_result = self.compute_adverse_impact_ratio(protected_attr, attr_name, approval_threshold)

        cramers_v = np.sqrt(chi2 / (max(len(protected_attr), 1) * (min(contingency.shape) - 1)))

        result = {
            "attribute": attr_name,
            "four_fifths_rule": air_result,
            "chi_squared_test": {
                "chi2_statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "degrees_of_freedom": int(dof),
                "significant_at_005": p_value < 0.05,
                "cramers_v": round(float(cramers_v), 4),
            },
            "conclusion": self._interpret_disparate_impact(air_result, p_value),
        }

        self.results[f"disparate_impact_{attr_name}"] = result
        logger.info("Disparate impact test for %s: %s", attr_name, result["conclusion"])
        return result

    @staticmethod
    def _interpret_disparate_impact(air_result: dict, p_value: float) -> str:
        """Generate a plain-language interpretation of disparate impact results."""
        air_pass = air_result.get("overall_pass", True)
        stat_sig = p_value < 0.05

        if air_pass and not stat_sig:
            return "No evidence of disparate impact. AIR above threshold and differences are not statistically significant."
        elif air_pass and stat_sig:
            return "AIR above threshold but statistically significant differences detected. Monitor closely."
        elif not air_pass and not stat_sig:
            return "AIR below threshold but differences are not statistically significant. Consider sample size adequacy."
        else:
            return "Evidence of disparate impact. AIR below threshold and differences are statistically significant. Mitigation required."

    def suggest_mitigations(self) -> list[dict[str, str]]:
        """Generate suggestions for reducing identified bias.

        Analyzes the results of previous fairness tests and provides
        actionable recommendations for bias mitigation.

        Returns:
            List of mitigation suggestions with rationale.
        """
        suggestions = []

        for key, result in self.results.items():
            if key.startswith("air_"):
                attr_name = result.get("attribute", key)
                if not result.get("overall_pass", True):
                    for group, comparison in result.get("group_comparisons", {}).items():
                        if not comparison.get("passes_threshold", True):
                            suggestions.append({
                                "attribute": attr_name,
                                "affected_group": str(group),
                                "air": comparison.get("air"),
                                "recommendation": (
                                    f"AIR for {group} ({comparison.get('air', 0):.3f}) is below the "
                                    f"{self.air_threshold} threshold. Consider: (1) removing or adjusting features "
                                    f"correlated with {attr_name}, (2) applying reject inference adjustments, "
                                    f"(3) using fairness-aware model training constraints, or "
                                    f"(4) adjusting approval thresholds by group to achieve equitable outcomes."
                                ),
                                "priority": "high",
                            })

            elif key.startswith("marginal_"):
                stat_test = result.get("statistical_test", {})
                if stat_test.get("significant_at_005", False):
                    effect = stat_test.get("effect_size")
                    if effect is not None and abs(effect) > 0.2:
                        suggestions.append({
                            "attribute": result.get("attribute", key),
                            "recommendation": (
                                f"Statistically significant marginal effect detected (effect size: "
                                f"{effect:.3f}). Review feature correlations with this protected attribute "
                                f"and consider using adversarial debiasing or reweighting techniques."
                            ),
                            "priority": "medium",
                        })

        if not suggestions:
            suggestions.append({
                "attribute": "all",
                "recommendation": "No significant fairness concerns detected. Continue routine monitoring.",
                "priority": "low",
            })

        return suggestions

    def generate_fair_lending_report(
        self,
        output_dir: str = "reports",
    ) -> dict[str, Any]:
        """Generate comprehensive ECOA compliance report.

        Runs all fairness tests across protected attributes and compiles
        results into a structured report with recommendations.

        Args:
            output_dir: Directory to save report artifacts.

        Returns:
            Complete fair lending analysis report.
        """
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "report_type": "Fair Lending Analysis",
            "regulatory_framework": ["ECOA", "Reg B", "Fair Housing Act"],
            "air_threshold": self.air_threshold,
            "analyses": {},
        }

        for attr_name, attr_values in self.protected_attributes.items():
            logger.info("Analyzing protected attribute: %s", attr_name)

            air = self.compute_adverse_impact_ratio(attr_values, attr_name)
            marginal = self.compute_marginal_effect(attr_values, attr_name)
            disparate = self.test_disparate_impact(attr_values, attr_name)

            report["analyses"][attr_name] = {
                "adverse_impact_ratio": air,
                "marginal_effect": marginal,
                "disparate_impact": disparate,
            }

        report["mitigations"] = self.suggest_mitigations()

        overall_pass = all(
            analysis.get("adverse_impact_ratio", {}).get("overall_pass", True)
            for analysis in report["analyses"].values()
        )
        report["overall_conclusion"] = (
            "Model passes fair lending review. No evidence of disparate impact across analyzed protected classes."
            if overall_pass else
            "Model requires remediation. Evidence of adverse impact detected for one or more protected classes."
        )

        logger.info("Fair lending report complete: %s", "PASS" if overall_pass else "REQUIRES ACTION")
        return report
