"""
Credit risk model training with LightGBM, Optuna optimization,
logistic regression baseline, probability calibration, and traditional
scorecard generation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class CreditRiskModel:
    """Credit risk model with LightGBM, scorecard generation, and model governance.

    Trains gradient boosting models with monotonic constraints for
    interpretability, supports Optuna-based hyperparameter optimization,
    and generates traditional points-based credit scorecards.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.model: Optional[lgb.Booster] = None
        self.lr_model: Optional[LogisticRegression] = None
        self.calibrated_model = None
        self.feature_names: list[str] = []
        self.best_params: dict[str, Any] = {}
        self.training_metadata: dict[str, Any] = {}
        self.scorecard: Optional[pd.DataFrame] = None

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load model configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> lgb.Booster:
        """Train a LightGBM model with monotonic constraints.

        Monotonic constraints ensure that the model's predictions change
        monotonically with respect to specific features, which is crucial
        for regulatory acceptance of credit risk models.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Optional validation features.
            y_val: Optional validation target.
            params: Optional parameter overrides.

        Returns:
            Trained LightGBM Booster.
        """
        self.feature_names = list(X_train.columns)

        model_config = self.config.get("model", {})
        monotone_constraints = model_config.get("monotone_constraints", [])

        if len(monotone_constraints) < len(self.feature_names):
            monotone_constraints.extend([0] * (len(self.feature_names) - len(monotone_constraints)))
        monotone_constraints = monotone_constraints[:len(self.feature_names)]

        default_params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "max_depth": 6,
            "monotone_constraints": monotone_constraints,
            "verbose": -1,
            "seed": 42,
        }

        if params:
            default_params.update(params)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50, verbose=True),
        ]

        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.training_metadata = {
            "model_type": "LightGBM",
            "num_features": len(self.feature_names),
            "num_iterations": self.model.current_iteration(),
            "best_iteration": self.model.best_iteration,
            "params": default_params,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        train_pred = self.model.predict(X_train)
        train_auc = roc_auc_score(y_train, train_pred)
        logger.info("Training AUC: %.4f", train_auc)
        self.training_metadata["train_auc"] = train_auc

        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info("Validation AUC: %.4f", val_auc)
            self.training_metadata["val_auc"] = val_auc

        return self.model

    def optimize_with_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 200,
        timeout: int = 7200,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """Run Bayesian hyperparameter optimization with Optuna.

        Uses a custom objective function with cross-validation to find
        optimal hyperparameters that maximize AUC while preventing overfitting.

        Args:
            X: Full training features.
            y: Full training target.
            n_trials: Maximum optimization trials.
            timeout: Timeout in seconds.
            cv_folds: Number of CV folds.

        Returns:
            Dictionary of best hyperparameters found.
        """
        optuna_config = self.config.get("optuna", {})
        n_trials = optuna_config.get("n_trials", n_trials)
        timeout = optuna_config.get("timeout", timeout)
        cv_folds = optuna_config.get("cv_folds", cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "verbose": -1,
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            auc_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                preds = model.predict(X_fold_val)
                fold_auc = roc_auc_score(y_fold_val, preds)
                auc_scores.append(fold_auc)

            mean_auc = np.mean(auc_scores)
            return mean_auc

        study = optuna.create_study(
            direction="maximize",
            study_name="credit_risk_optimization",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        logger.info("Best AUC: %.4f", study.best_value)
        logger.info("Best params: %s", json.dumps(self.best_params, indent=2))

        self.train_lightgbm(X, y, params=self.best_params)

        return self.best_params

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> LogisticRegression:
        """Train a logistic regression model as a traditional scorecard baseline.

        Logistic regression with WOE-encoded features is the standard
        approach for credit scorecards and serves as an interpretable baseline.

        Args:
            X_train: WOE-encoded training features.
            y_train: Training target.

        Returns:
            Fitted LogisticRegression model.
        """
        self.lr_model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        self.lr_model.fit(X_train, y_train)

        train_pred = self.lr_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred)
        logger.info("Logistic Regression Training AUC: %.4f", train_auc)

        return self.lr_model

    def calibrate_probabilities(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "sigmoid",
    ) -> Any:
        """Apply Platt scaling for probability calibration.

        Ensures that predicted probabilities are well-calibrated, meaning
        a predicted 10% default probability corresponds to roughly 10%
        actual default rate.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration target.
            method: Calibration method ("sigmoid" for Platt, "isotonic").

        Returns:
            Calibrated classifier.
        """
        if self.lr_model is None:
            raise ValueError("Train logistic regression first before calibrating")

        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.lr_model,
            method=method,
            cv=5,
        )
        self.calibrated_model.fit(X_cal, y_cal)

        logger.info("Probability calibration complete (method=%s)", method)
        return self.calibrated_model

    def create_scorecard(
        self,
        woe_mappings: dict[str, dict[str, float]],
        base_score: int = 600,
        base_odds: int = 50,
        pdo: int = 20,
    ) -> pd.DataFrame:
        """Convert model coefficients into a traditional points-based scorecard.

        The scorecard maps WOE bins to point values using the PDO (points
        to double the odds) scaling framework, which is the industry standard
        for credit scoring.

        Args:
            woe_mappings: WOE values from the preprocessor.
            base_score: Score at the base odds.
            base_odds: Odds of good at the base score.
            pdo: Points to double the odds.

        Returns:
            DataFrame with scorecard points for each feature bin.
        """
        if self.lr_model is None:
            raise ValueError("Train logistic regression first before creating scorecard")

        scorecard_config = self.config.get("scorecard", {})
        base_score = scorecard_config.get("base_score", base_score)
        base_odds = scorecard_config.get("base_odds", base_odds)
        pdo = scorecard_config.get("pdo", pdo)

        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)

        intercept = self.lr_model.intercept_[0]
        coefficients = self.lr_model.coef_[0]

        n_features = len(self.feature_names)
        base_points = offset / n_features
        intercept_points = -(intercept * factor) / n_features

        scorecard_rows = []
        for i, feature in enumerate(self.feature_names):
            coef = coefficients[i]
            raw_feature = feature.replace("_woe", "")

            if raw_feature in woe_mappings:
                for bin_label, woe_val in woe_mappings[raw_feature].items():
                    points = -(coef * woe_val * factor) + base_points + intercept_points
                    scorecard_rows.append({
                        "feature": raw_feature,
                        "bin": bin_label,
                        "woe": round(woe_val, 4),
                        "coefficient": round(coef, 6),
                        "points": round(points, 0),
                    })

        self.scorecard = pd.DataFrame(scorecard_rows)
        logger.info("Scorecard created with %d rows", len(self.scorecard))
        return self.scorecard

    def validate_monotonicity(
        self,
        X: pd.DataFrame,
        feature: str,
        n_points: int = 100,
    ) -> dict[str, Any]:
        """Validate that model predictions are monotonic with respect to a feature.

        Generates a range of feature values while holding other features
        at their median, then checks if predictions consistently increase
        or decrease.

        Args:
            X: Reference dataset for median feature values.
            feature: Feature to test for monotonicity.
            n_points: Number of test points to evaluate.

        Returns:
            Monotonicity validation result with direction and violations.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        median_row = X.median().to_frame().T
        test_df = pd.concat([median_row] * n_points, ignore_index=True)

        feat_min = X[feature].quantile(0.01)
        feat_max = X[feature].quantile(0.99)
        test_values = np.linspace(feat_min, feat_max, n_points)
        test_df[feature] = test_values

        predictions = self.model.predict(test_df)

        diffs = np.diff(predictions)
        increasing_count = np.sum(diffs > 0)
        decreasing_count = np.sum(diffs < 0)

        if increasing_count > decreasing_count:
            expected_direction = "increasing"
            violations = int(decreasing_count)
        else:
            expected_direction = "decreasing"
            violations = int(increasing_count)

        is_monotonic = violations == 0
        result = {
            "feature": feature,
            "is_monotonic": is_monotonic,
            "direction": expected_direction,
            "violations": violations,
            "total_intervals": n_points - 1,
            "violation_rate": violations / (n_points - 1) if n_points > 1 else 0,
        }

        logger.info(
            "Monotonicity check for %s: %s (%s, %d violations)",
            feature, "PASS" if is_monotonic else "FAIL",
            expected_direction, violations,
        )
        return result

    def generate_model_inventory(self) -> dict[str, Any]:
        """Generate model risk management documentation per SR 11-7.

        Creates a model inventory entry with all required fields for
        regulatory model risk management processes.

        Returns:
            Model inventory document as a dictionary.
        """
        inventory = {
            "model_id": f"CR-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            "model_name": "Credit Risk Scoring Model",
            "model_type": self.training_metadata.get("model_type", "LightGBM"),
            "model_tier": "Tier 1",
            "business_purpose": "Consumer credit underwriting risk assessment",
            "model_owner": "Credit Risk Analytics",
            "validation_frequency": "Annual",
            "regulatory_requirements": ["SR 11-7", "ECOA", "Reg B"],
            "development_details": {
                "training_date": self.training_metadata.get("trained_at", ""),
                "training_data_period": "Rolling 24 months",
                "sample_size": "See training metadata",
                "feature_count": self.training_metadata.get("num_features", 0),
                "target_variable": "90+ DPD default indicator",
                "methodology": "Gradient boosting with monotonic constraints",
            },
            "performance_metrics": {
                "train_auc": self.training_metadata.get("train_auc", None),
                "val_auc": self.training_metadata.get("val_auc", None),
            },
            "limitations": [
                "Model trained on synthetic data for demonstration",
                "Monotonic constraints may limit predictive power",
                "Requires WOE-encoded input features",
            ],
            "monitoring_plan": {
                "frequency": "Monthly",
                "metrics": ["PSI", "Gini", "KS", "Default rate by decile"],
                "triggers": {
                    "psi_threshold": 0.25,
                    "gini_degradation": 0.10,
                    "ks_degradation": 0.05,
                },
            },
        }

        logger.info("Model inventory generated: %s", inventory["model_id"])
        return inventory

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate default probability predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of default probabilities.
        """
        if self.model is not None:
            return self.model.predict(X)
        elif self.lr_model is not None:
            return self.lr_model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No model has been trained")
