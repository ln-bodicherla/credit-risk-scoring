"""
Credit data preprocessing pipeline.

Handles data loading, missing value imputation, Weight of Evidence binning,
Information Value calculation, feature encoding, and train/test splitting
with temporal stratification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CreditDataPreprocessor:
    """Preprocessor for credit application data with WOE/IV feature engineering.

    Implements a complete preprocessing pipeline from raw credit application
    data through WOE-encoded features ready for model training. Includes
    synthetic data generation for demonstration purposes.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.woe_mappings: dict[str, dict[str, float]] = {}
        self.iv_scores: dict[str, float] = {}
        self.feature_bins: dict[str, list[float]] = {}
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load preprocessing configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load credit application data from a CSV file or generate synthetic data.

        Args:
            file_path: Path to CSV file. If None, generates synthetic data.

        Returns:
            DataFrame with credit application features and target variable.
        """
        if file_path is not None:
            logger.info("Loading data from %s", file_path)
            df = pd.read_csv(file_path)
        else:
            logger.info("Generating synthetic credit data")
            df = self.generate_synthetic_credit_data()

        logger.info("Data shape: %s, default rate: %.3f", df.shape, df["default"].mean())
        return df

    def generate_synthetic_credit_data(
        self,
        n_samples: int = 10000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Generate realistic synthetic credit application data.

        Creates a dataset with features commonly found in credit scoring:
        income, age, employment length, debt-to-income ratio, credit
        utilization, number of open accounts, delinquency history, etc.

        Args:
            n_samples: Number of synthetic applications to generate.
            random_state: Random seed for reproducibility.

        Returns:
            DataFrame with synthetic credit features and a binary default target.
        """
        rng = np.random.RandomState(random_state)

        annual_income = rng.lognormal(mean=10.8, sigma=0.6, size=n_samples).clip(15000, 500000)
        age = rng.normal(loc=42, scale=12, size=n_samples).clip(18, 85).astype(int)
        employment_length = rng.exponential(scale=5, size=n_samples).clip(0, 40).round(1)
        dti_ratio = rng.beta(a=2, b=5, size=n_samples) * 0.6
        credit_utilization = rng.beta(a=2, b=3, size=n_samples)
        num_open_accounts = rng.poisson(lam=8, size=n_samples).clip(0, 30)
        num_delinquencies = rng.poisson(lam=0.5, size=n_samples).clip(0, 10)
        credit_history_months = rng.normal(loc=180, scale=60, size=n_samples).clip(6, 600).astype(int)
        loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n_samples).clip(1000, 500000)
        num_inquiries = rng.poisson(lam=1.5, size=n_samples).clip(0, 15)

        home_ownership = rng.choice(["OWN", "MORTGAGE", "RENT", "OTHER"], size=n_samples, p=[0.15, 0.45, 0.35, 0.05])
        loan_purpose = rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"],
            size=n_samples,
            p=[0.40, 0.25, 0.15, 0.10, 0.10],
        )

        race = rng.choice(["white", "black", "hispanic", "asian", "other"], size=n_samples, p=[0.60, 0.13, 0.18, 0.06, 0.03])
        gender = rng.choice(["male", "female"], size=n_samples, p=[0.52, 0.48])

        log_odds = (
            -2.5
            - 0.000015 * annual_income
            + 0.8 * dti_ratio
            + 1.2 * credit_utilization
            + 0.15 * num_delinquencies
            - 0.002 * credit_history_months
            - 0.03 * employment_length
            + 0.05 * num_inquiries
            + 0.000003 * loan_amount
        )

        noise = rng.normal(0, 0.3, size=n_samples)
        prob_default = 1.0 / (1.0 + np.exp(-(log_odds + noise)))
        default = (rng.random(n_samples) < prob_default).astype(int)

        mask = rng.random(n_samples) < 0.03
        annual_income_with_missing = annual_income.copy()
        annual_income_with_missing[mask] = np.nan

        mask2 = rng.random(n_samples) < 0.05
        employment_with_missing = employment_length.copy()
        employment_with_missing[mask2] = np.nan

        df = pd.DataFrame({
            "annual_income": annual_income_with_missing,
            "age": age,
            "employment_length": employment_with_missing,
            "dti_ratio": dti_ratio,
            "credit_utilization": credit_utilization,
            "num_open_accounts": num_open_accounts,
            "num_delinquencies": num_delinquencies,
            "credit_history_months": credit_history_months,
            "loan_amount": loan_amount,
            "num_inquiries": num_inquiries,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "race": race,
            "gender": gender,
            "default": default,
        })

        logger.info(
            "Generated %d samples, default rate: %.3f",
            n_samples,
            default.mean(),
        )
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific missing value imputation.

        Numeric features are imputed with the median. Missing indicators
        are created for features where missingness itself may be predictive.

        Args:
            df: Input DataFrame with potential missing values.

        Returns:
            DataFrame with imputed values and missing indicator columns.
        """
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "default"]

        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_rate = missing_count / len(df)
                logger.info("Imputing %s: %d missing (%.1f%%)", col, missing_count, missing_rate * 100)
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(df[col].median())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            df[col] = df[col].fillna("UNKNOWN")

        return df

    def woe_binning(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str = "default",
        n_bins: int = 10,
    ) -> pd.Series:
        """Calculate Weight of Evidence bins for a numeric feature.

        WOE measures the predictive power of each bin by comparing the
        distribution of goods (non-defaults) vs bads (defaults).

        Args:
            df: DataFrame containing the feature and target.
            feature: Name of the numeric feature to bin.
            target: Name of the binary target variable.
            n_bins: Number of bins to create.

        Returns:
            Series with WOE values for each observation.
        """
        data = df[[feature, target]].copy()
        data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")

        grouped = data.groupby("bin", observed=True)[target].agg(["sum", "count"])
        grouped.columns = ["bads", "total"]
        grouped["goods"] = grouped["total"] - grouped["bads"]

        total_goods = grouped["goods"].sum()
        total_bads = grouped["bads"].sum()

        grouped["dist_goods"] = grouped["goods"] / max(total_goods, 1)
        grouped["dist_bads"] = grouped["bads"] / max(total_bads, 1)

        grouped["dist_goods"] = grouped["dist_goods"].clip(lower=0.0001)
        grouped["dist_bads"] = grouped["dist_bads"].clip(lower=0.0001)

        grouped["woe"] = np.log(grouped["dist_goods"] / grouped["dist_bads"])

        woe_mapping = {}
        for interval, row in grouped.iterrows():
            woe_mapping[str(interval)] = row["woe"]

        self.woe_mappings[feature] = woe_mapping

        bin_labels = pd.qcut(data[feature], q=n_bins, duplicates="drop")
        bin_edges = pd.qcut(data[feature], q=n_bins, duplicates="drop", retbins=True)[1]
        self.feature_bins[feature] = bin_edges.tolist()

        woe_series = bin_labels.map(lambda x: woe_mapping.get(str(x), 0.0))
        return woe_series

    def information_value(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str = "default",
        n_bins: int = 10,
    ) -> float:
        """Calculate the Information Value for a feature.

        IV summarizes the overall predictive power of a feature.
        IV < 0.02: not useful, 0.02-0.1: weak, 0.1-0.3: medium,
        0.3-0.5: strong, > 0.5: suspicious (possible overfitting).

        Args:
            df: DataFrame with feature and target.
            feature: Feature name.
            target: Target variable name.
            n_bins: Number of bins for numeric features.

        Returns:
            Information Value score.
        """
        data = df[[feature, target]].dropna().copy()

        if data[feature].dtype in ("object", "category"):
            grouped = data.groupby(feature)[target].agg(["sum", "count"])
        else:
            data["bin"] = pd.qcut(data[feature], q=n_bins, duplicates="drop")
            grouped = data.groupby("bin", observed=True)[target].agg(["sum", "count"])

        grouped.columns = ["bads", "total"]
        grouped["goods"] = grouped["total"] - grouped["bads"]

        total_goods = max(grouped["goods"].sum(), 1)
        total_bads = max(grouped["bads"].sum(), 1)

        grouped["dist_goods"] = (grouped["goods"] / total_goods).clip(lower=0.0001)
        grouped["dist_bads"] = (grouped["bads"] / total_bads).clip(lower=0.0001)

        grouped["iv_component"] = (grouped["dist_goods"] - grouped["dist_bads"]) * \
                                   np.log(grouped["dist_goods"] / grouped["dist_bads"])

        iv = grouped["iv_component"].sum()
        self.iv_scores[feature] = iv

        logger.info("IV(%s) = %.4f", feature, iv)
        return iv

    def encode_features(
        self,
        df: pd.DataFrame,
        target: str = "default",
        n_bins: int = 10,
        iv_threshold: float = 0.02,
    ) -> pd.DataFrame:
        """Apply WOE encoding to all eligible features.

        Calculates IV for each feature, selects those above the threshold,
        and replaces original values with WOE-encoded values.

        Args:
            df: Input DataFrame.
            target: Target variable name.
            n_bins: Number of bins for numeric features.
            iv_threshold: Minimum IV for feature selection.

        Returns:
            DataFrame with WOE-encoded features.
        """
        df = df.copy()
        protected = {"race", "gender", "age"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target and c not in protected and not c.endswith("_missing")]

        self.numeric_features = []
        for col in numeric_cols:
            iv = self.information_value(df, col, target, n_bins)
            if iv >= iv_threshold:
                df[f"{col}_woe"] = self.woe_binning(df, col, target, n_bins)
                self.numeric_features.append(f"{col}_woe")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in protected]

        self.categorical_features = []
        for col in cat_cols:
            iv = self.information_value(df, col, target)
            if iv >= iv_threshold:
                woe_map = self._compute_categorical_woe(df, col, target)
                df[f"{col}_woe"] = df[col].map(woe_map).fillna(0.0)
                self.categorical_features.append(f"{col}_woe")

        selected = self.numeric_features + self.categorical_features
        missing_cols = [c for c in df.columns if c.endswith("_missing")]
        selected.extend(missing_cols)

        logger.info("Selected %d features (IV >= %.3f)", len(selected), iv_threshold)
        return df[selected + [target]]

    def _compute_categorical_woe(
        self,
        df: pd.DataFrame,
        feature: str,
        target: str,
    ) -> dict[str, float]:
        """Compute WOE values for each category level."""
        grouped = df.groupby(feature)[target].agg(["sum", "count"])
        grouped.columns = ["bads", "total"]
        grouped["goods"] = grouped["total"] - grouped["bads"]

        total_goods = max(grouped["goods"].sum(), 1)
        total_bads = max(grouped["bads"].sum(), 1)

        grouped["dist_goods"] = (grouped["goods"] / total_goods).clip(lower=0.0001)
        grouped["dist_bads"] = (grouped["bads"] / total_bads).clip(lower=0.0001)
        grouped["woe"] = np.log(grouped["dist_goods"] / grouped["dist_bads"])

        woe_map = grouped["woe"].to_dict()
        self.woe_mappings[feature] = {str(k): v for k, v in woe_map.items()}
        return woe_map

    def create_train_test(
        self,
        df: pd.DataFrame,
        target: str = "default",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create stratified train/test split.

        Performs stratified splitting to preserve the default rate
        distribution across both sets.

        Args:
            df: Encoded DataFrame with features and target.
            target: Target variable name.
            test_size: Proportion of data for test set.
            random_state: Random seed.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(
            "Train: %d samples (default rate: %.3f), Test: %d samples (default rate: %.3f)",
            len(X_train), y_train.mean(),
            len(X_test), y_test.mean(),
        )

        return X_train, X_test, y_train, y_test

    def run_pipeline(
        self,
        file_path: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute the full preprocessing pipeline.

        Args:
            file_path: Optional path to CSV data file.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) with WOE-encoded features.
        """
        df = self.load_data(file_path)
        df = self.handle_missing(df)
        df_encoded = self.encode_features(df)
        X_train, X_test, y_train, y_test = self.create_train_test(df_encoded)
        return X_train, X_test, y_train, y_test
