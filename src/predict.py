"""
CLI for scoring new credit applications using a trained model.

Loads a trained credit risk model and applies it to new application
data, producing risk scores and approval recommendations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CreditScorer:
    """Production scorer for credit applications.

    Loads a trained model and preprocessing artifacts, then applies
    them to new applications to produce risk scores and recommendations.
    """

    def __init__(
        self,
        model_path: str = "models/credit_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
        scorecard_path: Optional[str] = None,
    ):
        self.model = None
        self.preprocessor = None
        self.scorecard = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        self._load_artifacts(model_path, preprocessor_path, scorecard_path)

    def _load_artifacts(
        self,
        model_path: str,
        preprocessor_path: str,
        scorecard_path: Optional[str],
    ) -> None:
        """Load model and preprocessing artifacts from disk."""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
        else:
            logger.warning("Model file not found: %s", model_path)

        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded from %s", preprocessor_path)
        else:
            logger.warning("Preprocessor file not found: %s", preprocessor_path)

        if scorecard_path and os.path.exists(scorecard_path):
            self.scorecard = pd.read_csv(scorecard_path)
            logger.info("Scorecard loaded from %s", scorecard_path)

    def score_application(
        self,
        application: dict[str, Any],
        approval_threshold: float = 0.5,
        base_score: int = 600,
        pdo: int = 20,
    ) -> dict[str, Any]:
        """Score a single credit application.

        Args:
            application: Dictionary of application features.
            approval_threshold: Probability threshold for approval.
            base_score: Base score for scorecard conversion.
            pdo: Points to double the odds.

        Returns:
            Scoring result with probability, score, and recommendation.
        """
        df = pd.DataFrame([application])

        if self.preprocessor is not None:
            df = self.preprocessor.handle_missing(df)

        if self.model is not None:
            probability = float(self.model.predict(df)[0])
        else:
            probability = 0.5

        factor = pdo / np.log(2)
        odds = (1 - probability) / max(probability, 1e-10)
        credit_score = int(base_score + factor * np.log(odds))
        credit_score = max(300, min(850, credit_score))

        if probability < approval_threshold * 0.5:
            recommendation = "AUTO_APPROVE"
            risk_tier = "Low Risk"
        elif probability < approval_threshold:
            recommendation = "MANUAL_REVIEW"
            risk_tier = "Medium Risk"
        else:
            recommendation = "DECLINE"
            risk_tier = "High Risk"

        return {
            "default_probability": round(probability, 6),
            "credit_score": credit_score,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "approval_threshold": approval_threshold,
        }

    def score_batch(
        self,
        applications: pd.DataFrame,
        approval_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Score a batch of credit applications.

        Args:
            applications: DataFrame of application features.
            approval_threshold: Probability threshold for approval.

        Returns:
            DataFrame with original features plus scoring columns.
        """
        results = []
        for _, row in applications.iterrows():
            result = self.score_application(row.to_dict(), approval_threshold)
            results.append(result)

        scores_df = pd.DataFrame(results)
        output = pd.concat([applications.reset_index(drop=True), scores_df], axis=1)

        logger.info(
            "Batch scoring complete: %d applications, %.1f%% approved",
            len(output),
            (output["recommendation"] != "DECLINE").mean() * 100,
        )

        return output


def main():
    """CLI entry point for credit application scoring."""
    parser = argparse.ArgumentParser(
        description="Score credit applications using a trained risk model",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file with credit applications",
    )
    parser.add_argument(
        "--output", "-o",
        default="scored_applications.csv",
        help="Path to output CSV file with scores (default: scored_applications.csv)",
    )
    parser.add_argument(
        "--model", "-m",
        default="models/credit_model.pkl",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--preprocessor", "-p",
        default="models/preprocessor.pkl",
        help="Path to preprocessor artifacts",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Approval probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    scorer = CreditScorer(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
    )

    logger.info("Reading applications from %s", args.input)
    applications = pd.read_csv(args.input)
    logger.info("Loaded %d applications", len(applications))

    scored = scorer.score_batch(applications, approval_threshold=args.threshold)

    if args.format == "json":
        scored.to_json(args.output, orient="records", indent=2)
    else:
        scored.to_csv(args.output, index=False)

    logger.info("Scores written to %s", args.output)

    summary = scored["recommendation"].value_counts()
    logger.info("Decision summary:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
