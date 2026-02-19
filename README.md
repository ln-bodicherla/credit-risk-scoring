# Credit Risk Scoring Model

LightGBM-based credit risk assessment with Optuna hyperparameter optimization and monotonic constraints. Built for regulatory compliance (ECOA, fair lending). Includes scorecard generation and model governance documentation.

## Overview

A production-grade credit risk scoring system that combines modern gradient boosting with traditional scorecard methodology. The model is designed to meet regulatory requirements for interpretability, fairness, and model risk management per SR 11-7 guidelines.

## Features

- **WOE/IV Feature Engineering**: Weight of Evidence binning and Information Value calculation for feature selection and transformation
- **LightGBM with Monotonic Constraints**: Gradient boosting with enforced monotonic relationships between features and risk
- **Optuna Hyperparameter Tuning**: Bayesian optimization with custom objectives and early stopping
- **Traditional Scorecard Generation**: Converts model output to points-based scorecard format (base score, PDO scaling)
- **Comprehensive Evaluation**: Gini, KS statistic, PSI, calibration analysis, and decile-level performance
- **Fair Lending Analysis**: Adverse impact ratio, marginal effect analysis, and disparate impact testing
- **Model Governance**: SR 11-7 aligned validation reports and model inventory documentation

## Quick Start

```bash
pip install -r requirements.txt

# Train and evaluate a model
python -c "
from src.data_preparation import CreditDataPreprocessor
from src.model_training import CreditRiskModel
from src.evaluation import CreditModelEvaluator

# Prepare data
prep = CreditDataPreprocessor()
X_train, X_test, y_train, y_test = prep.run_pipeline()

# Train model
model = CreditRiskModel()
model.train_lightgbm(X_train, y_train)

# Evaluate
evaluator = CreditModelEvaluator(model, X_test, y_test)
evaluator.generate_validation_report('reports/')
"

# Run fair lending analysis
python -c "
from src.fairness import FairLendingAnalyzer
analyzer = FairLendingAnalyzer(model, X_test, y_test, protected_attrs)
analyzer.generate_fair_lending_report('reports/')
"

# Score new applications
python src/predict.py --input applications.csv --output scores.csv
```

## Project Structure

```
credit-risk-scoring/
├── src/
│   ├── data_preparation.py    # Data loading, WOE binning, IV calculation
│   ├── model_training.py      # LightGBM, Optuna, scorecard generation
│   ├── evaluation.py          # Gini, KS, PSI, calibration metrics
│   ├── fairness.py            # ECOA compliance, disparate impact testing
│   └── predict.py             # CLI for scoring new applications
├── configs/
│   └── config.yaml            # Model and evaluation configuration
├── requirements.txt
└── README.md
```

## Model Performance

Typical results on synthetic credit data:

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.82 |
| Gini | 0.64 |
| KS Statistic | 0.48 |
| PSI (train vs test) | 0.03 |
| AIR (all groups) | > 0.80 |

## Regulatory Compliance

- **SR 11-7**: Model risk management documentation with model inventory, validation report, and ongoing monitoring plan
- **ECOA / Fair Lending**: Adverse impact ratio analysis, marginal effect testing, and bias mitigation recommendations
- **Interpretability**: Monotonic constraints ensure directionally consistent risk factors; WOE encoding provides interpretable feature contributions

## License

MIT License
