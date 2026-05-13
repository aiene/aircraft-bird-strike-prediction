# Bird Strike Analysis

Binary classification model to predict whether a bird strike incident resulted in aircraft damage, using machine learning on operational and environmental factors from incident records.

## Overview

This project builds a predictive classifier from bird strike incident data. The target variable is `INDICATED_DAMAGE`—a binary indicator of whether damage was sustained. The model takes inputs like aircraft type, flight phase, altitude, bird species, and conditions at the time of strike to estimate damage risk.

The implementation combines data preprocessing, class imbalance correction via SMOTE, and model comparison between Random Forest and Logistic Regression pipelines, with threshold optimization based on F1 score.

## Data & Features

The analysis uses bird strike records with the following feature groups:

- **Aircraft characteristics**: type, mass, number of engines
- **Temporal**: incident month, time of day
- **Location**: airport
- **Flight operations**: altitude, speed, distance, phase of flight
- **Environment**: sky conditions
- **Wildlife**: species, number of birds seen/struck, bird size

Missing values are handled via median imputation for numeric features and mode imputation for categorical. Categorical variables are one-hot encoded, and numeric features are standardized.

## Methodology

**Train-test split**: 80-20 stratified split to maintain class distribution.

**Class imbalance**: SMOTE is applied during training to address imbalance in the target variable.

**Models**:
- Random Forest (300 trees, max depth 12, class weighting)
- Logistic Regression (saga solver, L2 regularization, class weighting)

**Evaluation**: 5-fold stratified cross-validation with metrics including ROC-AUC, F1, precision, and recall. Test set predictions are threshold-optimized using precision-recall curves.

**Feature analysis**: Permutation importance scores rank features by their impact on ROC-AUC on the test set.

## Running the Analysis

```bash
python bird_strike_analysis.py
```

This will:
1. Load and clean the raw dataset
2. Run feature diagnostics and correlation analysis
3. Preprocess and split the data
4. Train both models with cross-validation
5. Generate predictions on the test set
6. Output classification reports, confusion matrices, and ROC curves to `outputs/figures/`
7. Save the fitted Random Forest pipeline and feature importance rankings

## Outputs

- `confusion_matrix.png`: Confusion matrix for final model predictions
- `roc_curve.png`: ROC curve with area under the curve
- `feature_importance.csv`: Permutation importance for each feature
- `random_forest_pipeline.pkl`: Serialized trained model

## Requirements

- Python 3.8+
- scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, seaborn, joblib

Install via `pip install -r requirements.txt`