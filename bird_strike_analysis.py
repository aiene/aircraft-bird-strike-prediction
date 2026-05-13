from src.feature_diagnostics import run_feature_diagnostics
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline

from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_STATE = 42

DATA_PATH = (
    r"C:\Users\Lenovo\python_birdstrike\data\raw\massdata.csv"
)

TARGET = "INDICATED_DAMAGE"


# =====================================================
# LOAD DATA
# =====================================================

print("\nLOADING DATASET")


df = pd.read_csv(
    DATA_PATH,
    sep=';',
    encoding='utf-8',
    low_memory=False,
    on_bad_lines='skip'
)


# =====================================================
# COLUMN CLEANING
# =====================================================


df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_", regex=False)
)


print("\nDATASET SHAPE")
print(df.shape)


# =====================================================
# TARGET CLEANING
# =====================================================


y = (
    df[TARGET]
    .astype(str)
    .str.upper()
    .str.strip()
)


target_mapping = {
    "FALSO": 0,
    "VERO": 1
}


y = y.map(target_mapping)


valid_idx = y.notnull()


df = df.loc[valid_idx].copy()

y = y.loc[valid_idx].astype(int)


print("\nTARGET DISTRIBUTION")
print(y.value_counts())


# =====================================================
# FEATURE POLICY
# =====================================================


selected_features = [

    # aircraft
    "ATYPE",
    "AC_MASS",
    "NUM_ENGS",

    # temporal
    "INCIDENT_MONTH",
    "TIME_OF_DAY",

    # location
    "AIRPORT",

    # operational
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "PHASE_OF_FLT",

    # environmental
    "SKY",

    # wildlife
    "SPECIES",
    "BIRDS_SEEN",
    "BIRDS_STRUCK",
    "SIZE"
]


existing_features = [
    col for col in selected_features
    if col in df.columns
]


X = df[existing_features].copy()


print("\nFINAL FEATURES")
print(existing_features)


# =====================================================
# NUMERIC / CATEGORICAL SPLIT
# =====================================================


numeric_features = (
    X
    .select_dtypes(include=["number"])
    .columns
)


categorical_features = (
    X
    .select_dtypes(include=["object", "string"])
    .columns
)


print("\nNUMERIC FEATURES")
print(list(numeric_features))


print("\nCATEGORICAL FEATURES")
print(list(categorical_features))


# =====================================================
# PREPROCESSING
# =====================================================


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=0.01
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


print("\nPREPROCESSING COMPLETED")

feature_results = run_feature_diagnostics(
    df=df,
    target=TARGET,
    selected_features=existing_features
)
# =====================================================
# TRAIN TEST SPLIT
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)


print("\nTRAIN SHAPE")
print(X_train.shape)


print("\nTEST SHAPE")
print(X_test.shape)


# =====================================================
# RANDOM FOREST PIPELINE
# =====================================================


rf_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(random_state=RANDOM_STATE)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# =====================================================
# LOGISTIC REGRESSION PIPELINE
# =====================================================


log_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(random_state=RANDOM_STATE)
        ),
        (
            "classifier",
            LogisticRegression(
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


# =====================================================
# CROSS VALIDATION
# =====================================================


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall"
}


print("\nRUNNING RANDOM FOREST CV")


rf_cv = cross_validate(
    rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


print("\nRANDOM FOREST CV RESULTS")

for metric in scoring.keys():
    print(
        metric,
        np.mean(rf_cv[f"test_{metric}"])
    )


print("\nRUNNING LOGISTIC REGRESSION CV")


log_cv = cross_validate(
    log_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


print("\nLOGISTIC REGRESSION CV RESULTS")

for metric in scoring.keys():
    print(
        metric,
        np.mean(log_cv[f"test_{metric}"])
    )


# =====================================================
# FINAL RANDOM FOREST FIT
# =====================================================


print("\nTRAINING FINAL RANDOM FOREST")


rf_pipeline.fit(X_train, y_train)


# =====================================================
# PREDICTIONS
# =====================================================


y_pred = rf_pipeline.predict(X_test)


y_proba = rf_pipeline.predict_proba(X_test)[:, 1]

# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_proba
)

f1_scores = (
    2 * precision * recall
    / (precision + recall + 1e-10)
)

best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]

print("\nBEST THRESHOLD")
print(best_threshold)

print("\nBEST F1")
print(f1_scores[best_idx])

# optimized predictions

y_pred_optimized = (
    y_proba >= best_threshold
).astype(int)

print("\nOPTIMIZED CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred_optimized
    )
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================


print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =====================================================
# ROC AUC
# =====================================================


roc_auc = roc_auc_score(
    y_test,
    y_proba
)


print("\nROC-AUC")
print(roc_auc)


# =====================================================
# CONFUSION MATRIX
# =====================================================


cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")


plt.savefig(
    "outputs/figures/confusion_matrix.png",
    bbox_inches='tight'
)

plt.close()


# =====================================================
# ROC CURVE
# =====================================================


RocCurveDisplay.from_predictions(
    y_test,
    y_proba
)


plt.title("Random Forest ROC Curve")


plt.savefig(
    "outputs/figures/roc_curve.png",
    bbox_inches='tight'
)

plt.close()


# =====================================================
# FEATURE IMPORTANCE
# =====================================================


print("\nRUNNING PERMUTATION IMPORTANCE")


importance = permutation_importance(
    rf_pipeline,
    X_test,
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="roc_auc"
)


importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance.importances_mean
})


importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)


print("\nTOP FEATURES")
print(importance_df.head(20))


# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================


importance_df.to_csv(
    "outputs/models/feature_importance.csv",
    index=False
)


# =====================================================
# SAVE MODEL
# =====================================================


import joblib


joblib.dump(
    rf_pipeline,
    "outputs/models/random_forest_pipeline.pkl"
)


print("\nPIPELINE COMPLETED SUCCESSFULLY")