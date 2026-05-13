import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import mutual_info_classif

import seaborn as sns
import matplotlib.pyplot as plt


# =====================================================
# CRAMER'S V
# =====================================================


def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x, y)

    chi2 = chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

    rcorr = r - ((r - 1) ** 2) / (n - 1)

    kcorr = k - ((k - 1) ** 2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# =====================================================
# FEATURE DIAGNOSTICS
# =====================================================


def run_feature_diagnostics(df, target, selected_features):

    print("RUNNING FEATURE DIAGNOSTICS")

    X = df[selected_features].copy()

    y = df[target].copy()


    # =================================================
    # MISSINGNESS
    # =================================================

    missingness = (
        X.isnull()
        .mean()
        .sort_values(ascending=False)
    )

    print("MISSINGNESS")
    print(missingness)


    # =================================================
    # CARDINALITY
    # =================================================

    cardinality = pd.Series({
        col: X[col].nunique()
        for col in X.columns
    }).sort_values(ascending=False)

    print("CARDINALITY")
    print(cardinality)


    # =================================================
    # NUMERIC CORRELATION
    # =================================================

    numeric_features = (
        X
        .select_dtypes(include=["number"])
        .columns
    )

    if len(numeric_features) > 1:

        corr_matrix = (
            X[numeric_features]
            .corr(method="spearman")
        )

        print("NUMERIC CORRELATION")
        print(corr_matrix)

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0
        )

        plt.title("Numeric Feature Correlation")

        plt.savefig(
            "outputs/figures/numeric_correlation.png",
            bbox_inches='tight'
        )

        plt.close()


    # =================================================
    # CATEGORICAL ASSOCIATION
    # =================================================

    categorical_features = (
        X
        .select_dtypes(include=["object", "string"])
        .columns
    )

    cramers_results = []

    for i, col1 in enumerate(categorical_features):

        for col2 in categorical_features[i+1:]:

            try:

                value = cramers_v(
                    X[col1].fillna("MISSING"),
                    X[col2].fillna("MISSING")
                )

                cramers_results.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "cramers_v": value
                })

            except:
                pass


    cramers_df = pd.DataFrame(cramers_results)

    cramers_df = (
        cramers_df
        .sort_values(by="cramers_v", ascending=False)
    )


    print("TOP CATEGORICAL ASSOCIATIONS")
    print(cramers_df.head(20))


    cramers_df.to_csv(
        "outputs/models/cramers_v_results.csv",
        index=False
    )


    # =================================================
    # MUTUAL INFORMATION
    # =================================================

    X_encoded = X.copy()

    for col in X_encoded.columns:

        if not pd.api.types.is_numeric_dtype(X_encoded[col]):

            le = LabelEncoder()

            X_encoded[col] = le.fit_transform(
                X_encoded[col]
                .astype(str)
            )


    X_encoded = X_encoded.fillna(-999)


    mi_scores = mutual_info_classif(
        X_encoded,
        y,
        discrete_features='auto',
        random_state=42
    )


    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mutual_information": mi_scores
    })


    mi_df = (
        mi_df
        .sort_values(
            by="mutual_information",
            ascending=False
        )
    )


    print("MUTUAL INFORMATION")
    print(mi_df)


    mi_df.to_csv(
        "outputs/models/mutual_information.csv",
        index=False
    )


    # =================================================
    # PROFESSIONAL PRUNING RECOMMENDATIONS
    # =================================================

    print("FEATURE PRUNING RECOMMENDATIONS")


    high_missing = missingness[
        missingness > 0.60
    ]

    if len(high_missing) > 0:

        print("HIGH MISSINGNESS FEATURES")
        print(high_missing)


    high_cardinality = cardinality[
        cardinality > 500
    ]

    if len(high_cardinality) > 0:

        print("HIGH CARDINALITY FEATURES")
        print(high_cardinality)


    weak_mi = mi_df[
        mi_df["mutual_information"] < 0.001
    ]

    if len(weak_mi) > 0:

        print("WEAK PREDICTIVE FEATURES")
        print(weak_mi)


    strong_association = cramers_df[
        cramers_df["cramers_v"] > 0.80
    ]

    if len(strong_association) > 0:

        print("REDUNDANT CATEGORICAL FEATURES")
        print(strong_association)


    return {
        "missingness": missingness,
        "cardinality": cardinality,
        "mutual_information": mi_df,
        "cramers_v": cramers_df
        }