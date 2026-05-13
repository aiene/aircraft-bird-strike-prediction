import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from joblib import dump

DATA_PATH = r"C:\Users\Lenovo\python_birdstrike\data\raw\cleaned_massdata.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "INDICATED_DAMAGE"

df = df.dropna(subset=[TARGET])

df[TARGET] = df[TARGET].astype(str)

selected_features = [
    "AIRCRAFT",
    "FLIGHT_PHASE",
    "HEIGHT",
    "SPEED",
    "SKY",
    "TIME_OF_DAY",
    "OPERATOR"
]

selected_features = [
    col for col in selected_features
    if col in df.columns
]

X = df[selected_features]
y = df[TARGET]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

dump(preprocessor,
     r"C:\Users\Lenovo\python_birdstrike\outputs\models\preprocessor.joblib")

dump(X_train,
     r"C:\Users\Lenovo\python_birdstrike\outputs\models\X_train.joblib")

dump(X_test,
     r"C:\Users\Lenovo\python_birdstrike\outputs\models\X_test.joblib")

dump(y_train,
     r"C:\Users\Lenovo\python_birdstrike\outputs\models\y_train.joblib")

dump(y_test,
     r"C:\Users\Lenovo\python_birdstrike\outputs\models\y_test.joblib")

print("\nPREPROCESSING COMPLETED")

print("\nNUMERIC FEATURES")
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)