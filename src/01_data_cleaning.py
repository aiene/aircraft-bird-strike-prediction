import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\Lenovo\python_birdstrike\data\raw\massdata.csv"

df = pd.read_csv(
    DATA_PATH,
    sep=';',
    encoding='utf-8',
    low_memory=False,
    on_bad_lines='skip'
)

print("\nDATASET SHAPE")
print(df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nTARGET DISTRIBUTION")
print(df["INDICATED_DAMAGE"].value_counts(dropna=False))

df.columns = (
    df.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
)

print("\nCLEANED COLUMN NAMES")
print(df.columns.tolist())

df.to_csv(
    r"C:\Users\Lenovo\python_birdstrike\data\raw\cleaned_massdata.csv",
    index=False
)

print("\nCLEANED DATA SAVED")