import pandas as pd

print("Loading dataset for validation...")

df = pd.read_csv("/opt/airflow/dags/iris_data.csv")

print("Checking for missing values...")

if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values!")

print("Checking dataset shape...")

if df.shape[0] == 0:
    raise ValueError("Dataset is empty!")

print("Validation successful.")
