from sklearn.datasets import load_iris
import pandas as pd

print("Loading Iris dataset...")

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["target"] = iris.target

df.to_csv("/opt/airflow/dags/iris_data.csv", index=False)

print("Dataset saved successfully.")
