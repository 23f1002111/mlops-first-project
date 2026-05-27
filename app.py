from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

#MODEL_PATH = "/mnt/c/Users/kanwa/mlops-first-project/airflow/models/latest_model.pkl"
MODEL_PATH = "/app/models/latest_model.pkl"

print("Loading latest trained model...")

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.get("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):

    features = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }