from fastapi import FastAPI
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("iris_model.pkl")

@app.get("/")
def home():
    return {"message": "ML API is running!"}

@app.get("/predict")
def predict():
    
    # Example flower measurements
    sample = [[5.1, 3.5, 1.4, 0.2]]
    
    prediction = model.predict(sample)

    return {
        "prediction": int(prediction[0])
    }