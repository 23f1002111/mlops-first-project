from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import json
import mlflow
import joblib

print("Loading dataset...")

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Splitting dataset...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Starting MLflow run...")

# Start MLflow tracking
with mlflow.start_run():

    print("Training model...")

    # Model
    model = RandomForestClassifier(n_estimators=100)

    # Train
    model.fit(X_train, y_train)

    print("Making predictions...")

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")

    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    #joblib.dump(model, "iris_model.pkl")
    #joblib.dump(model, "/opt/airflow/models/latest_model.pkl")

    from datetime import datetime
    import os

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Versioned model name
    model_name = f"model_{timestamp}.pkl"

    # Model paths
    models_dir = "/opt/airflow/models"

    versioned_model_path = os.path.join(models_dir, model_name)
    latest_model_path = os.path.join(models_dir, "latest_model.pkl")

    # # Save versioned model
    # joblib.dump(model, versioned_model_path)

    # print(f"Versioned model saved: {versioned_model_path}")

    # # Update latest model
    # joblib.dump(model, latest_model_path)

    # print(f"Latest model updated: {latest_model_path}")

    # Save versioned model

    joblib.dump(model, versioned_model_path)

    print(f"Versioned model saved: {versioned_model_path}")

    # Metrics file
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    # Default best accuracy
    best_accuracy = 0

    # Check if metrics file exists
    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        best_accuracy = metrics_data.get("best_accuracy", 0)

    print(f"Previous best accuracy: {best_accuracy}")

    # Promote model only if accuracy improved
    if accuracy > best_accuracy:

        print("New best model found!")

        # Update latest model
        joblib.dump(model, latest_model_path)

        # Save metrics
        metrics_data = {
            "best_accuracy": accuracy,
            "best_model": model_name
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

        print(f"Latest model updated: {latest_model_path}")

    else:

        print("Model did not improve. Keeping previous production model.")

    print("Model saved successfully.")