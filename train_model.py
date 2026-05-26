from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import joblib

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow tracking
with mlflow.start_run():

    # Model
    model = RandomForestClassifier(n_estimators=100)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", 100)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    joblib.dump(model, "iris_model.pkl")

    print(f"Accuracy: {accuracy}")
    print("Model trained and tracked!")