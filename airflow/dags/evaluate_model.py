from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

print("Loading trained model...")

#model = joblib.load("iris_model.pkl")
model = joblib.load("/opt/airflow/models/latest_model.pkl")

print("Loading dataset...")

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Running evaluation...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Evaluation Accuracy: {accuracy}")

if accuracy < 0.8:
    raise ValueError("Model accuracy below acceptable threshold!")

print("Model passed evaluation.")
