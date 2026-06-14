from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

print("Creating Spark Session...")

spark = SparkSession.builder.getOrCreate()

print("Loading feature dataset...")

df = spark.read.parquet(
    "/opt/airflow/taxi_output/features"
)

print("Rows:", df.count())

FEATURES = [
    "trip_distance",
    "passenger_count",
    "trip_duration_min",
    "hour_of_day",
    "day_of_week",
    "is_airport"
]

TARGET = "fare_amount"

print("Selecting columns...")

model_df = (
    df.select(FEATURES + [TARGET])
      .na.drop()
      .sample(fraction=0.10, seed=42)
)

assembler = VectorAssembler(
    inputCols=FEATURES,
    outputCol="features"
)

assembled_df = assembler.transform(model_df)

final_df = assembled_df.select(
    "features",
    TARGET
)

train_df, test_df = final_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train rows:", train_df.count())
print("Test rows :", test_df.count())

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=TARGET,
    numTrees=20,
    maxDepth=10
)

print("Training model...")

model = rf.fit(train_df)

print("Model training complete.")

predictions = model.transform(test_df)

evaluator_rmse = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("RMSE:", rmse)
print("R2  :", r2)

model.save(
    "/opt/airflow/models/taxi_rf_model"
)

print("Model saved.")
