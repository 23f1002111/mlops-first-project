from pyspark.sql import SparkSession
import pandas as pd

print("Creating Spark session...")

spark = SparkSession.builder.getOrCreate()

print("Loading feature dataset...")

df = spark.read.parquet(
    "/opt/airflow/taxi_output/features"
)

print("Total rows:", df.count())

FEATURES = [
    "trip_distance",
    "passenger_count",
    "trip_duration_min",
    "hour_of_day",
    "day_of_week",
    "is_airport"
]

TARGET = "fare_amount"

model_df = df.select(
    FEATURES + [TARGET]
)

print("Sampling 500000 rows...")

sample_df = model_df.limit(500000)

print("Converting to pandas...")

pdf = sample_df.toPandas()

print("Pandas shape:", pdf.shape)

print(pdf.head())
