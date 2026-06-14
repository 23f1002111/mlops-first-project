import json
import time

from kafka import KafkaProducer

from pyspark.sql import SparkSession


print("Creating Spark Session...")

spark = SparkSession.builder.getOrCreate()

print("Loading Taxi Data...")

df = spark.read.parquet(
    "/opt/airflow/taxi_data/yellow_tripdata_2025-01.parquet"
)

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

topic = "taxi-rides"

print("Sending records...")

rows = (
    df.limit(1000)
      .toJSON()
      .collect()
)

for row in rows:

    producer.send(
        topic,
        json.loads(row)
    )

    print("Sent")

    time.sleep(0.1)

producer.flush()

print("Done.")
