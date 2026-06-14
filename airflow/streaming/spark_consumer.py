from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import from_json
from pyspark.sql.functions import col
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import round
from pyspark.sql.functions import (
    hour,
    dayofweek,
    year,
    unix_timestamp,
    when
)

spark = (
    SparkSession.builder
    .appName("TaxiKafkaConsumer")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2"
    )
    .getOrCreate()
)

# taxi_schema = StructType([
#     StructField("VendorID", IntegerType(), True),
#     StructField("trip_distance", DoubleType(), True),
#     StructField("fare_amount", DoubleType(), True),
#     StructField("tip_amount", DoubleType(), True)
# ])

taxi_schema = StructType([
    StructField("VendorID", IntegerType(), True),

    StructField("tpep_pickup_datetime", StringType(), True),
    StructField("tpep_dropoff_datetime", StringType(), True),

    StructField("passenger_count", DoubleType(), True),
    StructField("trip_distance", DoubleType(), True),

    StructField("RatecodeID", DoubleType(), True),

    StructField("store_and_fwd_flag", StringType(), True),

    StructField("PULocationID", IntegerType(), True),
    StructField("DOLocationID", IntegerType(), True),

    StructField("payment_type", IntegerType(), True),

    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("congestion_surcharge", DoubleType(), True),
    StructField("airport_fee", DoubleType(), True)
])


print("Spark Session Created")

df = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        "kafka:9092"
    )
    .option(
        "subscribe",
        "taxi-rides"
    )
    .load()
)

print("Kafka Connection Successful")

json_df = df.selectExpr(
    "CAST(value AS STRING) as taxi_json"
)

parsed_df = json_df.select(
    from_json(
        col("taxi_json"),
        taxi_schema
    ).alias("data")
)

final_df = parsed_df.select(
    "data.*"
)

final_df = (
    final_df
    .withColumn(
        "pickup_ts",
        to_timestamp(
            "tpep_pickup_datetime",
            "yyyy-MM-dd HH:mm:ss"
        )
    )
    .withColumn(
        "dropoff_ts",
        to_timestamp(
            "tpep_dropoff_datetime",
            "yyyy-MM-dd HH:mm:ss"
        )
    )
)

final_df = (
    final_df

    .withColumn(
        "trip_duration_min",
        (
            unix_timestamp("dropoff_ts")
            - unix_timestamp("pickup_ts")
        ) / 60.0
    )

    .withColumn(
        "hour_of_day",
        hour("pickup_ts")
    )

    .withColumn(
        "day_of_week",
        dayofweek("pickup_ts")
    )

    .withColumn(
        "year",
        year("pickup_ts")
    )

    .withColumn(
        "is_airport",
        col("RatecodeID").isin([2, 3]).cast("int")
    )

     .withColumn(
        "tip_pct",
        round(
            (col("tip_amount") / col("fare_amount")) * 100,
            2
        )
    )

    .filter(
        col("trip_duration_min").between(1, 180)
    )
)

final_df.printSchema()

#parsed_df.printSchema()

#json_df.printSchema()

#df.printSchema()

# query = (
#     json_df.writeStream
#     .format("console")
#     .option("truncate", False)
#     .outputMode("append")
#     .start()
# )

query = (
    final_df.select(
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "trip_duration_min",
        "hour_of_day",
        "day_of_week",
        "year",
        "tip_pct"
    )
    .writeStream
    .format("console")
    .outputMode("append")
    .start()
)

# query = (
#     final_df.writeStream
#     .format("console")
#     .outputMode("append")
#     .start()
# )

query.awaitTermination()
