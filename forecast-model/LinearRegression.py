# Databricks notebook source
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, sum as spark_sum

# Create Spark session
spark = SparkSession.builder.appName("CostForecasting").getOrCreate()

# Define file paths (assuming files are in a mounted directory '/mnt/data/')
files = {
    "ActiveContracts": "/mnt/data/ActiveContracts.csv",
    "CommercialContractLists": "/mnt/data/CommercialContractLists.csv",
    "FebMarHours": "/mnt/data/2023 SY Feb-Mar - All hours _ Earnings.csv",
    "DecJanHours": "/mnt/data/2023 SY Dec-Jan - All hours _ Earnings.csv",
    "AprJuneHours": "/mnt/data/2023 SY Apr-June - All hours _ Earnings.csv",
    "AugSepHours": "/mnt/data/2023 SY Aug-Sep - All hours _ Earnings.csv",
    "OctNovHours": "/mnt/data/2023 SY Oct-Nov - All hours _ Earnings.csv",
    "SchedulHrsPt1": "/mnt/data/Schedul Hrs Transactional pt1.csv",
    "RecruitingCosts": "/mnt/data/Recruiting SB School Year 2024.csv"
}

# Load the data into DataFrames
df_dict = {name: spark.read.option("header", "true").csv(path) for name, path in files.items()}


# COMMAND ----------

from pyspark.sql.functions import to_date

# Example of transforming one of the payroll data
df_payroll = df_dict["FebMarHours"]
df_payroll = df_payroll.withColumn("Month", month(to_date("PayDate", "MM/dd/yyyy")))
df_payroll = df_payroll.withColumn("Year", year(to_date("PayDate", "MM/dd/yyyy")))

# Aggregate data monthly across all CSVs if necessary
# Assuming each payroll DataFrame follows a similar schema
for key, df in df_dict.items():
    if "Hours" in key:  # Simplified check
        df = df.withColumn("Month", month(to_date("PayDate", "MM/dd/yyyy")))
        df = df.withColumn("Year", year(to_date("PayDate", "MM/dd/yyyy")))
        df_dict[key] = df.groupBy("CSC", "Year", "Month").agg(spark_sum("TotalPay").alias("TotalMonthlyCost"))

# Combine all payroll data into a single DataFrame
df_combined = df_dict["FebMarHours"]
for key in ["DecJanHours", "AprJuneHours", "AugSepHours", "OctNovHours"]:
    df_combined = df_combined.unionByName(df_dict[key])


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Assemble features for linear regression
vectorAssembler = VectorAssembler(inputCols=['MonthIndex'], outputCol='features')
df_combined = df_combined.withColumn("MonthIndex", col("Month") + col("Year") * 12)
df_combined = vectorAssembler.transform(df_combined)

# Fit a linear regression model
lr = LinearRegression(featuresCol='features', labelCol='TotalMonthlyCost')
lr_model = lr.fit(df_combined)

# Predict future costs
future_months = spark.range(1, 13).toDF("MonthIndex")  # Predicting for 12 future months
future_months = vectorAssembler.transform(future_months)
future_predictions = lr_model.transform(future_months)
future_predictions.select("MonthIndex", "prediction").show()


# COMMAND ----------

# Convert predictions to Pandas for visualization
import matplotlib.pyplot as plt
pdf_predictions = future_predictions.toPandas()
plt.plot(pdf_predictions['MonthIndex'], pdf_predictions['prediction'])
plt.title('Forecasted Monthly Costs')
plt.xlabel('Month Index')
plt.ylabel('Cost')
plt.show()

