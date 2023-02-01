#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import findspark
findspark.init()

import pyspark.sql.functions as F
import pyspark

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

import argparse
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('crimes_csv_path', type=str, help='path to crimes.csv')
parser.add_argument('offense_codes_csv_path', type=str, help='path to offense_codes.csv')
parser.add_argument('output_path', type=str, help='Output dir for parquet files')
args = parser.parse_args()

# Загружаем датафрейм из crime.csv:
Df_Crimes = spark.read.format("csv").option("header","true").option("inferschema", "true").option("path",args.crimes_csv_path).load();

# Очищаем датафрейм(crime.csv) от дубликатов и нулевых значений в столбце "DISTRICT":
Df_Crimes_Clean = Df_Crimes.dropDuplicates(subset=["INCIDENT_NUMBER"]).dropna(subset=["DISTRICT"]);

# Загружаем датафрейм из offense_codes.csv:
Df_Codes = spark.read.format("csv").option("header","true").option("inferschema", "true").option("path",args.offense_codes_csv_path).load();

# Очищаем датафрейм(offense_codes.csv) от дубликатов и нулевых значений в столбце "CODE":
Df_Codes_Clean = Df_Codes.dropDuplicates(subset=["CODE"]);

# В датафрейме(offense_codes.csv) создаем столбец "crime_type":
Df_Codes_SplitTypes = Df_Codes_Clean.withColumn("crime_type",F.split(Df_Codes_Clean["NAME"], '-').getItem(0));

# Объединяем датафрейм(crime.csv) c датафреймом(offense_codes.csv):
Df_Crimes_Codes = Df_Crimes_Clean.join(Df_Codes_SplitTypes, (Df_Crimes_Clean["OFFENSE_CODE"] == Df_Codes_SplitTypes["CODE"]), 'inner');

# Создаем столбец "crimes_total":
Df_Crimes_Total= Df_Crimes_Codes.groupBy("DISTRICT").agg(F.count("DISTRICT").alias("crimes_total"));

# Создаем промежуточный столбец "crimes_per_month":
Df_Crimes_Monthly=Df_Crimes_Codes.groupBy("DISTRICT","YEAR","MONTH").agg(F.count("*").alias("crimes_per_month"));

# Создаем столбец "crimes_monthly":
Df_Crimes_Monthly_Mediana= Df_Crimes_Monthly.groupBy("DISTRICT").agg(F.percentile_approx("crimes_per_month", 0.5, 100).alias('crimes_monthly'));

# Создаем промежуточный столбец "frequency":
Df_Crimes_Frequency = Df_Crimes_Codes.groupBy("DISTRICT","OFFENSE_CODE","CODE","crime_type").agg(F.count("OFFENSE_CODE").alias("frequency")).orderBy(F.col("DISTRICT"),F.col("frequency").desc());

# Создаем столбец "frequent_crime_types":
Df_Crimes_Frequent = Df_Crimes_Frequency.groupBy("DISTRICT").agg(F.concat_ws(", ",F.collect_list("crime_type")[0],F.collect_list("crime_type")[1],F.collect_list("crime_type")[2]).alias("frequent_crime_types")).orderBy("DISTRICT");

# Создаем столбец "lat":
Df_Crimes_lat = Df_Crimes_Codes.groupBy("DISTRICT").agg(F.avg("lat").alias("lat"));

# Создаем столбец "lng":
Df_Crimes_lng = Df_Crimes_Codes.groupBy("DISTRICT").agg(F.avg("long").alias("lng"));

# Создаем итоговый датафрейм:
Df_Crimes_result = Df_Crimes_Total.join(Df_Crimes_Monthly_Mediana,["DISTRICT"]).join(Df_Crimes_Frequent,["DISTRICT"]).join(Df_Crimes_lat,["DISTRICT"]).join(Df_Crimes_lng,["DISTRICT"]);

# Записываем вывод датафрейма в файл в формате parquet:
Df_Crimes_result.write.format("parquet").mode("overwrite").option("path",args.output_path).saveAsTable("Df_Crimes_result");

# Для проверки вычитываем датафрейм из созданного файла и выводим в терминал:
spark.read.parquet(args.output_path).show();


# In[42]:





# In[ ]:




