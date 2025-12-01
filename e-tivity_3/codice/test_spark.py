from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as _sum, max as _max, when
import json

# start
spark = SparkSession.builder \
    .appName("Etivity3_SparkSQL") \
    .master("local[*]") \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("ERROR")

print("Versione di Spark:", spark.version)

# creazione di un df da una lista Python
dati_impiegati = [
    ("Roberto", 35, "M", 3000.0),
    ("Giovanni",   30, "M", 4000.0),
    ("Michela", 27, "F", 4100.0),
    ("Pietro",   37, "M", 3500.0),
    ("Roberta", 31,"F", 8500.0)
]

colonne = ["Nome", "Eta", "Genere", "Stipendio"]

df_lista = spark.createDataFrame(dati_impiegati, schema=colonne)

print("\nPunto 2.a --> DataFrame creato da Python: ")
df_lista.show()

# creazione di  df da CSV 
contenuto_csv = """Nome,Eta,Genere,Stipendio
Roberto,35,M,3000.0
Giovanni,30,M,4000.0
Michela,27,F,4100.0
Pietro,37,M,3500.0
Roberta,31,F,8500.0
"""

with open("impiegati.csv", "w", encoding="utf-8") as f:
    f.write(contenuto_csv)

df_csv = spark.read.csv("impiegati.csv", header=True, inferSchema=True)

print("\nPunto 2.b --> DataFrame creato da file csv:")
df_csv.show()

# creazione df da JSON 
dati_json = [
    {"Nome": "Roberto", "Eta": 35, "Genere": "M", "Stipendio": 3000.0},
    {"Nome": "Giovanni", "Eta": 30, "Genere": "M", "Stipendio": 4000.0},
    {"Nome": "Michela", "Eta": 27, "Genere": "F", "Stipendio": 4100.0},
    {"Nome": "Pietro", "Eta": 37, "Genere": "M", "Stipendio": 3500.0},
    {"Nome": "Roberta", "Eta": 31, "Genere": "F", "Stipendio": 8500.0}
]

with open("impiegati.json", "w", encoding="utf-8") as f:
    for r in dati_json:
        f.write(json.dumps(r) + "\n")

df_json = spark.read.json("impiegati.json")

print("\nPunto 2.c --> DataFrame creato da JSON:")
df_json.show()

# struttura
print("\nSchema del DataFrame:")
df_lista.printSchema()

# seleziono le colonne nome ed età
print("\nSelect delle colonne Nome ed Eta:")
df_lista.select("Nome", "Eta").show()

# creo una nuova colonna Reparto su cui eseguire il raggruppamento
df_con_reparto = df_lista.withColumn(
    "Reparto",
    when(col("Genere") == "M", "HW")   
    .when(col("Genere") == "F", "SW")  
    .otherwise("Altro")
)

print("\nDataFrame con colonna Reparto:")
df_con_reparto.show()

print("\nStipendio medio e massimo suddiviso per Reparto:")
df_con_reparto.groupBy("Reparto").agg(
    avg("Stipendio").alias("Stipendio_medio"),
    _max("Stipendio").alias("Stipendio_max")
).show()

# seleziono i soli record che hanno uno stipendio maggiore di 3.500
print("\nImpiegati con Stipendio > 3500:")
df_lista.select("Nome", "Stipendio") \
       .filter(col("Stipendio") > 3500) \
       .show()

# ordine decrescente
print("\nImpiegati ordinati per Stipendio decrescente:")
df_lista.orderBy(col("Stipendio").desc()).show()

# SQL tramite SparkSQL
df_lista.createOrReplaceTempView("impiegati")

print("\nQuery SQL: impiegati con Eta > 28, ordinati per Stipendio decrescente")
query = """
SELECT Nome, Eta, Stipendio
FROM impiegati
WHERE Eta > 28
ORDER BY Stipendio DESC
"""
df_sql = spark.sql(query)
df_sql.show()

# end
spark.stop()
print("\nSessione Spark terminata.")
