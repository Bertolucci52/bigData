from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# start
spark = SparkSession.builder \
    .appName("Etivity4_SparkMLlib") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Versione di Spark:", spark.version)

# importo il file txt
libsvm_path = "../dataset/sample_libsvm_data.txt"
data = spark.read.format("libsvm").load(libsvm_path)

print("\nSchema del dataset:")
data.printSchema()

print("\nPrime righe del dataset:")
data.show(10, truncate=False)

# ripartisco il dataset in 70% training set e 30% test
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

print(f"\nNumero di righe training: {train_data.count()}")
print(f"Numero di righe test: {test_data.count()}")

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

print("\n=== PUNTO 1: REGRESSIONE LINEARE PER LA CLASSIFICAZIONE ===")

# Modello di regressione lineare
lin_reg = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.0
)

# Addestramento 
model_lin = lin_reg.fit(train_data)

print("\nCoefficiente e intercetta del modello di regressione lineare:")
print("Coefficients:", model_lin.coefficients)
print("Intercept:", model_lin.intercept)

# Predizioni sul test set 
pred_lin = model_lin.transform(test_data)

print("\nPredizioni continue (prime righe):")
pred_lin.select("label", "prediction").show(10, truncate=False)

# Trasformo le predizioni continue in classi 0/1 con una soglia 
pred_lin_class = pred_lin.withColumn(
    "prediction",
    when(pred_lin["prediction"] >= 0.5, 1.0).otherwise(0.0)
)

print("\nPredizioni di classe (prime righe) ottenute dalla regressione lineare:")
pred_lin_class.select("label", "prediction").show(10, truncate=False)

# accuratezza
accuracy_lin = evaluator.evaluate(pred_lin_class)
print(f"\nAccuracy della regressione lineare usata per classificare: {accuracy_lin:.4f}")

print("\n=== PUNTO 2: CLASSIFICAZIONE CON ALBERO DI DECISIONE ===")

# Modello di classificazione
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5
)

# Addestramento 
model_dt = dt.fit(train_data)

# Stampa della struttura dell'albero
print("\nStruttura dell'albero di decisione:")
print(model_dt.toDebugString)

# Predizioni sul test set
pred_dt = model_dt.transform(test_data)

print("\nPredizioni dell'albero di decisione (prime righe):")
pred_dt.select("label", "prediction", "probability").show(10, truncate=False)

# accuratezza
accuracy_dt = evaluator.evaluate(pred_dt)
print(f"\nAccuracy della classificazione con albero di decisione: {accuracy_dt:.4f}")

#end
spark.stop()
print("\nSessione Spark terminata.")