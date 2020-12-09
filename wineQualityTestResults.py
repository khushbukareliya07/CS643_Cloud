# import findspark
# findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import pyspark.sql.functions as func
import pyspark
if(len(sys.argv) < 3):
    print("Please provide testfilepth")
    sys.exit(-1)

modelpath=sys.argv[1]
testfilepath = sys.argv[2] 


conf = SparkConf().setAppName("Wine Quality Testing").setMaster("local[1]")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

rf = RandomForestClassifier.load(modelpath)
#rf = RandomForestClassifier.load("s3://myprogrambucket/rfwine_model.model")

defTest = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv(testfilepath)

#defTest = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv(
 #   "s3://myprogrambucket/ValidationDataset.csv")
defTest.printSchema()

featureColumns = [col for col in defTest.columns if (
    col != '""""quality"""""')]

assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')

rfPipeline = Pipeline(stages=[assembler, rf])

fit = rfPipeline.fit(defTest)
transformed = fit.transform(defTest)
transformed = transformed.withColumn(
    "prediction", func.round(transformed['prediction']))
transformed = transformed.withColumn('""""quality"""""', transformed['""""quality"""""'].cast(
    'double')).withColumnRenamed('""""quality"""""', "label")

results = transformed.select(['prediction', 'label'])
predictionAndLabels = results.rdd

metrics = MulticlassMetrics(predictionAndLabels)
predictionAndLabels.take(5)

# Calculate the Accuracy, Precision and Recall

"""
cm = metrics.confusionMatrix().toArray()
print(cm)

print("(Parameter, Precision, Recall, Accuracy, F1 Score)")
for i in range(3, 9):
    if metrics.precision(i) != 0:
        print(i, ", ", round(metrics.precision(i), 2), ", ", round(metrics.recall(i), 2),
              ", ", round(metrics.accuracy, 2), ", ", round(metrics.fMeasure(float(i), 1.0), 2))
"""

print("F1 score is %s"%(metrics.accuracy))
