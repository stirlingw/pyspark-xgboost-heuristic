
import os
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer, Bucketizer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import xgboost as xgb
import sklearn

os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'

conf = SparkConf().setMaster("local[2]").setAppName("xgbooster").set("spark.executor.memory", "6g").set("spark.driver.memory", "6g") 
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
sc.addPyFile("sparkxgb.zip")
spark = SparkSession.builder.appName("spark play").getOrCreate()


class XGBoostEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol):
    def __init__(self, xgb_param_map={}):
        super(XGBoostEstimator, self).__init__()
        sc = SparkContext._active_spark_context
        scala_map = sc._jvm.PythonUtils.toScalaMap(xgb_param_map)
        self._defaultParamMap = xgb_param_map
        self._paramMap = xgb_param_map
        self._from_XGBParamMap_to_params()
        self._java_obj = self._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator", self.uid, scala_map)

    def _create_model(self, java_model):
        return JavaModel(java_model)

    def _from_XGBParamMap_to_params(self):
        for param, value in self._paramMap.items():
            setattr(self, param, value)


gender_csv = "gender_submission.csv"
test_csv = "test.csv"
train_csv = "train.csv"

gender_df = spark.read.csv(gender_csv, header=True, mode="DROPMALFORMED", inferSchema='true', encoding="utf-8").persist()
testing_df = spark.read.csv(test_csv, header=True, mode="DROPMALFORMED", inferSchema='true', encoding="utf-8").persist()
training_df = spark.read.csv(train_csv, header=True, mode="DROPMALFORMED", inferSchema='true', encoding="utf-8").persist()


training_features = (training_df
                    .withColumn("Surname", regexp_extract(col("Name"),"([\\w ']+),",1))
                    .withColumn("Honorific", regexp_extract(col("Name"),"(.*?)([\\w]+?)[.]",2))
                    .withColumn("Mil", when((col("Honorific") == "Col") | (col("Honorific") == "Major") | (col("Honorific") == "Capt"), 1).otherwise(0))
                    .withColumn("Doc", when(col("Honorific") == "Dr", 1).otherwise(0))
                    .withColumn("Rev", when(col("Honorific") == "Rev", 1).otherwise(0))
                    .withColumn("Nob", when((col("Honorific") == "Sir") |
                        (col("Honorific") == "Countess") |
                        (col("Honorific") == "Count") |
                        (col("Honorific") == "Duke") |
                        (col("Honorific") == "Duchess") |
                        (col("Honorific") == "Jonkheer") |
                        (col("Honorific") == "Don") |
                        (col("Honorific") == "Dona") |
                        (col("Honorific") == "Lord") |
                        (col("Honorific") == "Lady") |
                        (col("Honorific") == "Earl") |
                        (col("Honorific") == "Baron"), 1).otherwise(0))
                    .withColumn("Mr", when(col("Honorific") == "Mr", 1).otherwise(0))
                    .withColumn("Mrs", when((col("Honorific") == "Mrs") | (col("Honorific") == "Mme"), 1).otherwise(0))
                    .withColumn("Miss", when((col("Honorific") == "Miss") | (col("Honorific") == "Mlle"), 1).otherwise(0))
                    .withColumn("Mstr", when(col("Honorific") == "Master", 1).otherwise(0))
                    .withColumn("TotalFamSize", col("SibSp") + col("Parch") + 1)
                    .withColumn("Singleton", when(col("TotalFamSize") == 1, 1).otherwise(0))
                    .withColumn("SmallFam", when((col("TotalFamSize") <= 4) & (col("TotalFamSize") > 1), 1).otherwise(0))
                    .withColumn("LargeFam", when(col("TotalFamSize") >= 5, 1).otherwise(0))
                    .withColumn("Child", when((col("Age") <= 18), 1).otherwise(0))
                    .withColumn("Mother", when((col("Age") > 15) &
                        (col("Parch") > 0) & 
                        (col("Miss") == 0) & 
                        (col("Sex") == "female"),1).otherwise(0)))
training_features.show()


# // Explore the data
(training_features
  .groupBy("Pclass","Embarked")
  .agg(count("*"),avg("Fare"),min("Fare"),max("Fare"),stddev("Fare"))
  .orderBy("Pclass","Embarked")
  .show())

training_features.createOrReplaceTempView("training_features")


spark.sql("SELECT Pclass,Embarked,percentile_approx(Fare, 0.5) AS Median_Fare FROM training_features WHERE Fare IS NOT NULL AND Pclass = 1 GROUP BY Pclass,Embarked").show()


# // Impute Embarked column
# // From the discovery above, the likely port is C, since the Median for C is closest to 80.
train_embarked = training_features.na.fill("C", ["Embarked"])
print(train_embarked.show())


# // Perform discovery on missing data in Age column
# // Create the temp table view so we can perform spark.sql queries on the dataframe
train_embarked.createOrReplaceTempView("train_embarked")

# // Explore the data
# // Count nulls for each Honorific.  Some titles can imply age (miss,master,etc)
spark.sql("SELECT Honorific,count(*) as nullAge FROM train_embarked WHERE Age IS NULL GROUP BY Honorific").show()


# // Calculate the average age for the Honorific titles that have nulls
spark.sql("SELECT Honorific,round(avg(Age)) as avgAge FROM train_embarked WHERE Age IS NOT NULL AND Honorific IN ('Miss','Master','Mr','Dr','Mrs') GROUP BY Honorific").show()


# // Impute the missing Age values for the relevant Honorific columns and union the data back together
train_miss_df = train_embarked.na.fill(22.0).where("Honorific = 'Miss'")
train_master_df = train_embarked.na.fill(5.0).where("Honorific = 'Master'")
train_mr_df = train_embarked.na.fill(32.0).where("Honorific = 'Mr'")
train_dr_df = train_embarked.na.fill(42.0).where("Honorific = 'Dr'")
train_mrs_df = train_embarked.na.fill(36.0).where("Honorific = 'Mrs'")
train_remainder_df = spark.sql("SELECT * FROM train_embarked WHERE Honorific NOT IN ('Miss','Master','Dr','Mr','Mrs')")
train_combined_df = train_remainder_df.union(train_miss_df).union(train_master_df).union(train_mr_df).union(train_dr_df).union(train_mrs_df)


print(train_combined_df.show())

# // Convert the categorical (string) values into numeric values
# // Convert the categorical (string) values into numeric values
gender_indexer = StringIndexer(inputCol="Sex", outputCol="SexIndex").setHandleInvalid("keep")
embark_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkIndex").setHandleInvalid("keep")

# // Convert the numerical index columns into One Hot columns
# // The One Hot columns are binary {0,1} values of the categories
gender_encoder = OneHotEncoder(dropLast=False, inputCol="SexIndex", outputCol="SexVec")
embark_encoder = OneHotEncoder(dropLast=False, inputCol="EmbarkIndex", outputCol="EmbarkVec")


# // Create 8 buckets for the fares, turning a continuous feature into a discrete range# // Cre 
fare_splits = [0.0,10.0,20.0,30.0,40.0,60.0,120.0, float("+inf")]
fare_bucketize = Bucketizer().setInputCol("Fare").setOutputCol("FareBucketed").setSplits(fare_splits)

# // Create a vector of the features.  
assembler = VectorAssembler().setInputCols(["Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "FareBucketed", "EmbarkVec", "Mil", "Doc", "Rev", "Nob", "Mr", "Mrs", "Miss", "Mstr", "TotalFamSize", "Singleton", "SmallFam", "LargeFam", "Child", "Mother"]).setOutputCol("features")

# // Create the features pipeline and data frame
# // The order is important here, Indexers have to come before the encoders
training_features_pipeline = (Pipeline().setStages([gender_indexer, embark_indexer, gender_encoder, embark_encoder, fare_bucketize, assembler]))
training_features_df = training_features_pipeline.fit(train_combined_df).transform(train_combined_df)


# // Now that the data has been prepared, let's split the dataset into a training and test dataframe
train_df, test_df = training_features_df.randomSplit([0.8, 0.2], seed = 12345)
train_df.show(2)

params = {}
params["eta"] = 0.1
params["max_depth"] = 8
params["gamma"] = 0.0
params["colsample_bylevel"] = 1
params["objective"] = "binary:logistic"
params["num_class"] = 2
params["booster"] = "gbtree"
params["num_rounds"] = 20
params["nWorkers"] = 3

# // Create an XGBoost Classifier
xgbEstimator = XGBoostEstimator(params).setFeaturesCol("features").setLabelCol("Survived")

# // XGBoost paramater grid
xgbParamGrid = ParamGridBuilder().addGrid(xgbEstimator.max_depth, [16]).addGrid(xgbEstimator.eta, [0.015]).build()

# // Create the XGBoost pipeline
pipeline = Pipeline().setStages([xgbEstimator])

# // Setup the binary classifier evaluator
evaluator = BinaryClassificationEvaluator().setLabelCol("Survived").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(xgbParamGrid).setNumFolds(10)
xgb_model = cv.fit(train_df)
results = xgb_model.transform(test_df)

results.createOrReplaceTempView("results")
spark.sql("SELECT PassengerID as PID,Pclass,Sex,Age,SibSp,Parch,Honorific as Hon,TotalFamSize as Fam,Survived,prediction,probabilities FROM results where Survived != cast(prediction as int)").show(100)

# // What was the overall accuracy of the model, using AUC
evaluator.evaluate(results)

