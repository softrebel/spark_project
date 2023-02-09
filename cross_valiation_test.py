from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType,IntegerType,FloatType,BooleanType

conf = SparkConf()\
    .setMaster("local[*]")\
        .setAppName("My App")
sc = SparkContext.getOrCreate(conf = conf)
sc._conf.set('spark.executor.memory','15g')\
    .set('spark.driver.memory','15g')\
        .set('spark.driver.maxResultsSize','0')
spark=SparkSession.builder\
    .appName('myApp')\
        .config("spark.driver.memory", "15g")\
            .getOrCreate()







from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when, lit
# for float variables
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier,LogisticRegression,RandomForestClassifier


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np




def get_available_models():
    return {
        'rf':RandomForestClassifier,
        'lr':LogisticRegression,
        'dt':DecisionTreeClassifier,

    }
def evaluate_from_scratch(pred,model_name='Logistic Regression'):
    pred.groupBy('label', 'prediction').count().show()

    # Calculate the elements of the confusion matrix
    TN = pred.filter('prediction = 0 AND label = prediction').count()
    TP = pred.filter('prediction = 1 AND label = prediction').count()
    FN = pred.filter('prediction = 0 AND label = 1').count()
    FP = pred.filter('prediction = 1 AND label = 0').count()

    # Accuracy measures the proportion of correct predictions
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    recall = (TP) / (TP+FN)
    precision= (TP) / (TP+FP)
    f1=2*(precision*recall)/(precision+recall)
    print(f'EVALUATION SUMMARY for {model_name}:')
    print(f" accuracy:{accuracy}")
    print(f" precision:{precision}")
    print(f" recall:{recall}")
    print(f" f1-score:{f1}")

def evaluate_from_spark(predictions,model_name='Logistic Regression'):
    eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label")
    eval2= MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
    AUC  = eval.evaluate(predictions)
    ACC  = eval2.evaluate(predictions, {eval2.metricName:"accuracy"})
    PREC  = eval2.evaluate(predictions, {eval2.metricName:"weightedPrecision"})
    REC  = eval2.evaluate(predictions, {eval2.metricName:"weightedRecall"})
    F1  = eval2.evaluate(predictions, {eval2.metricName:"f1"})
    WeightedFMeasure=eval2.evaluate(predictions, {eval2.metricName:"weightedFMeasure"})
    print(f"{model_name} Performance Measure")
    print(" Accuracy = %0.8f" % ACC)
    print(" Weighted Precision = %0.8f" % PREC)
    print(" Weighted Recall = %0.8f" % REC)
    print(" F1 = %0.8f" % F1)
    print(" Weighted F Measure = %0.8f" % WeightedFMeasure)

    print(" AUC = %.8f" % AUC)
    print(" ROC curve:")
    PredAndLabels           = predictions.select("probability", "label")
    PredAndLabels_collect   = PredAndLabels.collect()
    PredAndLabels_list      = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect]
    PredAndLabels           = sc.parallelize(PredAndLabels_list)
    fpr = dict()                                                        # FPR: False Positive Rate
    tpr = dict()                                                        # TPR: True Positive Rate
    roc_auc = dict()

    y_test = [i[1] for i in PredAndLabels_list]
    y_score = [i[0] for i in PredAndLabels_list]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.8f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0,1.03,0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'{model_name}_roc.png')
    plt.close()



def evaluate(predictions,model_name=None):
    print('Evaluate From Scratch:')
    evaluate_from_scratch(predictions,model_name)
    print('\nEvaluate From Spark Library:')
    evaluate_from_spark(predictions,model_name)
def feature_engineering(input_df,feature_list,label_name):
    assembler = VectorAssembler(inputCols=feature_list,
                             outputCol='features')
    assembled_df = assembler.transform(input_df)
    output_df=assembled_df.select('features', label_name)
    return output_df


def convert_label_binary(input_df):
    temp = input_df.withColumn('label',
                             when(input_df['is_match']==True,
                                  lit(1)).otherwise(0)
                                  )
    return temp

def fill_missing_values(input_df):
    miss_df=input_df.drop('id_1','id_2')
    miss_df=miss_df.replace('?',None)
    float_cols=[
    'cmp_fname_c1',
    'cmp_fname_c2',
    'cmp_lname_c1',
    'cmp_lname_c2',
    ]
    float_imputer = Imputer(
        inputCols=float_cols,
        outputCols=[f"{col}_imputed" for col in float_cols]
    ).setStrategy('mean')

    # for binary variables
    binary_cols=[
        'cmp_sex',
        'cmp_bd',
        'cmp_bm',
        'cmp_by',
        'cmp_plz',
    ]
    binary_imputer = Imputer(
        inputCols=binary_cols,
        outputCols=[f"{col}_imputed" for col in binary_cols]
    ).setStrategy('mode')
    imputed_df=float_imputer.fit(miss_df).transform(miss_df)
    output_df=binary_imputer.fit(imputed_df).transform(imputed_df)
    output_df=output_df.select([x for x in output_df.columns if '_imputed' in x or x=='is_match'])
    return output_df


def preprocessing_df(input_df):
    return convert_label_binary(fill_missing_values(input_df))

def load_data(files,schema):
    df=spark.read.csv(files,header=True
                  ,schema=schema)
    return df

def load_record_linkage_data():
    schema = StructType() \
      .add("id_1",IntegerType(),True) \
      .add("id_2",IntegerType(),True) \
      .add("cmp_fname_c1",FloatType(),True) \
      .add("cmp_fname_c2",FloatType(),True) \
      .add("cmp_lname_c1",FloatType(),True) \
      .add("cmp_lname_c2",FloatType(),True) \
      .add("cmp_sex",IntegerType(),True) \
      .add("cmp_bd",IntegerType(),True) \
      .add("cmp_bm",IntegerType(),True) \
      .add("cmp_by",IntegerType(),True) \
      .add("cmp_plz",IntegerType(),True) \
      .add("is_match",BooleanType(),False)
    files=[f'./data/block_{id}.csv' for id in range(1,11)]
    return load_data(files,schema=schema)

def test_train_split(input_df,train_size=0.7):
    train, test = input_df.randomSplit([train_size,1 - train_size], seed=42)
    return train,test

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import argparse
p = argparse.ArgumentParser()
p.add_argument('--model', default='lr')

if __name__ == "__main__":
    args = p.parse_args()
    models=get_available_models()
    print('classification with:',args.model)
    current_model=models[args.model]
    df=load_record_linkage_data()
    miss_df=df.drop('id_1','id_2')
    miss_df=miss_df.replace('?',None)
    prep_df=preprocessing_df(df)

    features=list(set(prep_df.columns) - set(['label','is_match']))
    assembled_df = feature_engineering(prep_df,features,'label')
    print(assembled_df.show(1,truncate=False))
    train, test = test_train_split(assembled_df,0.7)
    print(train.show(1,truncate=False))
    

    # model_cs = current_model()
    # pipeline = Pipeline(stages=[model_cs])
    # grid = ParamGridBuilder().addGrid(model_cs.regParam, [0.0,0.1]) \
    #     .addGrid(model_cs.elasticNetParam, [0.0, 1.0])\
    #     .build()
    # evaluator = BinaryClassificationEvaluator()
    # cv = CrossValidator(estimator=pipeline,
    #                  estimatorParamMaps=grid,
    #                  evaluator=evaluator,
    #                  numFolds=3)
    # cvModel = cv.fit(train)
    # lrprediction=cvModel.transform(test)
    # evaluate(lrprediction)
    # print('Accuracy:', evaluator.evaluate(lrprediction))
    # print('AUC:', BinaryClassificationMetrics(lrprediction['label','prediction'].rdd).areaUnderROC)
    # spark.stop()
