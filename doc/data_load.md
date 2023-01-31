# بارگذاری داده
ابتدا فایل فشرده دادگان را از لینک زیر دانلود کرده و سپس در پوشه data پروژه استخراج می‌کنیم.

https://archive.ics.uci.edu/ml/machine-learning-databases/00210/

سپس برای بارگذاری داده از قطعه کد زیر استفاده می‌کنیم.

```python
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType,IntegerType,FloatType,BooleanType
conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext.getOrCreate(conf = conf)
spark=SparkSession.builder.appName('myApp').getOrCreate()

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
    
  
df=load_record_linkage_data()
```
