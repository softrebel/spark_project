# پیش پردازش
برای پیش‌پردازش داده، چند قدم لازم است:
1. بررسی مقادیر گم‌شده (missing values)
2. تبدیل ویژگی طبقه به عددی
3. مهندسی ویژگی

در اینجا مورد اول و دوم را بررسی می‌کنیم. همچنین مورد سوم یعنی مهندسی ویژگی به علت اهمیت ویژه آن و تفاوت syntax کتابخانه pyspark در صفحه جداگانه شرح خواهیم داد.
ویژگی‌های داده به دو نوع باینری و اعشاری تقسیم می‌شود (از طبقه و ویژگی‌های هویتی صرف نظر کرده‌ایم). 
```python

float_cols=[
    'cmp_fname_c1', 
    'cmp_fname_c2', 
    'cmp_lname_c1', 
    'cmp_lname_c2', 
    ]
    
binary_cols=[
        'cmp_sex', 
        'cmp_bd', 
        'cmp_bm', 
        'cmp_by',
        'cmp_plz',
    ]
```

در صورتی که بخواهیم تمام مقادیر گم‌شده را حذف کنیم، تنها 20 رکورد باقی خواهد ماند. لذا باید مقادیر گم‌شده را با یک مکانیزم مناسبی پر کنیم. 
در این پروژه از روش میانگین برای پر کردن داده‌های اعتشاری و از روش مد برای پر کردن داده‌های باینری استفاده می‌کنیم.
همچنین یک تابع جهت تبدیل رشته به باینری ویژگی is_matched که متغیر طبقه یا برچسب است تعریف می‌کنیم.   

```python
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when, lit
# for float variables



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

```
