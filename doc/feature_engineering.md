# مهندسی ویژگی
در کتابخانه pyspark نیاز است که تمام ویژگی‌ها در یک ستون به صورت برداری تجمیع یابد تا عملیات فقط بر روی یک ستون انجام شود؛
یعنی در هنگام تعریف الگوریتم یادگیری ماشینی، فقط نام ستون ویژگی(که برداری از ویژگی‌ها است) و نام ستون برچسب را می‌دهیم.


```python

from pyspark.ml.feature import VectorAssembler

def feature_engineering(input_df,feature_list,label_name):
    assembler = VectorAssembler(inputCols=feature_list,
                             outputCol='features')
    assembled_df = assembler.transform(input_df)
    output_df=assembled_df.select('features', label_name)
    return output_df



features=list(set(prep_df.columns) - set(['label','is_match']))
assembled_df = feature_engineering(prep_df,features,'label')
```
