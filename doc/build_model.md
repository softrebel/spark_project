# مدل‌سازی
در این پروژه سه الگوریتم زیر برای ساخت مدل یادگیری ماشینی در نظر گرفته شده است:
1. Logistic Regression
2. Decision Tree
3. Random Forest

ساختار کلی ایجاد مدل با استفاده از pipeline کتابخانه pyspark به صورت زیر است:

```python

lr=[Estimator](featuresCol='features', labelCol='label')
pipeline = Pipeline(stages=[lr])
model = pipeline.fit(train)
result = model.transform(test)

```
