# پیاده‌سازی پروژه یادگیری ماشینی با pyspark
در این پروژه قصد داریم مجموعه دستورات لازم برای انجام یک طبقه‌بندی باینری ساده را در pyspark بحث و بررسی کنیم.


## مجموعه داده
برای اطلاعات بیشتر به صفحه [توضیحات داده](./data_description.md) مراجعه کنید.
## بارگذاری داده
راهنمایی و کد در صفحه [بارگذاری داده](./data_load.md) قرار داده شده است.
## پیش‌پردازش
صفحه [پیش‌پردازش داده](./data_preprocess.md)
## مهندسی ویژگی
صفحه [مهندسی ویژگی](./feature_engineering.md)

## مدل‌سازی
صفحه [مدل‌سازی](./build_model.md)


## معیارهای ارزیابی
در این پروژه به علت نامتوازن بودن داده‌های طبقات (imbalanced data) از معیارهای ارزیابی F1(macro), ROC و AUC 
استفاده شده است. برای اطلاعات بیشتر به صفحه [معیارهای ارزیابی](./evaluation_metrics.md) مراجعه کنید

## مراحل انجام کار 
  
  
<div dir="rtl">
مراحل انجام در فلوچارت زیر آمده است:
</div

  
### ML Flowchart

```mermaid
  graph TD;
      A(Load Model)-->B(Preprocessing);
      B-->C(train,eval,test split);
      C-->D(Train ML model on train set);
      D --> E{hyper parameter tuning?};
      E -- yes --> F(Evaluate by eval set with respect of imbalanced data);
      F -->D;
      E -- no --> G(evaluate on test set);
      G --> H(Report);
```
