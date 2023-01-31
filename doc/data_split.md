# تقسیم داده
به طور کلی به نسبت 0.7 داده‌ها را به دو مجموعه آموزش و آزمون تقسیم می‌کنیم؛ یعنی 70% داده‌ها در مجموعه آموزش و 30% دیگر به عنوان مجموعه آزمون استفاده می‌شود.

```python
def test_train_split(input_df,train_size=0.7):
    train, test = assembled_df.randomSplit([train_size,1 - train_size], seed=42)
    return train,test
    
 
train, test = test_train_split(assembled_df,0.7)
```
