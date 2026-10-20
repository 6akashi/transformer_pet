import pandas as pd

df = pd.read_csv('/home/semyon/ml_education/Transformers/dataset/archive/IMDB Dataset.csv').head(5000)
texts = df['review'].tolist()
# Лейблы в датасете — строки ("positive"/"negative"), модель ждёт индексы классов.
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
