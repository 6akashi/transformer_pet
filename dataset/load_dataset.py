import pandas as pd

df = pd.read_csv('archive/IMDB Dataset.csv')
texts = df['review'].tolist()
# Лейблы в датасете — строки ("positive"/"negative"), модель ждёт индексы классов.
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
