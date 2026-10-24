import matplotlib
matplotlib.use('Agg')

import pandas as pd
from sklearn.model_selection import train_test_split
from predict_finetuned import predict_fine_tuned
from test_predict import model, tokenizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./dataset/archive/IMDB Dataset.csv').head(1000)
_, test_df, _, _ = train_test_split(df, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])

test_texts = test_df['review'].tolist()
test_labels = test_df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()

preds_ft_all = predict_fine_tuned(test_texts, model, tokenizer)
y_pred_ft = [p['prediction'] for p in preds_ft_all]

cm_ft = confusion_matrix(test_labels, y_pred_ft)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Fine-tuned Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_finetuned.png')
