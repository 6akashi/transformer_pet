#%%
from numpy.matlib import ma
from numpy.random.mtrand import sample

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
#%%
model_name = "distilbert-base-uncased"

#%%
# ###Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

#%%
#
def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

#%%
sample_texts = [
    "This movie was great!",
    "I didn't like the plot.",
    "Terrible experience.",
    "Terrible film."
]

#%%
sample_tokens = tokenize_texts(sample_texts)
print(f"Sample tokens shape: {sample_tokens['input_ids'].shape}")

#%%
# ###GET CLS EMBEDDINGS
def get_cls_embeddings(texts, batch_size=32, max_length=128):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenize_texts(batch_texts, max_length=max_length)

            outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

#%%
test_embeddings = get_cls_embeddings(sample_texts, batch_size=2)
print(f"Test embeddings shape: {test_embeddings.shape}")

#%%
# ###LOGISTIC REGRESSION
from datasets import load_dataset

#%%
dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
#%%
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

texts = train_df["text"].tolist() + test_df["text"].tolist()
y = train_df["label"].values.tolist() + test_df["label"].values.tolist()

X = get_cls_embeddings(texts, batch_size=32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
clf.fit(X_train, y_train)

#%%
y_pred = clf.predict(X_test)

#%%
report = classification_report(y_test, y_pred)
print(report)

#%%
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1: {macro_f1:.4f}")

#%%
results_text = f"Basline (DistilBERT + Logistic Regression)\nMacro F1: {macro_f1:.4f}"
with open("results.txt", "w") as f:
    f.write(results_text)
