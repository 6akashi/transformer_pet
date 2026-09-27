from transformers import AutoModelForSequenceClassification
from load_dataset import labels

# Определите количество классов
num_labels = len(set(labels))  # 2 для бинарной, 3 для positive/negative/neutral
model_name = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
