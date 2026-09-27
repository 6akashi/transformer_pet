from SentimalDataset import SentimalDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from load_dataset import texts, labels

train_texts, val_texts, train_labels, val_labels = train_test_split(texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = SentimalDataset(train_texts, train_labels, tokenizer=tokenizer)
val_dataset = SentimalDataset(val_texts, val_labels, tokenizer=tokenizer)
