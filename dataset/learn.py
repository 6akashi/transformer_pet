import torch
from torch.optim import AdamW
from prepare_data import tokenizer
from sklearn.metrics import accuracy_score, f1_score
from model import model
from data_loader import val_loader, train_loader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
device = "cpu"
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, f1


num_epochs = 3

# Инициализируем, чтобы финальный блок ниже всегда видел эти переменные
val_accuracy = 0.0
val_f1 = 0.0

for epoch in range(num_epochs):
    train_loss = train_epoch(model,
        train_loader,
        optimizer,
        device)
    val_accuracy, val_f1 = evaluate(model, val_loader, device)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
    print('-' * 50)

model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')

with open('fine_tuned_results.txt', 'w') as f:
    f.write(f"Final Validation F1: {val_f1:.4f}\n")
    f.write(f"Final Validation Accuracy: {val_accuracy:.4f}\n")
