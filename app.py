import os

import gradio as gr
import joblib
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# FT
model_name = "distilbert-base-uncased"
model_path = './finetuned_model'


tokenizer = AutoTokenizer.from_pretrained(model_path)
model_ft = AutoModelForSequenceClassification.from_pretrained(model_path)
def predict_sentiment(text):
    """
    Функция для Gradio интерфейса
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    label_map = {0: 'Negative', 1: 'Positive'}

    # Форматируем вывод вероятностей
    result = f"Prediction: {label_map.get(pred, pred)}\n\n"
    result += "Probabilities:\n"

    for i, prob in enumerate(probs[0]):
        result += f"{label_map.get(i, i)}: {prob*100:.2f}%\n"

    return result

# Создаём интерфейс
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с BERT",
    description="Введите текст и модель определит его тональность"
)

if __name__ == "__main__":
    demo.launch()