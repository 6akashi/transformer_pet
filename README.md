# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности текстов с использованием трансформеров.

## Структура
- `fine_tuned_model/` — обученная модель
- `app.py` — Gradio приложение
- `error_analysis.txt` — анализ ошибок
- `comparison_results.txt` — сравнение моделей

## Запуск демо

### Вариант с Gradio:
```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

## Результаты

### Fine-tuned модель:
- F1 (macro): 0.9850
- Accuracy: 0.9850

### Baseline модель:
- F1 (macro): 0.900
- Accuracy: 0.9000

Улучшение: 0.45%

## Использование в коде
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Требования
- Python 3.8+
- transformers
- torch
- gradio (для демо)