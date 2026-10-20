from predict_finetuned import predict_fine_tuned, baseline_model, baseline_vectorizer, predict_baseline
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long."
]
model_name = "distilbert-base-uncased"
model_path = './finetuned_model'
# 1. Загружаем токенайзер
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_distilb = AutoModel.from_pretrained(model_name)

# 2. Загружаем саму модель (если это классификатор)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

preds_ft = predict_fine_tuned(test_texts, model, tokenizer)

if baseline_model:
      preds_bl = predict_baseline(test_texts, baseline_model, baseline_vectorizer, model_distilb)

for i, text in enumerate(test_texts):
      print(f'\nТекст: {text}')
      print(f'Fine-tuned: {"positive" if preds_ft[i]["prediction"] == 1 else "negative"} (probs: {preds_ft[i]["probabilities"]})')
      if baseline_model:
          print(f'Baseline: {preds_bl[i]}')
          print(f'Совпадают: {("positive" if preds_ft[i]["prediction"] == 1 else "negative")== preds_bl[i]}')

