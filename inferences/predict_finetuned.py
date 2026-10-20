import joblib
import torch
import numpy as np
import os
from transformers import AutoTokenizer

# Узнаем, в какой папке физически лежит этот файл (predict_finetuned.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Строим точный путь к папке notebooks, поднимаясь из inferences на уровень выше
notebooks_dir = os.path.abspath(os.path.join(script_dir, "..", "notebooks"))

# Загружаем модели по абсолютным путям
baseline_model_path = os.path.join(notebooks_dir, "baseline_model.pkl")
baseline_vectorizer_path = os.path.join(notebooks_dir, "baseline_vectorizer.pkl")

baseline_model = joblib.load(baseline_model_path)
baseline_vectorizer = joblib.load(baseline_vectorizer_path)






def predict_fine_tuned(texts, model, tokenizer):
      if isinstance(texts, str):
            texts = [texts]

      prediction = []

      for text in texts:
            inputs = tokenizer(
                  text,
                  return_tensors = "pt",
                  truncation=True,
                  max_length=128
            ) 

            with torch.no_grad():
                  outputs = model(**inputs)

            probs = torch.nn.functional.softmax(
                  outputs.logits,
                  dim=1
                  )

            pred = torch.argmax(probs, dim=1).item()

            prediction.append({
                  'text': text,
                  'prediction': pred,
                  'probabilities': probs[0].cpu().numpy()}
            )

      return prediction

def predict_baseline(texts, baseline_logreg_model, tokenizer, distilbert_model, batch_size=32):
    # 1. Переводим модель в режим оценки
    distilbert_model.eval()
    all_embeddings = []

    # 2. Получаем эмбеддинги с помощью токенайзера и DistilBERT (как в вашем обучающем скрипте)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Токенизируем тем самым "векторизатором" (который на самом деле токенайзер)
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Прогоняем через модель DistilBERT
            outputs = distilbert_model(**inputs)
            
            # Берем CLS-токен ([0]-й элемент скрытых состояний)
            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
    # Объединяем в общую матрицу эмбеддингов
    X = np.vstack(all_embeddings)

    # 3. Передаем готовые эмбеддинги в логистическую регрессию
    preds = baseline_logreg_model.predict(X)
    return preds
