from sklearn.metrics import classification_report, f1_score, accuracy_score
from predict_finetuned import predict_fine_tuned, predict_baseline, baseline_vectorizer, baseline_model
from test_predict import model, tokenizer, model_distilb
from make_con_matrix import test_labels, y_pred_ft, test_texts

# Fine-tuned
print("=== Fine-tuned Model ===")
print(classification_report(test_labels, y_pred_ft))
f1_ft = f1_score(test_labels, y_pred_ft, average='macro')
acc_ft = accuracy_score(test_labels, y_pred_ft)

# Baseline (если есть)
if baseline_model:
      preds_baseline_all = predict_baseline(test_texts, baseline_model, baseline_vectorizer, model_distilb)
      y_pred_base = preds_baseline_all
      if len(y_pred_base) > 0 and isinstance(y_pred_base[0], str):
            y_pred_base = [1 if p == 'positive' else 0 for p in y_pred_base]
      print("\n=== Baseline Model ===")
      print(classification_report(test_labels, y_pred_base))
      f1_base = f1_score(test_labels, y_pred_base, average='macro')
      acc_base = accuracy_score(test_labels, y_pred_base)

      print(f'\n=== Сравнение ===')
      print(f'Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}')
      print(f'Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}')
      print(f'Улучшение F1: {(f1_ft - f1_base) / f1_base * 100:.2f}%')

with open('comparison_results.txt', 'w') as f:
    f.write('=== Сравнение моделей ===\n\n')
    f.write(f'Fine-tuned Model:\n')
    f.write(f'  F1 (macro): {f1_ft:.4f}\n')
    f.write(f'  Accuracy: {acc_ft:.4f}\n')

    if baseline_model:
        f.write(f'\nBaseline Model:\n')
        f.write(f'  F1 (macro): {f1_base:.4f}\n')
        f.write(f'  Accuracy: {acc_base:.4f}\n')
        f.write(f'\nУлучшение: {(f1_ft - f1_base) / f1_base * 100:.2f}%\n')