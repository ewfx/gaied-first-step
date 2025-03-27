from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score
from config import logger

def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Compute evaluation metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def get_label_name(label: int) -> str:
    """
    Helper function to get label names
    """
    return "update" if label == 0 else "request"

def heuristic_reason(text: str, actual: int, predicted: int) -> str:
    """
    Heuristic reasoning for misclassifications
    """
    text_lower = text.lower()
    reasons = []
    if actual == 0 and "update" not in text_lower:
        reasons.append("Missing 'update' keyword.")
    elif actual == 1 and "update" in text_lower:
        reasons.append("Contains 'update' keyword unexpectedly.")
    if len(text.strip()) < 20:
        reasons.append("Text is very short and ambiguous.")
    return ", ".join(reasons) if reasons else "No obvious reason."

def evaluate_model(trainer, eval_dataset, raw_eval_dataset):
    """
    Evaluates the model and reports misclassifications.
    """
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)
    print("Evaluation results:", eval_results)

    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    predicted_labels = np.argmax(logits, axis=-1)
    true_labels = raw_eval_dataset["label"]
    texts = raw_eval_dataset["text"]

    misclassified = []
    for i in range(len(true_labels)):
        if predicted_labels[i] != true_labels[i]:
            reason = heuristic_reason(texts[i], true_labels[i], predicted_labels[i])
            misclassified.append({
                "text_snippet": texts[i][:200],
                "actual": get_label_name(true_labels[i]),
                "predicted": get_label_name(predicted_labels[i]),
                "reason": reason,
            })

    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info("Post-training Accuracy on evaluation set: %.2f%%", accuracy * 100)
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

    if misclassified:
        logger.info("Misclassified examples:")
        for idx, mis in enumerate(misclassified, start=1):
            print(f"--- Misclassified Example {idx} ---")
            print(f"Text Snippet: {mis['text_snippet']}")
            print(f"Actual Label: {mis['actual']}, Predicted Label: {mis['predicted']}")
            print(f"Heuristic Reason: {mis['reason']}\n")
    else:
        logger.info("No misclassifications found on evaluation data.")