import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path="email_classifier_llm_latest"):
    """
    Loads the fine-tuned model and tokenizer from the specified directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer