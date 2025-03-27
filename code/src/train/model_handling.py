from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_LABELS, logger
import os

def model_init() -> AutoModelForSequenceClassification:
    """
    Function to instantiate a new model.
    Used during hyperparameter search to get fresh model instances.
    """
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

def load_model(model_dir: str) -> AutoModelForSequenceClassification:
    """Loads a pre-trained model from the specified directory."""
    logger.info("Loading existing model from: %s", model_dir)
    return AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=NUM_LABELS)

def save_model(model: AutoModelForSequenceClassification, tokenizer, model_dir: str):
     """Saves the trained model and tokenizer to the specified directory."""
     model.save_pretrained(model_dir)
     tokenizer.save_pretrained(model_dir)
     logger.info("Model and tokenizer saved to '%s'.", model_dir)