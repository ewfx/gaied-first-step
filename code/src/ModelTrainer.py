import os
import random
import logging
import re
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from pymongo import MongoClient
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score
from shutil import rmtree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EmailClassifierTraining")

# Reproducibility: Set seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Global definitions
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4  # Now supporting 4 classes


def extract_attachment_content(attachment):
    """Extract text content from different attachment types"""
    if attachment['type'] == 'msg':
        return f"{attachment.get('subject', '')} {attachment.get('body', '')}"
    else:
        return attachment.get('content', '')


def load_emails_from_mongo(
    uri: str = "mongodb://localhost:27017",
    db_name: str = "email_routing",
    collection_name: str = "email_datasets",
) -> List[Dict[str, Any]]:
    """Connect to MongoDB and retrieve email documents."""
    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        emails = list(collection.find({}))
        logger.info(f"Retrieved {len(emails)} emails from MongoDB.")
        return emails
    except Exception as e:
        logger.error(f"Error retrieving emails: {e}")
        raise


def process_emails(emails: List[Dict[str, Any]]) -> Dict[str, List]:
    """Enhanced email processing with more detailed labels"""
    label_mapping = {
        "information_update": 0,
        "loan_processing": 1,
        "account_management": 2,
        "general_inquiry": 3
    }

    texts = []
    labels = []
    counts = {label: 0 for label in label_mapping.keys()}

    for email in emails:
        subject = email.get("subject", "").lower()
        body = email.get("body", "").lower()

        # Enhanced labeling logic
        if "update" in subject or "update" in body:
            label_str = "information_update"
        elif "loan" in subject or "loan" in body:
            label_str = "loan_processing"
        elif "account" in subject or "account" in body:
            label_str = "account_management"
        else:
            label_str = "general_inquiry"

        # Include attachments in training text
        email_text = f"{subject} {body}"
        for attachment in email.get("attachments", []):
            email_text += " " + extract_attachment_content(attachment)

        texts.append(email_text)
        labels.append(label_mapping[label_str])
        counts[label_str] += 1

        logger.info("Email categorized as '%s'", label_str)

    logger.info("Categorization summary: %s", counts)
    return {"text": texts, "label": labels}


``


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 512) -> Dataset:
    """Tokenize a Dataset using the provided tokenizer."""
    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        processed_texts = [
            txt if txt.strip() else "No text available." for txt in batch["text"]]
        return tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns("text")
    logger.info("Dataset tokenization complete.")
    return tokenized_dataset


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def model_init() -> AutoModelForSequenceClassification:
    """Function to instantiate a new model."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def run_hyperparameter_search(trainer: Trainer, n_trials: int = 5) -> Optional[str]:
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=n_trials,
    )
    logger.info("Best trial: %s", best_trial)
    return best_trial.checkpoint if best_trial and hasattr(best_trial, "checkpoint") else None


def main() -> None:
    # Clear Transformers cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        try:
            rmtree(cache_dir)
            logger.info("Transformers cache cleared.")
        except Exception as e:
            logger.warning(f"Failed to clear Transformers cache: {e}")
    else:
        logger.info("Transformers cache directory not found.")

    # Load emails from MongoDB
    try:
        emails = load_emails_from_mongo()
    except Exception:
        logger.error("Failed to load emails from MongoDB. Exiting.")
        return

    data_dict = process_emails(emails)
    dataset = Dataset.from_dict(data_dict)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    raw_eval_dataset = Dataset.from_dict(eval_dataset.to_dict())
    logger.info("Dataset partitioned. Train columns: %s",
                train_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=512)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=512)
    logger.info("After tokenization, train dataset columns: %s",
                train_dataset.column_names)

    train_dataset.set_format(type="torch", columns=[
                             "input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=[
                            "input_ids", "attention_mask", "label"])

    # MODEL INITIALIZATION
    model_dir = "email_classifier_llm_latest"
    choice = input(
        f"Load an existing model from '{model_dir}'? (y/n): ").strip().lower()
    if choice == 'y' and os.path.exists(model_dir):
        logger.info("Loading existing model from: %s", model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=NUM_LABELS)
    else:
        if choice == 'y':
            logger.warning(
                "No model found at '%s'. Creating a new model.", model_dir)
        else:
            logger.info("Creating a new model.")
        model = model_init()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="output_llm",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Optional hyperparameter search
    best_checkpoint = run_hyperparameter_search(trainer, n_trials=3)
    if best_checkpoint:
        logger.info("Resuming training from best checkpoint: %s",
                    best_checkpoint)
        trainer.train(resume_from_checkpoint=best_checkpoint)
    else:
        logger.info("Starting training from scratch.")
        trainer.train()

    logger.info("Model training complete.")
    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)
    print("Evaluation results:", eval_results)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Model and tokenizer saved to '%s'.", model_dir)


if __name__ == "__main__":
    main()
