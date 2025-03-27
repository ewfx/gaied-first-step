from transformers import Trainer, TrainingArguments
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from typing import Optional
from config import logger, RANDOM_SEED
from model_handling import model_init  # Import model_init
from transformers import EarlyStoppingCallback # Import EarlyStoppingCallback

def train_model(
    model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
):
    """
    Trains the model using the Trainer.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="output_llm",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=50,  # Increase batch size (if memory allows)
        per_device_eval_batch_size=50,
        num_train_epochs=5,             # Increase epochs (start higher, use early stopping)
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
        # Add Early Stopping
        early_stopping_patience=3,   # How many epochs to wait
        early_stopping_threshold=0.001 # min_delta
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)], # Add callback to trainer
    )

    trainer.train()  # Train without hyperparameter search
    return trainer

def run_hyperparameter_search(trainer: Trainer, n_trials: int = 5) -> Optional[str]:
    """
    Performs hyperparameter search using the Trainer.
    """
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=n_trials,
    )
    logger.info("Best trial: %s", best_trial)
    return best_trial.checkpoint if best_trial and hasattr(best_trial, "checkpoint") else None