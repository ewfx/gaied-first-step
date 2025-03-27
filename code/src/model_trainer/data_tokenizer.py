from config import logger, MODEL_NAME
from transformers import AutoTokenizer
from tokenization import tokenize_dataset

def tokenize_datasets(train_dataset, eval_dataset):
    """Tokenizes the training and evaluation datasets."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=128)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, max_length=128)
    logger.info("After tokenization, train dataset columns: %s", train_dataset.column_names)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_dataset, eval_dataset, tokenizer