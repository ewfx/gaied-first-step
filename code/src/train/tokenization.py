from datasets import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Any
from config import logger

def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """
    Tokenize a Dataset using the provided tokenizer.
    """
    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        processed_texts = [txt if txt.strip() else "No text available." for txt in batch["text"]]
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