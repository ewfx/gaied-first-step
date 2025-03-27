from config import logger
from data_access import load_emails_from_mongo
from data_processing import process_emails
from datasets import Dataset
import time  # Import the time module

def load_and_prepare_data():
    """Loads emails from MongoDB and prepares the datasets."""
    start_time = time.time()  # Start timing
    try:
        emails = load_emails_from_mongo()
    except Exception:
        logger.error("Failed to load emails from MongoDB. Exiting.")
        return None, None, None
    mongo_time = time.time() - start_time  # Calculate time for MongoDB

    start_time = time.time()
    data_dict = process_emails(emails)
    dataset = Dataset.from_dict(data_dict)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    raw_eval_dataset = Dataset.from_dict(eval_dataset.to_dict())
    process_time = time.time() - start_time  # Calculate processing time

    logger.info("Dataset partitioned. Train columns: %s", train_dataset.column_names)
    logger.info(f"Time taken for MongoDB: {mongo_time:.2f} seconds")  # Log MongoDB time
    logger.info(f"Time taken for processing: {process_time:.2f} seconds")  # Log processing time

    return train_dataset, eval_dataset, raw_eval_dataset