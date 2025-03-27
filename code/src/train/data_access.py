from pymongo import MongoClient
from typing import List, Dict, Any
from config import logger  # Import logger

def load_emails_from_mongo(
    uri: str = "mongodb://localhost:27017",
    db_name: str = "email_train_db3",
    collection_name: str = "emails_train3",
) -> List[Dict[str, Any]]:
    """
    Connect to MongoDB and retrieve email documents.
    """
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