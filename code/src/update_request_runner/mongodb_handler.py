from pymongo import MongoClient
from typing import Dict, List

def connect_to_mongodb(
    uri: str = "mongodb://localhost:27017",
    db_name: str = "emails_train_db30",
    collection_name: str = "emails_train30",
) -> any:
    """
    Connects to MongoDB and returns the collection object.
    """
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

def update_email_document(
    collection: any, doc_id: any, predicted_label: str, confidence_score: float, important_tokens: List[Dict[str, any]]
) -> None:
    """
    Updates the email document in MongoDB with the predicted label,
    confidence score, and important tokens.
    """
    collection.update_one(
        {"_id": doc_id},
        {"$set": {"predicted_label": predicted_label, "confidence_score": confidence_score, "important_tokens": important_tokens}},
    )