import hashlib
from typing import List, Dict, Any
from pymongo import MongoClient

def generate_hash(details: Dict[str, Any]) -> str:
    """
    Generates a hash from the extracted key fields.
    """
    fields_string = f"{details.get('CustomerName', '')}-{details.get('SSN/TIN', '')}-" \
                    f"{details.get('LoanAmount', '')}-{details.get('RequestType', '')}-" \
                    f"{details.get('SubRequestType', '')}"
    return hashlib.sha256(fields_string.encode('utf-8')).hexdigest()

def fetch_extracted_data(mongo_collection) -> List[Dict[str, Any]]:
    """
    Fetches all documents with extracted key fields from the MongoDB collection.
    """
    return list(mongo_collection.find({"extractedKeyfields": {"$exists": True}}))

def check_duplicates(email_data: List[Dict[str, Any]], mongo_collection) -> None:
    """
    Checks for duplicate hashes and updates the MongoDB documents.
    """
    seen_hashes = {}

    for email in email_data:
        details = email["extractedKeyfields"]
        email_hash = generate_hash(details)
        ssn_tin = details.get("SSN/TIN")
        sender_email = email.get("from")

        # Check for duplicates based on SSN/TIN or sender email.
        for key in seen_hashes:
            if key["SSN/TIN"] == ssn_tin or key["Email"] == sender_email:
                if seen_hashes[key] == email_hash:
                    # Mark as duplicate in MongoDB with confidence code.
                    mongo_collection.update_one(
                        {"_id": email["_id"]},
                        {"$set": {"isDuplicate": True, "confidenceCode": "High"}}
                    )
                    print(f"Duplicate found: Email ID {email['_id']} is a duplicate of {key['_id']}.")
                    break

        # Add the current email's hash to seen_hashes.
        seen_hashes[email_hash] = {"_id": email["_id"], "SSN/TIN": ssn_tin, "Email": sender_email}

def main():
    """
    Main function to connect to MongoDB and check for duplicates.
    """
    mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
    database_name = "emails_train_db30"  # Replace with your database name
    collection_name = "emails_train30"  # Replace with your collection name

    try:
        client = MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]

        # Fetch extracted key fields from MongoDB.
        email_data = fetch_extracted_data(collection)

        # Check for duplicates and update MongoDB.
        check_duplicates(email_data, collection)

        print("Duplicate check and MongoDB updates completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()