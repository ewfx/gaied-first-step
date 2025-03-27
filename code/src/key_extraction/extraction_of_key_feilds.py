import re
from typing import Dict, Any, List
from pymongo import MongoClient

def extract_key_details(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts key details (CustomerName, SSN/TIN, Loan Amount, Request Type, Sub-Request Type)
    from the email subject, body, and attachments.
    """
    subject = email_data.get("subject", "").strip()
    body = email_data.get("body", "").strip()
    attachments = email_data.get("attachments", [])

    text = f"{subject} {body}"

    # Extract details from attachments.
    for attachment in attachments:
        if "content" in attachment:
            text += f"\n{attachment['content']}"
        if "body" in attachment:
            text += f"\n{attachment['body']}"

    # Regular expressions for extraction.
    customer_name_pattern = r"Name:\s*([A-Za-z\s.]+)"
    ssn_tin_pattern = r"(SSN|TIN):\s*([\d-]+)"
    loan_amount_pattern = r"Loan Amount:\s*([\d,]+)"
    request_type_pattern = r"Request Type:\s*([A-Za-z\s]+)"
    sub_request_type_pattern = r"Sub-Request Type:\s*([A-Za-z\s]+)"  # Added pattern for Sub-Request Type

    # Extract details using regular expressions.
    customer_name_match = re.search(customer_name_pattern, text)
    ssn_tin_match = re.search(ssn_tin_pattern, text)
    loan_amount_match = re.search(loan_amount_pattern, text)
    request_type_match = re.search(request_type_pattern, text)
    sub_request_type_match = re.search(sub_request_type_pattern, text)  # Extract Sub-Request Type

    # Store extracted details.
    extracted_details = {
        "CustomerName": customer_name_match.group(1).strip() if customer_name_match else None,
        "SSN/TIN": ssn_tin_match.group(2).strip() if ssn_tin_match else None,
        "LoanAmount": loan_amount_match.group(1).strip().replace(",", "") if loan_amount_match else None,
        "RequestType": request_type_match.group(1).strip() if request_type_match else None,
        "SubRequestType": sub_request_type_match.group(1).strip() if sub_request_type_match else None,  # Sub-Request Type
    }

    return extracted_details

def process_requests(request_emails: List[Dict[str, Any]], mongo_collection) -> None:
    """
    Processes a list of emails classified as "request," extracts key details,
    and updates the MongoDB documents with the extracted information.
    """
    for email in request_emails:
        details = extract_key_details(email)
        details["_id"] = email["_id"]
        details["from"] = email["from"]
        details["date"] = email["date"]

        # Update the MongoDB document with the extracted details.
        mongo_collection.update_one(
            {"_id": email["_id"]},
            {"$set": {"extractedKeyfields": details}}
        )

def main():
    """
    Main function to connect to MongoDB, fetch emails, and process them.
    """
    # MongoDB connection details.
    mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
    database_name = "emails_train_db30"  # Replace with your database name
    collection_name = "emails_train30"  # Replace with your collection name

    try:
        client = MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]

        # Fetch emails classified as "request" (assuming you have a classification field).
        request_emails = list(collection.find({"classification": "request"}))

        # Process the emails and update MongoDB.
        process_requests(request_emails, collection)

        print("Email processing and MongoDB updates completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()