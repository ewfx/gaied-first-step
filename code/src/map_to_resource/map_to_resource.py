from pymongo import MongoClient

def create_users() -> List[Dict[str, Any]]:
    """
    Creates 10 users with skill sets for different request types and sub-types.
    """
    users = [
        {"UserID": 1, "Name": "Alice", "SkillSet": {"Billing Issue": ["Invoice Discrepancy", "Payment Delay", "Refund Request"]}},
        {"UserID": 2, "Name": "Bob", "SkillSet": {"Technical Support": ["Bug Report", "Feature Assistance", "Hardware Setup"]}},
        {"UserID": 3, "Name": "Charlie", "SkillSet": {"Account Update": ["Profile Change", "Password Reset", "Account Deactivation"]}},
        {"UserID": 4, "Name": "David", "SkillSet": {"General Inquiry": ["Product Info", "Service Feedback", "Partnership Request"]}},
        {"UserID": 5, "Name": "Ella", "SkillSet": {"Billing Issue": ["Invoice Discrepancy", "Payment Delay", "Refund Request"]}},
        {"UserID": 6, "Name": "Frank", "SkillSet": {"Technical Support": ["Bug Report", "Feature Assistance", "Hardware Setup"]}},
        {"UserID": 7, "Name": "Grace", "SkillSet": {"Account Update": ["Profile Change", "Password Reset", "Account Deactivation"]}},
        {"UserID": 8, "Name": "Hank", "SkillSet": {"General Inquiry": ["Product Info", "Service Feedback", "Partnership Request"]}},
        {"UserID": 9, "Name": "Ivy", "SkillSet": {"Billing Issue": ["Invoice Discrepancy", "Payment Delay", "Refund Request"]}},
        {"UserID": 10, "Name": "Jack", "SkillSet": {"Technical Support": ["Bug Report", "Feature Assistance", "Hardware Setup"]}},
    ]
    return users

def fetch_requests(mongo_collection) -> List[Dict[str, Any]]:
    """
    Fetches all request documents with extracted key fields from MongoDB.
    """
    return list(mongo_collection.find({"extractedKeyfields": {"$exists": True}}))

def assign_requests_to_users(users: List[Dict[str, Any]], requests: List[Dict[str, Any]], mongo_collection) -> None:
    """
    Assigns requests to users based on their skill set and updates MongoDB.
    """
    for request in requests:
        assigned_user = None
        request_type = request.get("extractedKeyfields", {}).get("RequestType")
        sub_request_type = request.get("extractedKeyfields", {}).get("SubRequestType")

        for user in users:
            if request_type in user["SkillSet"]:
                if sub_request_type in user["SkillSet"][request_type]:
                    assigned_user = user
                    break

        # Update the MongoDB document with the assigned user's details.
        if assigned_user:
            mongo_collection.update_one(
                {"_id": request["_id"]},
                {"$set": {"AssignedUser": {"UserID": assigned_user["UserID"], "Name": assigned_user["Name"]}}}
            )
            print(f"Request ID {request['_id']} assigned to User {assigned_user['Name']} (UserID: {assigned_user['UserID']}).")
        else:
            print(f"No user found for Request ID {request['_id']}.")

def main():
    """
    Main function to connect to MongoDB, create users, fetch requests, and assign them to users.
    """
    # MongoDB connection details.
    mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
    database_name = "emails_train_db30"  # Replace with your database name
    collection_name = "emails_train30"  # Replace with your collection name

    try:
        client = MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]

        # Create users.
        users = create_users()

        # Fetch requests from MongoDB.
        requests = fetch_requests(collection)

        # Assign requests to users.
        assign_requests_to_users(users, requests, collection)

        print("Request assignment completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()