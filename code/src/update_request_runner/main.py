from model_loader import load_model
from email_classifier import classify_email
from mongodb_handler import connect_to_mongodb, update_email_document

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()

    # Connect to MongoDB
    collection = connect_to_mongodb()

    # Iterate over each email document.
    for doc in collection.find():
        subject = doc.get("subject", "")
        body = doc.get("body", "")
        # Combine subject and body to create the text input.
        email_text = f"{subject} {body}"

        # Classify the email.
        predicted_label, confidence_score, important_tokens = classify_email(
            email_text, model, tokenizer
        )

        # Update the document with the predicted label and confidence score.
        update_email_document(
            collection, doc["_id"], predicted_label, confidence_score, important_tokens
        )

        print(
            f"Updated document {doc['_id']} with predicted label: {predicted_label}, "
            f"confidence score: {confidence_score:.2f}, Important Tokens: {important_tokens}"
        )

    print("Document classification and update process completed.")


if __name__ == "__main__":
    main()