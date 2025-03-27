import torch
import torch.nn.functional as F
from pymongo import MongoClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import datetime
import logging
from bson.objectid import ObjectId

# Load model and tokenizer
model_path = "email_classifier_llm_latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# After loading the model, add this check:
print("Model architecture:")
print(model)

# Check classifier weights
print("\nClassifier layer weights:")
print(model.classifier.state_dict())

# Verify number of classes matches your setup
print(f"\nNumber of output classes: {model.num_labels}")

logger = logging.getLogger("EmailClassifierTraining")


def extract_attachment_content(attachment):
    """Extract text content from different attachment types"""
    if attachment['type'] == 'msg':
        return f"{attachment.get('subject', '')} {attachment.get('body', '')}"
    else:
        return attachment.get('content', '')


def preprocess_email(doc):
    """Combine email subject, body, and attachment content"""
    subject = doc.get("subject", "")
    body = doc.get("body", "")
    email_text = f"{subject} {body}"

    # Process attachments
    attachments_content = []
    for attachment in doc.get("attachments", []):
        attachments_content.append(extract_attachment_content(attachment))

    if attachments_content:
        email_text += " ATTACHMENTS: " + " ".join(attachments_content)

    return email_text


def classify_email(email_text, max_length=512):
    """Enhanced classification with proper confidence scoring"""
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)  # Note the dim=-1

    # Get the predicted class and its confidence
    confidence, predicted_class_id = torch.max(probabilities, dim=-1)
    confidence_score = confidence.item()
    predicted_class_id = predicted_class_id.item()

    # Enhanced label mapping
    label_mapping = {
        0: {"type": "update", "subtype": "information_update"},
        1: {"type": "request", "subtype": "loan_processing"},
        2: {"type": "request", "subtype": "account_management"},
        3: {"type": "request", "subtype": "general_inquiry"}
    }

    return label_mapping.get(predicted_class_id, {"type": "unknown", "subtype": "unknown"}), confidence_score


def extract_fields(text):
    """Enhanced field extraction with more patterns and error handling"""
    fields = {}
    patterns = {
        'account_number': r'(?:account|acct)[\s#:]*(\d{4,})',
        'loan_amount': r'(?:loan|amount)[\s$:]*([\d,]+)',
        'customer_name': r'(?:name|contact)[\s:]*([^\n\r]+)',
        'phone': r'(?:phone|mobile|tel)[\s:]*([\+\(\)\d\s-]{7,})',
        'email': r'([\w\.-]+@[\w\.-]+)',
        'ssn': r'(\d{3}-\d{2}-\d{4})',
        'address': r'(?:address|addr)[\s:]*([^\n\r]+)'
    }

    for field, pattern in patterns.items():
        try:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            field_values = []
            for match in matches:
                if match.groups():  # Check if any groups were captured
                    value = match.group(1)  # Get first capturing group
                    if value:  # Only add non-empty values
                        field_values.append(value.strip())
            if field_values:  # Only add to fields if we found values
                fields[field] = field_values
        except Exception as e:
            logger.warning(f"Error extracting {field}: {str(e)}")
            continue

    return fields


def analyze_intent(text):
    """Determine the intent of the email"""
    text_lower = text.lower()
    intent = "unknown"

    if any(word in text_lower for word in ["urgent", "immediately", "asap"]):
        intent = "urgent"
    elif "loan" in text_lower:
        intent = "loan_related"
    elif "account" in text_lower:
        intent = "account_related"
    elif "update" in text_lower:
        intent = "information_update"
    elif any(word in text_lower for word in ["question", "inquiry", "help"]):
        intent = "general_inquiry"

    return intent


def analyze_email(doc):
    """Complete email analysis pipeline"""
    email_text = preprocess_email(doc)
    classification, confidence = classify_email(email_text)
    extracted_fields = extract_fields(email_text)
    intent = analyze_intent(email_text)

    return {
        "classification": classification,
        "confidence": round(confidence, 4),
        "extracted_fields": extracted_fields,
        "intent": intent,
        "processed_text": email_text[:500] + "..." if len(email_text) > 500 else email_text,
        "analysis_date": datetime.datetime.now().isoformat()
    }


def main():
    """Main processing function"""
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["emails_train_db30"]
    collection = db["emails_train30"]

    # Process each email document
    for doc in collection.find({"is_duplicate": False}):
        try:
            analysis_result = analyze_email(doc)

            # Update the document with full analysis
            collection.update_one(
                {"_id": ObjectId(doc["_id"])},
                {"$set": {
                    "analysis": analysis_result
                }}
            )
            print(
                f"Processed document {doc['_id']} with confidence {analysis_result['confidence']:.2f}")
        except Exception as e:
            print(f"Error processing document {doc['_id']}: {str(e)}")

    print("Document classification and analysis completed.")


if __name__ == "__main__":
    main()
