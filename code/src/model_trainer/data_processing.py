from typing import List, Dict, Any
from config import logger

def process_emails(emails: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Process emails: combine subject and body and assign label:
       - "update" if 'is_update_case' is True
       - "request" otherwise.
    """
    label_mapping = {"update": 0, "request": 1}
    texts: List[str] = []
    labels: List[int] = []
    update_count = 0
    request_count = 0

    for email in emails:
        if email.get("is_update_case", False):
            label_str = "update"
            update_count += 1
        else:
            label_str = "request"
            request_count += 1

        subject = email.get("subject", "").strip()
        body = email.get("body", "").strip()
        combined_text = f"{subject} {body}".strip()
        texts.append(combined_text)
        labels.append(label_mapping[label_str])
        
        logger.info("Email subject: '%s' categorized as '%s'.", subject, label_str)
    
    logger.info("Categorization summary: update=%d, request=%d", 
                update_count, request_count)
    return {"text": texts, "label": labels}