import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

# Define taxonomy-related keywords (you can expand this)
taxonomy_keywords = {
    "update": ["progress", "report", "status", "changes", "modify", "revised"],
    "request": ["need", "require", "assistance", "help", "information", "please provide"]
}

def classify_email(
    email_text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> tuple[str, float, List[Dict[str, any]]]:
    """
    Tokenizes the input text, performs inference using the model,
    and returns the predicted label, confidence score,
    and a simplified analysis of token contributions,
    incorporating taxonomy-based insights.
    """
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,  # Get offsets mapping
    )

    # Extract model inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Pass only model inputs

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)[0]  # Apply softmax to obtain probabilities
    predicted_class_id = logits.argmax(dim=1).item()
    confidence_score = probabilities[predicted_class_id].item()

    # Map model output to a label.
    label_mapping = {0: "update", 1: "request"}
    predicted_label = label_mapping[predicted_class_id]

    # Simplified token analysis (very rough)
    offset_mapping = inputs["offset_mapping"][0] # extract offset_mapping here
    input_ids = inputs["input_ids"][0]


    important_tokens = []

    # Get the probabilities for the predicted class
    predicted_class_probs = probabilities  # Shape: [num_labels]

    # Sort by probability (descending)
    sorted_token_indices = probabilities.argsort(descending=True)

    for i in sorted_token_indices[:5]:  # Top 5 tokens
        token = tokenizer.decode([input_ids[i]])
        start, end = offset_mapping[i]
        word = email_text[start:end]
        probability = float(probabilities[int(predicted_class_probs[i].item())])  # Convert to float
        
        is_taxonomy_word = False
        if predicted_label in taxonomy_keywords:
            if word.lower() in taxonomy_keywords[predicted_label]:
                is_taxonomy_word = True
        
        important_tokens.append(
            {"token": token, "word": word, "probability": probability, "is_taxonomy_word": is_taxonomy_word}
        )

    return predicted_label, confidence_score, important_tokens