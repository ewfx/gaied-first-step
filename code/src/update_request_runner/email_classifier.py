import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any

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
) -> tuple[str, float, List[Dict[str, Any]]]:
    """
    Tokenizes the input text, performs inference using the model,
    and returns the predicted label, confidence score, and
    an analysis of token contributions based on attention weights.
    """
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    # Extract model inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True # Crucial: Get attention weights
        )  # Pass only model inputs

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)[0]
    predicted_class_id = logits.argmax(dim=1).item()
    confidence_score = probabilities[predicted_class_id].item()

    # Map model output to a label.
    label_mapping = {0: "update", 1: "request"}
    predicted_label = label_mapping[predicted_class_id]

    # Attention-based token analysis
    attention_weights = outputs.attentions  # This is a tuple of attention tensors
    
    # For simplicity, let's take the attention from the last layer.
    # You might want to experiment with different layers or combinations.
    last_layer_attention = attention_weights[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
    
    # Average attention across heads and batch
    averaged_attention = torch.mean(last_layer_attention, dim=(0, 1))  # Shape: [seq_len, seq_len]
    
    # Get attention for the [CLS] token (the first token)
    cls_attention = averaged_attention[0]  # Shape: [seq_len]
    
    # Get token-word mapping
    offset_mapping = inputs["offset_mapping"][0]
    input_ids = input_ids[0]  # Get the first (and only) sequence

    important_tokens: List[Dict[str, Any]] = []
    for i, attention_score in enumerate(cls_attention):
        start, end = offset_mapping[i]
        word = email_text[start:end]
        token = tokenizer.decode([input_ids[i]])
        
        is_taxonomy_word = False
        if predicted_label in taxonomy_keywords:
            if word.lower() in taxonomy_keywords[predicted_label]:
                is_taxonomy_word = True
        
        important_tokens.append({
            "token": token,
            "word": word,
            "attention_score": attention_score.item(),  # Convert to float
            "is_taxonomy_word": is_taxonomy_word,
        })
    
    # Sort tokens by attention score (descending)
    important_tokens.sort(key=lambda x: x["attention_score"], reverse=True)

    return predicted_label, confidence_score, important_tokens