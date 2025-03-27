import unittest
from unittest.mock import MagicMock, patch
from complete_extraction_data.final_extraion_runner import (
    preprocess_email,
    classify_email,
    extract_fields,
    analyze_intent,
    analyze_email,
)

class TestFinalExtractionRunner(unittest.TestCase):
    @patch("complete_extraction_data.final_extraion_runner.extract_attachment_content")
    def test_preprocess_email(self, mock_extract_attachment_content):
        mock_extract_attachment_content.return_value = "Attachment Content"
        doc = {
            "subject": "Test Subject",
            "body": "Test Body",
            "attachments": [{"type": "msg", "content": "Attachment Content"}],
        }
        result = preprocess_email(doc)
        expected_result = "Test Subject Test Body ATTACHMENTS: Attachment Content"
        self.assertEqual(result, expected_result)

    @patch("complete_extraction_data.final_extraion_runner.tokenizer")
    @patch("complete_extraction_data.final_extraion_runner.model")
    def test_classify_email(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_model.return_value.logits = MagicMock()
        mock_model.return_value.logits.softmax.return_value = [0.1, 0.7, 0.2]

        email_text = "Test email"
        classification, confidence_score = classify_email(email_text)

        self.assertIsInstance(classification, dict)
        self.assertIn("type", classification)
        self.assertIn("subtype", classification)
        self.assertIsInstance(confidence_score, float)

    def test_extract_fields(self):
        text = "Account: 123456 Loan Amount: $50,000 Name: John Doe Phone: +1-234-567-8901"
        result = extract_fields(text)
        expected_result = {
            "account_number": ["123456"],
            "loan_amount": ["50000"],
            "customer_name": ["John Doe"],
            "phone": ["+1-234-567-8901"],
        }
        self.assertEqual(result, expected_result)

    def test_analyze_intent(self):
        text = "This is an urgent loan request."
        result = analyze_intent(text)
        self.assertEqual(result, "urgent")

        text = "I need help with my account."
        result = analyze_intent(text)
        self.assertEqual(result, "account_related")

        text = "Can you provide information about loans?"
        result = analyze_intent(text)
        self.assertEqual(result, "loan_related")

    @patch("complete_extraction_data.final_extraion_runner.preprocess_email")
    @patch("complete_extraction_data.final_extraion_runner.classify_email")
    @patch("complete_extraction_data.final_extraion_runner.extract_fields")
    @patch("complete_extraction_data.final_extraion_runner.analyze_intent")
    def test_analyze_email(
        self, mock_analyze_intent, mock_extract_fields, mock_classify_email, mock_preprocess_email
    ):
        mock_preprocess_email.return_value = "Processed email text"
        mock_classify_email.return_value = ({"type": "request", "subtype": "loan_processing"}, 0.95)
        mock_extract_fields.return_value = {"account_number": ["123456"]}
        mock_analyze_intent.return_value = "loan_related"

        doc = {"_id": "123", "subject": "Test Subject", "body": "Test Body"}
        result = analyze_email(doc)

        self.assertIn("classification", result)
        self.assertIn("confidence", result)
        self.assertIn("extracted_fields", result)
        self.assertIn("intent", result)
        self.assertIn("processed_text", result)
        self.assertIn("analysis_date", result)

if __name__ == "__main__":
    unittest.main()