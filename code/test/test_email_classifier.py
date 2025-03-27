import unittest
from unittest.mock import MagicMock
from runner.email_classifier import classify_email

class TestEmailClassifier(unittest.TestCase):
    def test_classify_email(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_model.return_value.logits = MagicMock()
        mock_model.return_value.logits.softmax.return_value = [0.1, 0.7, 0.2]

        email_text = "Test email"
        predicted_label, confidence_score, important_tokens = classify_email(
            email_text, mock_model, mock_tokenizer
        )

        self.assertIsInstance(predicted_label, str)
        self.assertIsInstance(confidence_score, float)
        self.assertIsInstance(important_tokens, list)

if __name__ == "__main__":
    unittest.main()