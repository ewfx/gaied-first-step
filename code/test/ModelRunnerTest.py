import unittest
from unittest.mock import patch, MagicMock
import torch
from ModelRunner import classify_email


class TestModelRunner(unittest.TestCase):

    @patch('ModelRunner.AutoTokenizer.from_pretrained')
    @patch('ModelRunner.AutoModelForSequenceClassification.from_pretrained')
    def test_classify_email(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.return_value = MagicMock(
            return_value={"input_ids": torch.tensor([[1]*128]),
                          "attention_mask": torch.tensor([[1]*128])}
        )

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = MagicMock(
            logits=torch.tensor([[1.0, 0.0]])
        )
        mock_model.return_value = mock_model_instance

        # Test
        label, confidence = classify_email("Test email content")
        self.assertIn(label, ["update", "request"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Verify tokenizer called with correct params
        mock_tokenizer.return_value.assert_called_once_with(
            "Test email content",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

    @patch('ModelRunner.MongoClient')
    @patch('ModelRunner.classify_email')
    def test_script_execution(self, mock_classify, mock_mongo):
        # Setup mocks
        mock_classify.return_value = ("update", 0.95)

        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": 1, "subject": "Test", "body": "Content"}
        ]
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo.return_value.__getitem__.return_value = mock_db

        # Import and run the script
        import ModelRunner

        # Verify calls
        mock_collection.update_one.assert_called_once()
        mock_classify.assert_called_once()


if __name__ == '__main__':
    unittest.main()
