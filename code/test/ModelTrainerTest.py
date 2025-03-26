import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from ModelTrainer import (
    load_emails_from_mongo,
    process_emails,
    tokenize_dataset,
    compute_metrics,
    heuristic_reason,
    main
)


class TestModelTrainer(unittest.TestCase):

    @patch('ModelTrainer.MongoClient')
    def test_load_emails_from_mongo(self, mock_mongo):
        # Setup mock
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"subject": "Test", "body": "Content", "is_update_case": True},
            {"subject": "Test2", "body": "Content2", "is_update_case": False}
        ]
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo.return_value.__getitem__.return_value = mock_db

        # Test
        emails = load_emails_from_mongo()
        self.assertEqual(len(emails), 2)
        mock_collection.find.assert_called_once_with({})

    def test_process_emails(self):
        test_emails = [
            {"subject": "Test", "body": "Content", "is_update_case": True},
            {"subject": "Test2", "body": "Content2", "is_update_case": False}
        ]
        result = process_emails(test_emails)
        self.assertEqual(len(result["text"]), 2)
        self.assertEqual(len(result["label"]), 2)
        self.assertEqual(result["label"][0], 0)  # update
        self.assertEqual(result["label"][1], 1)  # request

    @patch('ModelTrainer.AutoTokenizer.from_pretrained')
    def test_tokenize_dataset(self, mock_tokenizer):
        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock(
            side_effect=lambda texts, **kwargs: {
                "input_ids": [[1]*128]*len(texts),
                "attention_mask": [[1]*128]*len(texts)
            }
        )

        # Test dataset
        test_data = {
            "text": ["Test email content", "Another test email"],
            "label": [0, 1]
        }
        dataset = MagicMock()
        dataset.__getitem__.side_effect = lambda x: test_data[x]
        dataset.map.return_value = dataset

        tokenized = tokenize_dataset(dataset, mock_tokenizer.return_value)
        self.assertTrue(mock_tokenizer.called)
        dataset.map.assert_called_once()

    def test_compute_metrics(self):
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        metrics = compute_metrics((logits, labels))
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_heuristic_reason(self):
        # Test cases
        self.assertIn("Missing 'update' keyword",
                      heuristic_reason("request content", 0, 1))
        self.assertIn("Contains 'update' keyword",
                      heuristic_reason("update content", 1, 0))
        self.assertIn("Text is very short",
                      heuristic_reason("hi", 0, 1))

    @patch('ModelTrainer.load_emails_from_mongo')
    @patch('ModelTrainer.AutoModelForSequenceClassification.from_pretrained')
    @patch('ModelTrainer.AutoTokenizer.from_pretrained')
    @patch('ModelTrainer.Trainer')
    def test_main(self, mock_trainer, mock_tokenizer, mock_model, mock_load):
        # Setup mocks
        mock_load.return_value = [
            {"subject": "Test", "body": "Content", "is_update_case": True},
            {"subject": "Test2", "body": "Content2", "is_update_case": False}
        ]
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Simulate user input
        with patch('builtins.input', return_value='n'):
            main()

        # Verify calls
        mock_model.assert_called()
        mock_tokenizer.assert_called()
        mock_trainer_instance.train.assert_called()


if __name__ == '__main__':
    unittest.main()
