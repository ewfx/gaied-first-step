import unittest
from unittest.mock import patch, MagicMock
from runner.main import main

# filepath: e:\hackathon_25\src\runner\test_main.py

class TestMain(unittest.TestCase):
    @patch("runner.main.load_model")
    @patch("runner.main.connect_to_mongodb")
    @patch("runner.main.classify_email")
    @patch("runner.main.update_email_document")
    def test_main(self, mock_update_email_document, mock_classify_email, mock_connect_to_mongodb, mock_load_model):
        # Mock load_model to return a fake model and tokenizer
        mock_load_model.return_value = ("mock_model", "mock_tokenizer")

        # Mock connect_to_mongodb to return a fake collection
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": 1, "subject": "Test Subject", "body": "Test Body"},
            {"_id": 2, "subject": "Another Subject", "body": "Another Body"},
        ]
        mock_connect_to_mongodb.return_value = mock_collection

        # Mock classify_email to return fake classification results
        mock_classify_email.side_effect = [
            ("spam", 0.95, ["test", "subject"]),
            ("ham", 0.89, ["another", "body"]),
        ]

        # Call the main function
        main()

        # Assert load_model was called once
        mock_load_model.assert_called_once()

        # Assert connect_to_mongodb was called once
        mock_connect_to_mongodb.assert_called_once()

        # Assert classify_email was called twice (once for each document)
        mock_classify_email.assert_any_call("Test Subject Test Body", "mock_model", "mock_tokenizer")
        mock_classify_email.assert_any_call("Another Subject Another Body", "mock_model", "mock_tokenizer")
        self.assertEqual(mock_classify_email.call_count, 2)

        # Assert update_email_document was called twice with correct arguments
        mock_update_email_document.assert_any_call(mock_collection, 1, "spam", 0.95, ["test", "subject"])
        mock_update_email_document.assert_any_call(mock_collection, 2, "ham", 0.89, ["another", "body"])
        self.assertEqual(mock_update_email_document.call_count, 2)

if __name__ == "__main__":
    unittest.main()