import unittest
from unittest.mock import patch, MagicMock
from runner.main import main

class TestMainRunner(unittest.TestCase):
    @patch("runner.main.load_model")
    @patch("runner.main.connect_to_mongodb")
    @patch("runner.main.classify_email")
    @patch("runner.main.update_email_document")
    def test_main(self, mock_update_email_document, mock_classify_email, mock_connect_to_mongodb, mock_load_model):
        mock_load_model.return_value = ("mock_model", "mock_tokenizer")
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"_id": 1, "subject": "Test", "body": "Body"}]
        mock_connect_to_mongodb.return_value = mock_collection
        mock_classify_email.return_value = ("label", 0.9, ["token1", "token2"])

        main()

        mock_load_model.assert_called_once()
        mock_connect_to_mongodb.assert_called_once()
        mock_classify_email.assert_called_once()
        mock_update_email_document.assert_called_once()

if __name__ == "__main__":
    unittest.main()