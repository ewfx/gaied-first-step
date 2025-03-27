import unittest
from unittest.mock import patch
from runner.model_loader import load_model

class TestModelLoader(unittest.TestCase):
    @patch("runner.model_loader.AutoTokenizer.from_pretrained")
    @patch("runner.model_loader.AutoModelForSequenceClassification.from_pretrained")
    def test_load_model(self, mock_model, mock_tokenizer):
        mock_model.return_value = "mock_model"
        mock_tokenizer.return_value = "mock_tokenizer"

        model, tokenizer = load_model()

        self.assertEqual(model, "mock_model")
        self.assertEqual(tokenizer, "mock_tokenizer")
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

if __name__ == "__main__":
    unittest.main()