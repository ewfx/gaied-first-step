import unittest
from unittest.mock import patch
from train.model_initializer import initialize_model

class TestModelInitializer(unittest.TestCase):
    @patch("train.model_initializer.AutoModelForSequenceClassification.from_pretrained")
    def test_initialize_model(self, mock_model):
        mock_model.return_value = "mock_model"

        model = initialize_model("distilbert-base-uncased", num_labels=3)
        self.assertEqual(model, "mock_model")
        mock_model.assert_called_once_with("distilbert-base-uncased", num_labels=3)

if __name__ == "__main__":
    unittest.main()