import unittest
from unittest.mock import MagicMock
from train.training import train_model

class TestTraining(unittest.TestCase):
    def test_train_model(self):
        mock_model = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_tokenizer = MagicMock()

        trainer = train_model(mock_model, mock_train_dataset, mock_eval_dataset, mock_tokenizer)

        self.assertIsNotNone(trainer)
        self.assertTrue(hasattr(trainer, "train"))

if __name__ == "__main__":
    unittest.main()