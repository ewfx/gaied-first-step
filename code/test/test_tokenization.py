import unittest
from unittest.mock import MagicMock
from train.tokenization import tokenize_dataset

class TestTokenization(unittest.TestCase):
    def test_tokenize_dataset(self):
        mock_dataset = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        tokenized_dataset = tokenize_dataset(mock_dataset, mock_tokenizer, max_length=128)
        self.assertIsNotNone(tokenized_dataset)
        mock_dataset.map.assert_called_once()

if __name__ == "__main__":
    unittest.main()