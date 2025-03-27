import unittest
from unittest.mock import MagicMock
from train.evaluation import evaluate_model

class TestEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        mock_trainer = MagicMock()
        mock_trainer.evaluate.return_value = {"accuracy": 0.9}

        results = evaluate_model(mock_trainer)
        self.assertEqual(results["accuracy"], 0.9)
        mock_trainer.evaluate.assert_called_once()

if __name__ == "__main__":
    unittest.main()