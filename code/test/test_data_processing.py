import unittest
from train.data_processing import process_emails

class TestDataProcessing(unittest.TestCase):
    def test_process_emails(self):
        emails = [
            {"subject": "Test Subject", "body": "Test Body", "is_duplicate": True},
            {"subject": "Another Subject", "body": "Another Body", "is_update_case": True},
            {"subject": "New Request", "body": "Request Body"},
        ]

        result = process_emails(emails)
        self.assertEqual(len(result["text"]), 3)
        self.assertEqual(len(result["label"]), 3)
        self.assertEqual(result["label"], [0, 1, 2])  # Assuming label mapping: duplicate=0, update=1, new request=2

if __name__ == "__main__":
    unittest.main()