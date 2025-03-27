import unittest
from unittest.mock import MagicMock, patch
from duplicate_check.duplicate_check import generate_hash, fetch_extracted_data, check_duplicates

class TestDuplicateCheck(unittest.TestCase):
    def test_generate_hash(self):
        details = {
            "CustomerName": "John Doe",
            "SSN/TIN": "123-45-6789",
            "LoanAmount": "50000",
            "RequestType": "Loan Application",
            "SubRequestType": "Personal Loan",
        }
        expected_hash = "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2"
        result = generate_hash(details)
        self.assertEqual(len(result), 64)  # SHA-256 hash length is 64 characters
        self.assertIsInstance(result, str)

    @patch("duplicate_check.duplicate_check.MongoClient")
    def test_fetch_extracted_data(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": 1, "extractedKeyfields": {"CustomerName": "John Doe"}},
            {"_id": 2, "extractedKeyfields": {"CustomerName": "Jane Doe"}},
        ]

        result = fetch_extracted_data(mock_collection)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["_id"], 1)
        self.assertEqual(result[1]["_id"], 2)

    @patch("duplicate_check.duplicate_check.MongoClient")
    def test_check_duplicates(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=1)

        # Mock email data
        email_data = [
            {
                "_id": 1,
                "from": "test1@example.com",
                "extractedKeyfields": {
                    "CustomerName": "John Doe",
                    "SSN/TIN": "123-45-6789",
                    "LoanAmount": "50000",
                    "RequestType": "Loan Application",
                    "SubRequestType": "Personal Loan",
                },
            },
            {
                "_id": 2,
                "from": "test2@example.com",
                "extractedKeyfields": {
                    "CustomerName": "Jane Doe",
                    "SSN/TIN": "123-45-6789",
                    "LoanAmount": "50000",
                    "RequestType": "Loan Application",
                    "SubRequestType": "Personal Loan",
                },
            },
        ]

        # Call the function
        check_duplicates(email_data, mock_collection)

        # Assert update_one was called for duplicates
        mock_collection.update_one.assert_called_once_with(
            {"_id": 2},
            {"$set": {"isDuplicate": True, "confidenceCode": "High"}}
        )

if __name__ == "__main__":
    unittest.main()