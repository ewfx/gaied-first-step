import unittest
from unittest.mock import MagicMock, patch
from key_extraction.extraction_of_key_feilds import extract_key_details, process_requests

class TestExtractionOfKeyFields(unittest.TestCase):
    def test_extract_key_details(self):
        email_data = {
            "subject": "Name: John Doe Request Type: Loan Application",
            "body": "SSN: 123-45-6789 Loan Amount: 50,000 Sub-Request Type: Personal Loan",
            "attachments": [
                {"content": "Additional Info: None"}
            ],
        }

        expected_output = {
            "CustomerName": "John Doe",
            "SSN/TIN": "123-45-6789",
            "LoanAmount": "50000",
            "RequestType": "Loan Application",
            "SubRequestType": "Personal Loan",
        }

        result = extract_key_details(email_data)
        self.assertEqual(result, expected_output)

    def test_extract_key_details_with_missing_fields(self):
        email_data = {
            "subject": "Request Type: Account Update",
            "body": "",
            "attachments": [],
        }

        expected_output = {
            "CustomerName": None,
            "SSN/TIN": None,
            "LoanAmount": None,
            "RequestType": "Account Update",
            "SubRequestType": None,
        }

        result = extract_key_details(email_data)
        self.assertEqual(result, expected_output)

    @patch("key_extraction.extraction_of_key_feilds.MongoClient")
    def test_process_requests(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=1)

        # Mock emails
        request_emails = [
            {
                "_id": 1,
                "from": "test@example.com",
                "date": "2025-03-27",
                "subject": "Name: Jane Doe Request Type: Account Update",
                "body": "SSN: 987-65-4321 Loan Amount: 10,000 Sub-Request Type: Business Loan",
                "attachments": [],
            }
        ]

        # Call the function
        process_requests(request_emails, mock_collection)

        # Assert update_one was called with correct arguments
        mock_collection.update_one.assert_called_once_with(
            {"_id": 1},
            {"$set": {
                "extractedKeyfields": {
                    "CustomerName": "Jane Doe",
                    "SSN/TIN": "987-65-4321",
                    "LoanAmount": "10000",
                    "RequestType": "Account Update",
                    "SubRequestType": "Business Loan",
                    "_id": 1,
                    "from": "test@example.com",
                    "date": "2025-03-27",
                }
            }}
        )

    @patch("key_extraction.extraction_of_key_feilds.MongoClient")
    def test_process_requests_with_empty_emails(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()

        # Call the function with an empty list
        process_requests([], mock_collection)

        # Assert update_one was not called
        mock_collection.update_one.assert_not_called()

if __name__ == "__main__":
    unittest.main()