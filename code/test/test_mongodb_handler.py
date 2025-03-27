import unittest
from unittest.mock import patch, MagicMock
from runner.mongodb_handler import connect_to_mongodb, update_email_document

class TestMongoDBHandler(unittest.TestCase):
    @patch("runner.mongodb_handler.MongoClient")
    def test_connect_to_mongodb(self, mock_mongo_client):
        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_client
        mock_client.get_collection.return_value = "mock_collection"

        collection = connect_to_mongodb()
        self.assertEqual(collection, "mock_collection")
        mock_mongo_client.assert_called_once()

    def test_update_email_document(self):
        mock_collection = MagicMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=1)

        result = update_email_document(mock_collection, 1, "label", 0.9, ["token1", "token2"])
        self.assertTrue(result)
        mock_collection.update_one.assert_called_once_with(
            {"_id": 1},
            {"$set": {"predicted_label": "label", "confidence_score": 0.9, "important_tokens": ["token1", "token2"]}},
        )

if __name__ == "__main__":
    unittest.main()