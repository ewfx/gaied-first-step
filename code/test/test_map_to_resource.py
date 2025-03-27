import unittest
from unittest.mock import MagicMock, patch
from map_to_resource.map_to_resource import create_users, fetch_requests, assign_requests_to_users

class TestMapToResource(unittest.TestCase):
    def test_create_users(self):
        users = create_users()
        self.assertEqual(len(users), 10)  # Ensure 10 users are created
        self.assertIn("UserID", users[0])  # Check if UserID exists in the first user
        self.assertIn("SkillSet", users[0])  # Check if SkillSet exists in the first user

    @patch("map_to_resource.map_to_resource.MongoClient")
    def test_fetch_requests(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_collection.find.return_value = [
            {"_id": 1, "extractedKeyfields": {"RequestType": "Billing Issue", "SubRequestType": "Invoice Discrepancy"}},
            {"_id": 2, "extractedKeyfields": {"RequestType": "Technical Support", "SubRequestType": "Bug Report"}},
        ]

        result = fetch_requests(mock_collection)
        self.assertEqual(len(result), 2)  # Ensure 2 requests are fetched
        self.assertEqual(result[0]["_id"], 1)  # Check the first request's ID
        self.assertEqual(result[1]["_id"], 2)  # Check the second request's ID

    @patch("map_to_resource.map_to_resource.MongoClient")
    def test_assign_requests_to_users(self, mock_mongo_client):
        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_collection.update_one.return_value = MagicMock(matched_count=1)

        # Mock users
        users = [
            {"UserID": 1, "Name": "Alice", "SkillSet": {"Billing Issue": ["Invoice Discrepancy", "Payment Delay"]}},
            {"UserID": 2, "Name": "Bob", "SkillSet": {"Technical Support": ["Bug Report", "Feature Assistance"]}},
        ]

        # Mock requests
        requests = [
            {"_id": 1, "extractedKeyfields": {"RequestType": "Billing Issue", "SubRequestType": "Invoice Discrepancy"}},
            {"_id": 2, "extractedKeyfields": {"RequestType": "Technical Support", "SubRequestType": "Bug Report"}},
            {"_id": 3, "extractedKeyfields": {"RequestType": "General Inquiry", "SubRequestType": "Product Info"}},
        ]

        # Call the function
        assign_requests_to_users(users, requests, mock_collection)

        # Assert update_one was called for the first two requests
        mock_collection.update_one.assert_any_call(
            {"_id": 1},
            {"$set": {"AssignedUser": {"UserID": 1, "Name": "Alice"}}}
        )
        mock_collection.update_one.assert_any_call(
            {"_id": 2},
            {"$set": {"AssignedUser": {"UserID": 2, "Name": "Bob"}}}
        )

        # Assert no user was found for the third request
        self.assertEqual(mock_collection.update_one.call_count, 2)

if __name__ == "__main__":
    unittest.main()