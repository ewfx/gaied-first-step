import unittest
import os
import tempfile
import json
from unittest.mock import patch
from GenerateEmailFilesFromJSON import create_email_from_json_file


class TestGenerateEmailFiles(unittest.TestCase):

    def setUp(self):
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.test_dir, "test_emails.json")
        self.attachment_dir = os.path.join(self.test_dir, "attachments")
        self.output_dir = os.path.join(self.test_dir, "output")

        os.makedirs(self.attachment_dir, exist_ok=True)

        # Create test JSON file
        test_data = [
            {
                "email_id": "TEST001",
                "sender": "Test Sender",
                "sender_email": "test@example.com",
                "subject": "Test Subject",
                "body": "Test Body",
                "attachments": [],
                "received_at": "2023-01-01T00:00:00.000Z",
                "is_duplicate": False
            }
        ]

        with open(self.json_path, 'w') as f:
            json.dump(test_data, f)

    def test_create_email_from_json_file(self):
        # Test with no attachments
        create_email_from_json_file(
            self.json_path,
            self.attachment_dir,
            self.output_dir
        )

        # Verify output file was created
        output_file = os.path.join(self.output_dir, "TEST001.eml")
        self.assertTrue(os.path.exists(output_file))

        # Verify content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("Test Subject", content)
            self.assertIn("Test Body", content)

    @patch('GenerateEmailFilesFromJSON.os.path.exists', return_value=True)
    @patch('GenerateEmailFilesFromJSON.open')
    def test_with_attachments(self, mock_open, mock_exists):
        # Add attachment to test data
        test_data = [
            {
                "email_id": "TEST002",
                "sender": "Test Sender",
                "sender_email": "test@example.com",
                "subject": "Test Subject",
                "body": "Test Body",
                "attachments": ["test.pdf"],
                "received_at": "2023-01-01T00:00:00.000Z",
                "is_duplicate": False
            }
        ]

        with open(self.json_path, 'w') as f:
            json.dump(test_data, f)

        # Create dummy attachment
        with open(os.path.join(self.attachment_dir, "test.pdf"), 'wb') as f:
            f.write(b"Dummy PDF content")

        create_email_from_json_file(
            self.json_path,
            self.attachment_dir,
            self.output_dir
        )

        # Verify calls
        mock_exists.assert_called()
        mock_open.assert_called()

    def tearDown(self):
        # Clean up temporary directories
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)


if __name__ == '__main__':
    unittest.main()
