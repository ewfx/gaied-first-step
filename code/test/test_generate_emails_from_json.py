import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
from extract_data_from_emails_attachments.generate_emails_from_json import create_email_from_json_file


class TestGenerateEmailsFromJson(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps([
        {
            "email_id": "123",
            "sender": "John Doe",
            "sender_email": "john.doe@example.com",
            "subject": "Test Email",
            "body": "This is a test email.",
            "attachments": ["file1.txt"],
            "received_at": "2025-03-27T10:00:00.000",
            "is_duplicate": True
        }
    ]))
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    @patch("builtins.open", new_callable=mock_open)
    def test_create_email_from_json_file(
        self, mock_file_open, mock_path_join, mock_path_exists, mock_makedirs, mock_json_open
    ):
        # Mock os.path.exists to simulate attachment existence
        mock_path_exists.side_effect = lambda path: "file1.txt" in path

        # Call the function
        create_email_from_json_file(
            json_file_path="mock_json_file.json",
            attachment_folder="mock_attachments",
            output_folder="mock_output"
        )

        # Assert that the output folder was created
        mock_makedirs.assert_called_once_with("mock_output", exist_ok=True)

        # Assert that the email file was written
        self.assertEqual(mock_file_open.call_count, 2)  # One for primary email, one for duplicate
        mock_file_open.assert_any_call("mock_output/123.eml", "w", encoding="utf-8")
        mock_file_open.assert_any_call("mock_output/123_DUPLICATE.eml", "w", encoding="utf-8")

        # Assert that the attachment was read
        mock_file_open.assert_any_call("mock_attachments/file1.txt", "rb")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps([]))
    @patch("os.makedirs")
    def test_create_email_from_empty_json(self, mock_makedirs, mock_json_open):
        # Call the function with an empty JSON file
        create_email_from_json_file(
            json_file_path="mock_json_file.json",
            attachment_folder="mock_attachments",
            output_folder="mock_output"
        )

        # Assert that the output folder was created
        mock_makedirs.assert_called_once_with("mock_output", exist_ok=True)

        # Assert that no emails were generated
        mock_json_open.assert_called_once_with("mock_json_file.json", "r", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps([
        {
            "email_id": "123",
            "sender": "John Doe",
            "sender_email": "john.doe@example.com",
            "subject": "Test Email",
            "body": "This is a test email.",
            "attachments": ["missing_file.txt"],
            "received_at": "2025-03-27T10:00:00.000",
            "is_duplicate": False
        }
    ]))
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    @patch("builtins.open", new_callable=mock_open)
    def test_create_email_with_missing_attachment(
        self, mock_file_open, mock_path_join, mock_path_exists, mock_makedirs, mock_json_open
    ):
        # Mock os.path.exists to simulate missing attachment
        mock_path_exists.side_effect = lambda path: False

        # Call the function
        create_email_from_json_file(
            json_file_path="mock_json_file.json",
            attachment_folder="mock_attachments",
            output_folder="mock_output"
        )

        # Assert that the output folder was created
        mock_makedirs.assert_called_once_with("mock_output", exist_ok=True)

        # Assert that the email file was written
        mock_file_open.assert_called_once_with("mock_output/123.eml", "w", encoding="utf-8")

        # Assert that the missing attachment was logged
        mock_file_open.assert_any_call("mock_json_file.json", "r", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()