import unittest
import os
import tempfile
import email
from email.message import Message
from unittest.mock import patch, MagicMock
from ExtractEmailContentToMongoDB import (
    extract_text_from_image,
    process_attachment,
    process_email_file,
    worker
)


class TestExtractEmailContent(unittest.TestCase):

    def test_extract_text_from_image(self):
        # This will actually try to run OCR, so we just verify it doesn't crash
        # In a real test, you'd mock pytesseract
        try:
            from PIL import Image
            import io

            # Create a simple white image
            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')

            result = extract_text_from_image(img_bytes.getvalue())
            self.assertIsInstance(result, str)
        except ImportError:
            self.skipTest("Pillow not available for image testing")

    def test_process_attachment_text(self):
        result = process_attachment("test.txt", b"Test content")
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["content"], "Test content")

    @patch('ExtractEmailContentToMongoDB.Document')
    def test_process_attachment_docx(self, mock_docx):
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Test paragraph")]
        mock_docx.return_value = mock_doc

        result = process_attachment("test.docx", b"dummy")
        self.assertEqual(result["type"], "docx")
        self.assertEqual(result["content"], "Test paragraph")

    def test_process_email_file(self):
        # Create a test email file
        msg = Message()
        msg["From"] = "test@example.com"
        msg["Subject"] = "Test Subject"
        msg["Date"] = "Mon, 1 Jan 2023 00:00:00 +0000"
        msg.set_payload("Test body")

        with tempfile.NamedTemporaryFile(suffix='.eml', delete=False) as tmp:
            tmp.write(msg.as_bytes())
            tmp_path = tmp.name

        try:
            result = process_email_file(tmp_path)
            self.assertEqual(result["from"], "test@example.com")
            self.assertEqual(result["subject"], "Test Subject")
            self.assertEqual(result["body"], "Test body")
        finally:
            os.unlink(tmp_path)

    @patch('ExtractEmailContentToMongoDB.MongoClient')
    @patch('ExtractEmailContentToMongoDB.process_email_file')
    def test_worker(self, mock_process, mock_mongo):
        # Setup mocks
        test_data = {
            "filename": "test.eml",
            "from": "test@example.com",
            "status": "processed"
        }
        mock_process.return_value = test_data

        mock_collection = MagicMock()
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_mongo.return_value.__getitem__.return_value = mock_db

        # Test
        worker("dummy_path")

        # Verify
        mock_collection.update_one.assert_called_once_with(
            {"filename": "test.eml"},
            {"$set": test_data},
            upsert=True
        )


if __name__ == '__main__':
    unittest.main()
