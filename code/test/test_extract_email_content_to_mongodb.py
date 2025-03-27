import unittest
from unittest.mock import patch, MagicMock, mock_open
from extract_data_from_emails_attachments.extract_email_content_to_mongodb import (
    extract_text_from_image,
    process_msg_attachment,
    process_attachment,
    process_email_file,
    worker,
)


class TestExtractEmailContentToMongoDB(unittest.TestCase):
    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.Image.open")
    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.pytesseract.image_to_string")
    def test_extract_text_from_image(self, mock_image_to_string, mock_image_open):
        mock_image_to_string.return_value = "Extracted text from image"
        mock_image_open.return_value = MagicMock()

        image_bytes = b"fake_image_data"
        result = extract_text_from_image(image_bytes)

        self.assertEqual(result, "Extracted text from image")
        mock_image_open.assert_called_once()
        mock_image_to_string.assert_called_once()

    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.message_from_bytes")
    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.tempfile.NamedTemporaryFile")
    def test_process_msg_attachment(self, mock_tempfile, mock_message_from_bytes):
        mock_tempfile.return_value.__enter__.return_value.name = "temp_msg_file.msg"
        mock_message_from_bytes.return_value = MagicMock(
            get=lambda x: "Test Value",
            is_multipart=lambda: False,
            get_payload=lambda decode: b"Test Body",
        )

        attachment_name = "test.msg"
        attachment_content = b"fake_msg_data"
        result = process_msg_attachment(attachment_name, attachment_content)

        self.assertEqual(result["type"], "msg")
        self.assertEqual(result["from"], "Test Value")
        self.assertEqual(result["body"], "Test Body")

    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.Document")
    def test_process_attachment_docx(self, mock_document):
        mock_document.return_value.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]

        attachment_name = "test.docx"
        attachment_content = b"fake_docx_data"
        result = process_attachment(attachment_name, attachment_content)

        self.assertEqual(result["type"], "docx")
        self.assertEqual(result["content"], "Paragraph 1\nParagraph 2")

    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.PdfReader")
    def test_process_attachment_pdf(self, mock_pdf_reader):
        mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Page 1 text")]

        attachment_name = "test.pdf"
        attachment_content = b"fake_pdf_data"
        result = process_attachment(attachment_name, attachment_content)

        self.assertEqual(result["type"], "pdf")
        self.assertEqual(result["content"], "Page 1 text")

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_email_data")
    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.message_from_bytes")
    def test_process_email_file(self, mock_message_from_bytes, mock_open_file):
        mock_message_from_bytes.return_value = MagicMock(
            get=lambda x: "Test Value",
            is_multipart=lambda: False,
            get_payload=lambda decode: b"Test Body",
        )

        file_path = "test_email.eml"
        result = process_email_file(file_path)

        self.assertEqual(result["from"], "Test Value")
        self.assertEqual(result["body"], "Test Body")
        self.assertEqual(result["status"], "processed")

    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.MongoClient")
    @patch("extract_data_from_emails_attachments.extract_email_content_to_mongodb.process_email_file")
    def test_worker(self, mock_process_email_file, mock_mongo_client):
        mock_process_email_file.return_value = {
            "filename": "test_email.eml",
            "from": "Test Sender",
            "status": "processed",
        }
        mock_collection = MagicMock()
        mock_mongo_client.return_value.__getitem__.return_value = mock_collection

        file_path = "test_email.eml"
        worker(file_path)

        mock_process_email_file.assert_called_once_with(file_path)
        mock_collection.update_one.assert_called_once_with(
            {"filename": "test_email.eml"},
            {"$set": mock_process_email_file.return_value},
            upsert=True,
        )


if __name__ == "__main__":
    unittest.main()