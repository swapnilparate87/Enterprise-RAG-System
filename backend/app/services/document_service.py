"""
Document Processing Service
Handles document loading and processing for PDF, TXT, DOCX, and URLs
"""

import logging
import os
import tempfile
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document types"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }

    @staticmethod
    def process_uploaded_file(
        file_content: bytes,
        filename: str
    ) -> List[Document]:
        """
        Process an uploaded file (PDF, TXT, DOCX)

        Args:
            file_content: Raw file bytes
            filename: Original filename

        Returns:
            List of Document objects

        Raises:
            ValueError: If file type not supported
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {list(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())}"
            )

        # Save to a temporary file for the loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            loader_class = DocumentProcessor.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(tmp_path)
            documents = loader.load()

            # Attach source metadata
            for doc in documents:
                doc.metadata["source"] = filename
                doc.metadata["file_type"] = ext

            logger.info(f"Processed '{filename}': {len(documents)} page(s)/chunk(s)")
            return documents

        finally:
            os.unlink(tmp_path)

    @staticmethod
    def process_text(
        text: str,
        source_name: str = "user_input"
    ) -> List[Document]:
        """
        Process raw text input

        Args:
            text: Raw text content
            source_name: Label for source metadata

        Returns:
            List containing a single Document
        """
        doc = Document(
            page_content=text,
            metadata={
                "source": source_name,
                "file_type": "text"
            }
        )
        logger.info(f"Processed text input from '{source_name}'")
        return [doc]

    @staticmethod
    def process_url(url: str) -> List[Document]:
        """
        Load and process content from a web URL

        Args:
            url: Web page URL to scrape

        Returns:
            List of Document objects

        Raises:
            ValueError: If URL cannot be loaded
        """
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["file_type"] = "url"

            logger.info(f"Processed URL '{url}': {len(documents)} document(s)")
            return documents

        except Exception as e:
            logger.error(f"Failed to load URL '{url}': {e}")
            raise ValueError(f"Could not load URL: {url}. Error: {e}")