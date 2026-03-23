"""
Document Processing Service
Handles document loading and processing
"""

from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process various document types"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
    }
    
    @staticmethod
    def process_uploaded_file(
        file_content: bytes,
        filename: str
    ) -> List[Document]:
        """
        Process an uploaded file
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type not supported
        """
        # Get file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {list(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Load document
            loader_class = DocumentProcessor.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(tmp_path)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = filename
                doc.metadata['file_type'] = ext
            
            logger.info(f"Processed {filename}: {len(documents)} pages/chunks")
            
            return documents
        
        finally:
            # Clean up temporary file
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
            source_name: Name to use for source metadata
            
        Returns:
            List containing single Document
        """
        doc = Document(
            page_content=text,
            metadata={
                'source': source_name,
                'file_type': 'text'
            }
        )
        
        return [doc]