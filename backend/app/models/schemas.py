"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# Request Models
class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the revenue for Q3?",
                "k": 5
            }
        }


class TextIngestionRequest(BaseModel):
    """Request model for ingesting raw text"""
    text: str = Field(..., min_length=1, description="Text content to ingest")
    source_name: str = Field(default="user_input", description="Name for this text source")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is a subset of AI...",
                "source_name": "ml_notes.txt"
            }
        }


# Response Models
class SourceMetadata(BaseModel):
    """Metadata for a source document"""
    source: str
    page: Optional[str] = None
    chunk_id: Optional[str] = None


class Source(BaseModel):
    """Source document information"""
    id: int
    content: str
    metadata: SourceMetadata


class QueryResponse(BaseModel):
    """Response model for query results"""
    answer: str
    sources: List[Source]
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_time: float
    generation_time: float
    total_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The Q3 revenue was $5.2 billion.",
                "sources": [
                    {
                        "id": 1,
                        "content": "Q3 financial results showed revenue of $5.2B...",
                        "metadata": {
                            "source": "q3_report.pdf",
                            "page": "3"
                        }
                    }
                ],
                "confidence_score": 0.95,
                "retrieval_time": 0.5,
                "generation_time": 1.2,
                "total_time": 1.7,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class IngestionResponse(BaseModel):
    """Response model for document ingestion"""
    message: str
    num_documents: int
    num_chunks: int
    ingestion_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Documents ingested successfully",
                "num_documents": 5,
                "num_chunks": 47,
                "ingestion_time": 12.3,
                "timestamp": "2024-01-15T10:25:00"
            }
        }


class StatsResponse(BaseModel):
    """Response model for system statistics"""
    total_documents: int
    embedding_dimension: int
    app_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 150,
                "embedding_dimension": 1536,
                "app_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    app_name: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "app_name": "Enterprise RAG System",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:20:00"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "File type not supported",
                "detail": "Only PDF, TXT, and DOCX files are supported",
                "timestamp": "2024-01-15T10:22:00"
            }
        }