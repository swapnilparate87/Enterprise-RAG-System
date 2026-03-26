"""
Configuration Management - FREE Version
No API keys needed!
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings for FREE version"""

    # Application
    APP_NAME: str = "OpenRAG-Studio"
    APP_VERSION: str = "1.0.0-FREE"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API
    API_V1_PREFIX: str = "/api/v1"
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB

    # FREE LLM Settings - Ollama (runs locally)
    OLLAMA_MODEL: str = "gemma2:2b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # FREE Embeddings - Ollama nomic-embed-text (runs locally)
    # Run: ollama pull nomic-embed-text
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector Database - ChromaDB (FREE, local)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 5

    # Database
    DATABASE_URL: str = "sqlite:///./rag_system.db"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency to get settings"""
    return settings


if __name__ == "__main__":
    print("=" * 60)
    print("🎉 FREE RAG System Configuration")
    print("=" * 60)
    print(f"App Name:   {settings.APP_NAME}")
    print(f"Version:    {settings.APP_VERSION}")
    print(f"LLM Model:  {settings.OLLAMA_MODEL} (Ollama)")
    print(f"Embeddings: {settings.EMBEDDING_MODEL} (Ollama)")
    print(f"Vector DB:  ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY}")
    print("=" * 60)
    print("💰 Total Cost: $0/month")
    print("=" * 60)