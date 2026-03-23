"""
Main FastAPI Application - 100% FREE VERSION
Uses: Ollama + HuggingFace + ChromaDB
Total Cost: $0/month! 🎉
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import settings
from app.core.rag_engine import FreeRAGEngine
from app.services.document_service import DocumentProcessor
from app.models.schemas import (
    QueryRequest, QueryResponse, TextIngestionRequest,
    IngestionResponse, StatsResponse, HealthResponse, ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - 100% FREE Version 🆓",
    version=settings.APP_VERSION,
    description="""
    Production-ready RAG system built with 100% FREE resources!
    
    **Technology Stack:**
    - 🤖 LLM: Ollama (runs locally, completely free)
    - 🧮 Embeddings: HuggingFace (runs locally, completely free)
    - 🗄️ Vector DB: ChromaDB (runs locally, completely free)
    - 🚀 Backend: FastAPI
    
    **Total Cost: $0/month** 💰
    
    Compare this to paid alternatives:
    - OpenAI API: $20-100/month
    - Pinecone: $70/month
    - Total savings: $1,000+/year!
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine instance
rag_engine: FreeRAGEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize FREE RAG engine on startup"""
    global rag_engine
    
    logger.info("=" * 60)
    logger.info("🎉 Starting 100% FREE RAG System")
    logger.info("=" * 60)
    logger.info("💰 Total Cost: $0/month")
    logger.info("🤖 LLM: Ollama (local, free)")
    logger.info("🧮 Embeddings: HuggingFace (local, free)")
    logger.info("🗄️ Vector DB: ChromaDB (local, free)")
    logger.info("=" * 60)
    
    try:
        logger.info("Initializing RAG Engine...")
        logger.info("⚠️  Make sure Ollama is running: ollama serve")
        
        rag_engine = FreeRAGEngine(
            ollama_model=settings.OLLAMA_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        logger.info("=" * 60)
        logger.info("✅ FREE RAG Engine initialized successfully!")
        logger.info("💰 Cost so far: $0")
        logger.info("🌐 API docs: http://localhost:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ Failed to initialize RAG engine")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        logger.error("🔧 Troubleshooting:")
        logger.error("1. Is Ollama running? Run: ollama serve")
        logger.error(f"2. Is model installed? Run: ollama pull {settings.OLLAMA_MODEL}")
        logger.error("3. Check your .env file configuration")
        logger.error("=" * 60)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down FREE RAG System...")
    logger.info("💰 Total cost during this session: $0")


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Invalid input",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else "An error occurred"
        ).dict()
    )


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with system info"""
    return HealthResponse(
        status="healthy - 100% FREE! 🎉",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        app_name=f"{settings.APP_NAME} (FREE Version 💰$0/mo)",
        version=settings.APP_VERSION
    )


@app.post("/api/v1/upload", response_model=IngestionResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (100% FREE)
    
    Supported formats: PDF, TXT, DOCX
    
    **Cost**: $0 (runs locally with free models)
    """
    logger.info(f"📤 Received file upload: {file.filename}")
    logger.info("💰 Processing cost: $0")
    
    # Validate file size
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    try:
        # Process document
        documents = DocumentProcessor.process_uploaded_file(
            file_content=content,
            filename=file.filename
        )
        
        # Ingest into RAG system (100% FREE)
        logger.info("🔄 Creating embeddings with FREE HuggingFace model...")
        stats = rag_engine.ingest_documents(documents)
        
        logger.info("✅ Document ingested successfully!")
        logger.info(f"💰 Total cost: $0 (vs ~${len(content)/1000 * 0.0004:.4f} with OpenAI)")
        
        return IngestionResponse(
            message=f"Document ingested successfully (FREE! Saved ~${len(content)/1000 * 0.0004:.4f})",
            num_documents=stats['num_documents'],
            num_chunks=stats['num_chunks'],
            ingestion_time=stats['ingestion_time']
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing document")


@app.post("/api/v1/ingest-text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """
    Ingest raw text content (100% FREE)
    
    **Cost**: $0 (runs locally)
    """
    logger.info(f"📝 Ingesting text from: {request.source_name}")
    logger.info("💰 Cost: $0")
    
    try:
        # Process text
        documents = DocumentProcessor.process_text(
            text=request.text,
            source_name=request.source_name
        )
        
        # Ingest (FREE)
        stats = rag_engine.ingest_documents(documents)
        
        return IngestionResponse(
            message="Text ingested successfully (100% FREE!)",
            num_documents=stats['num_documents'],
            num_chunks=stats['num_chunks'],
            ingestion_time=stats['ingestion_time']
        )
    
    except Exception as e:
        logger.error(f"Error ingesting text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error ingesting text")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system (100% FREE)
    
    Uses:
    - FREE Ollama for answer generation (runs locally)
    - FREE HuggingFace for embeddings (runs locally)
    - FREE ChromaDB for vector search (runs locally)
    
    **Cost**: $0 per query
    **Compare to**: $0.002-0.03 per query with OpenAI
    """
    logger.info(f"❓ Processing query: {request.question[:100]}")
    logger.info("💰 Query cost: $0 (runs locally)")
    
    try:
        # Query RAG engine (100% FREE)
        result = rag_engine.query(
            question=request.question,
            k=request.k
        )
        
        # Calculate savings
        estimated_openai_cost = 0.002  # Approximate cost per query with OpenAI
        
        logger.info(f"✅ Query completed in {result.total_time:.2f}s")
        logger.info(f"💰 Cost: $0 (vs ~${estimated_openai_cost} with OpenAI)")
        
        # Format sources
        sources = []
        for source in result.sources:
            sources.append({
                "id": source["id"],
                "content": source["content"],
                "metadata": source["metadata"]
            })
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            confidence_score=result.confidence_score,
            retrieval_time=result.retrieval_time,
            generation_time=result.generation_time,
            total_time=result.total_time
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get system statistics
    
    Shows how much you're saving by using FREE resources! 💰
    """
    try:
        stats = rag_engine.get_stats()
        
        return StatsResponse(
            total_documents=stats['total_documents'],
            embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
            app_version=f"{settings.APP_VERSION} (FREE Edition)"
        )
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


@app.delete("/api/v1/clear")
async def clear_database():
    """Clear all documents from the database"""
    logger.warning("🗑️ Database clear requested")
    
    try:
        rag_engine.clear_database()
        
        # Reinitialize
        global rag_engine
        rag_engine = FreeRAGEngine(
            ollama_model=settings.OLLAMA_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        return {
            "message": "Database cleared successfully",
            "cost": "$0 (always free!)"
        }
    
    except Exception as e:
        logger.error(f"Error clearing database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error clearing database")


@app.get("/api/v1/cost-savings")
async def get_cost_savings():
    """
    Calculate how much money you're saving by using FREE resources
    """
    stats = rag_engine.get_stats()
    
    # Rough estimates (conservative)
    docs_count = stats['total_documents']
    estimated_queries = 100  # Assume 100 queries
    
    # OpenAI costs (approximate)
    embedding_cost = docs_count * 0.0001  # $0.0001 per 1K tokens
    query_cost = estimated_queries * 0.002  # $0.002 per query
    pinecone_cost = 70  # $70/month for basic tier
    
    total_saved = embedding_cost + query_cost + pinecone_cost
    
    return {
        "your_cost": "$0/month",
        "estimated_openai_cost": f"${embedding_cost + query_cost:.2f}/month",
        "estimated_pinecone_cost": f"${pinecone_cost}/month",
        "total_monthly_savings": f"${total_saved:.2f}",
        "annual_savings": f"${total_saved * 12:.2f}",
        "message": "🎉 You're running a production RAG system for FREE!"
    }


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🎉 Starting 100% FREE RAG System")
    print("💰 Total Cost: $0/month")
    print("=" * 60)
    print("⚠️  Make sure Ollama is running in another terminal:")
    print("   ollama serve")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )