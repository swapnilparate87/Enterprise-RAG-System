"""
Main FastAPI Application - 100% FREE VERSION
Uses: Ollama + ChromaDB
Total Cost: $0/month! 🎉
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.core.rag_engine import FreeRAGEngine
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    TextIngestionRequest,
)
from app.services.document_service import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global RAG engine
rag_engine: FreeRAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up FREE RAG engine"""
    global rag_engine

    logger.info("=" * 60)
    logger.info("🎉 Starting 100% FREE RAG System")
    logger.info(f"🤖 LLM:        {settings.OLLAMA_MODEL} (Ollama, local)")
    logger.info(f"🧮 Embeddings: {settings.EMBEDDING_MODEL} (Ollama, local)")
    logger.info("🗄️  Vector DB:  ChromaDB (local)")
    logger.info("💰 Total Cost: $0/month")
    logger.info("=" * 60)

    try:
        rag_engine = FreeRAGEngine(
            ollama_model=settings.OLLAMA_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            ollama_base_url=settings.OLLAMA_BASE_URL,
        )
        logger.info("✅ RAG Engine initialized! API docs: http://localhost:8000/docs")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG engine: {e}")
        logger.error("🔧 Is Ollama running? Run: ollama serve")
        logger.error(f"🔧 Model installed? Run: ollama pull {settings.OLLAMA_MODEL}")
        raise

    yield  # App runs here

    logger.info("Shutting down FREE RAG System... 💰 Total session cost: $0")


# Create FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - 100% FREE Version 🆓",
    version=settings.APP_VERSION,
    description="""
    Production-ready RAG system built with 100% FREE resources!

    **Stack:** Ollama LLM + Ollama Embeddings + ChromaDB + FastAPI

    **Total Cost: $0/month** 💰
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="Invalid input", detail=str(exc)).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else "An error occurred",
        ).dict(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy 🎉",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        app_name=f"{settings.APP_NAME} (FREE 💰$0/mo)",
        version=settings.APP_VERSION,
    )


@app.post("/api/v1/upload", response_model=IngestionResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document — PDF, TXT, or DOCX"""
    logger.info(f"📤 Received file: {file.filename}")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE} bytes",
        )

    try:
        documents = DocumentProcessor.process_uploaded_file(
            file_content=content, filename=file.filename
        )
        stats = rag_engine.ingest_documents(documents)
        return IngestionResponse(
            message=f"'{file.filename}' ingested successfully (FREE!)",
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"],
            ingestion_time=stats["ingestion_time"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing document")


@app.post("/api/v1/ingest-text", response_model=IngestionResponse)
async def ingest_text(request: TextIngestionRequest):
    """Ingest raw text content"""
    logger.info(f"📝 Ingesting text from: {request.source_name}")
    try:
        documents = DocumentProcessor.process_text(
            text=request.text, source_name=request.source_name
        )
        stats = rag_engine.ingest_documents(documents)
        return IngestionResponse(
            message="Text ingested successfully (FREE!)",
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"],
            ingestion_time=stats["ingestion_time"],
        )
    except Exception as e:
        logger.error(f"Text ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error ingesting text")


class URLIngestionRequest(BaseModel):
    url: str
    

@app.post("/api/v1/ingest-url", response_model=IngestionResponse)
async def ingest_url(request: URLIngestionRequest):
    """Scrape and ingest content from a web URL"""
    logger.info(f"🌐 Ingesting URL: {request.url}")
    try:
        documents = DocumentProcessor.process_url(request.url)
        stats = rag_engine.ingest_documents(documents)
        return IngestionResponse(
            message=f"URL '{request.url}' ingested successfully (FREE!)",
            num_documents=stats["num_documents"],
            num_chunks=stats["num_chunks"],
            ingestion_time=stats["ingestion_time"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"URL ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error ingesting URL")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    logger.info(f"❓ Query: {request.question[:100]}")
    try:
        result = rag_engine.query(question=request.question, k=request.k)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence_score": result.confidence_score,
            "retrieval_time": result.retrieval_time,
            "generation_time": result.generation_time,
            "total_time": result.total_time,
            "model_used": rag_engine.ollama_model,  # ✅ actual model that answered
        }
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics"""
    try:
        stats = rag_engine.get_stats()
        return StatsResponse(
            total_documents=stats["total_documents"],
            embedding_dimension=768,  # nomic-embed-text dimension
            app_version=f"{settings.APP_VERSION} (FREE Edition)",
        )
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


@app.delete("/api/v1/clear")
async def clear_database():
    """Clear all documents from the vector store"""
    global rag_engine
    logger.warning("🗑️ Database clear requested")
    try:
        rag_engine.clear_database()
        # Reinitialize
        rag_engine = FreeRAGEngine(
            ollama_model=settings.OLLAMA_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            ollama_base_url=settings.OLLAMA_BASE_URL,
        )
        return {"message": "Database cleared and re-initialized successfully"}
    except Exception as e:
        logger.error(f"Clear error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error clearing database")




@app.post("/api/v1/query-stream")
async def query_stream(request: QueryRequest):
    """Query with streaming response — tokens appear as they are generated"""
    from fastapi.responses import StreamingResponse
    from langchain_core.output_parsers import StrOutputParser

    def generate():
        try:
            # Retrieve docs
            retrieved_docs = rag_engine.vectorstore.similarity_search(
                request.question, k=request.k
            )
            if not retrieved_docs:
                yield "I don't have any documents to answer this question."
                return

            context = rag_engine._format_docs(retrieved_docs)
            chain = rag_engine.qa_prompt | rag_engine.llm | StrOutputParser()

            # Stream tokens
            for chunk in rag_engine.llm.stream(
                rag_engine.qa_prompt.format(context=context, question=request.question)
            ):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


class SwitchModelRequest(BaseModel):
    model_name: str


# Model-specific settings optimized for each model size
MODEL_CONFIGS = {
    "qwen2:1.5b":     {"num_predict": 256,  "num_ctx": 1024, "top_k": 20},
    "gemma2:2b":      {"num_predict": 384,  "num_ctx": 2048, "top_k": 30},
    "mistral:latest": {"num_predict": 512,  "num_ctx": 3072, "top_k": 40},
    "deepseek-r1:8b": {"num_predict": 512,  "num_ctx": 3072, "top_k": 40},
}


@app.post("/api/v1/switch-model")
async def switch_model(request: SwitchModelRequest):
    """Switch the active LLM model at runtime with model-optimized settings"""
    global rag_engine
    logger.info(f"Switching model to: {request.model_name}")
    try:
        from langchain_ollama import OllamaLLM

        # Get optimized config for this model (fallback to safe defaults)
        cfg = MODEL_CONFIGS.get(request.model_name, {"num_predict": 256, "num_ctx": 1024, "top_k": 20})

        new_llm = OllamaLLM(
            model=request.model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=cfg["num_predict"],
            num_ctx=cfg["num_ctx"],
            repeat_penalty=1.1,
            top_k=cfg["top_k"],
            top_p=0.85,
        )

        # Test the model responds before committing the switch
        test = new_llm.invoke("Hi")
        if not test:
            raise Exception("Model did not respond to test prompt")

        rag_engine.llm = new_llm
        rag_engine.ollama_model = request.model_name
        logger.info(f"✅ Model switched to: {request.model_name} (ctx={cfg['num_ctx']}, predict={cfg['num_predict']})")
        return {
            "message": f"Model switched to {request.model_name}",
            "model": request.model_name,
            "config": cfg
        }
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🎉 Starting 100% FREE RAG System")
    print("⚠️  Make sure Ollama is running: ollama serve")
    print("=" * 60)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)