"""
FREE Version RAG Engine - Zero Cost!
Using: Ollama (local LLM + embeddings) + ChromaDB (vector store)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from RAG system"""
    answer: str
    sources: List[Dict]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_time: float


class FreeRAGEngine:
    """
    100% FREE RAG Engine
    Uses: Ollama LLM + Ollama Embeddings + ChromaDB
    Total Cost: $0/month!
    """

    def __init__(
        self,
        ollama_model: str = "mistral",
        embedding_model: str = "nomic-embed-text",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_model = ollama_model

        logger.info("Initializing FREE RAG Engine...")

        # FREE embeddings via Ollama
        logger.info(f"Loading Ollama embeddings: {embedding_model}")
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )
        logger.info("✅ Embeddings loaded (100% FREE!)")

        # FREE LLM via Ollama — optimized settings for speed
        logger.info(f"Connecting to Ollama LLM: {ollama_model}")
        try:
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.1,
                num_predict=512,         # enough for detailed answers
                num_ctx=2048,            # fits within 4GB VRAM
                repeat_penalty=1.1,
                top_k=40,
                top_p=0.9,
            )
            # ✅ No test call on startup — saves 5-10s every restart
            logger.info("✅ Ollama LLM configured (100% FREE!)")
        except Exception as e:
            logger.error(f"❌ Could not connect to Ollama: {e}")
            raise

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # FREE vector store via ChromaDB
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        logger.info("✅ ChromaDB initialized (100% FREE!)")

        # ✅ Improved RAG prompt — detailed, structured answers
        self.qa_prompt = PromptTemplate(
            template="""You are an expert AI assistant with deep knowledge. Your job is to give thorough, accurate, and well-structured answers based on the provided context.

INSTRUCTIONS:
- Read the context carefully and extract ALL relevant information
- Give a detailed, comprehensive answer — do not be vague or overly brief
- Use bullet points or numbered lists when listing multiple items
- Always cite which document/source supports each point
- If the context contains numbers, statistics, or specific details — include them
- If the context does NOT contain enough information, clearly say so and explain what you do know
- Never make up information that is not in the context

CONTEXT:
{context}

QUESTION: {question}

DETAILED ANSWER:""",
            input_variables=["context", "question"]
        )

        logger.info("🎉 FREE RAG Engine ready! Total cost: $0/month")

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved docs into a rich context string"""
        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", Page: {page}" if page != "" else ""
            parts.append(
                f"[Document {i+1}] Source: {source}{page_info}\n"
                f"{'-' * 40}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    def ingest_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict:
        """Ingest documents into the vector store"""
        logger.info(f"Ingesting {len(documents)} document(s)...")
        start_time = time.time()

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"Batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} done")

        ingestion_time = time.time() - start_time
        logger.info(f"✅ Ingestion complete in {ingestion_time:.2f}s")

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "ingestion_time": ingestion_time
        }

    def query(
        self,
        question: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> RAGResponse:
        """Query the RAG system"""
        logger.info(f"Query: {question[:100]}")
        total_start = time.time()

        # Retrieve relevant docs
        retrieval_start = time.time()
        if filter_dict:
            retrieved_docs = self.vectorstore.similarity_search(question, k=k, filter=filter_dict)
        else:
            retrieved_docs = self.vectorstore.similarity_search(question, k=k)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieved {len(retrieved_docs)} docs in {retrieval_time:.2f}s")

        if not retrieved_docs:
            return RAGResponse(
                answer="I don't have any documents in the knowledge base to answer this question. Please upload some documents first.",
                sources=[],
                confidence_score=0.0,
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.time() - total_start
            )

        # Build LCEL chain with current k
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

        # Generate answer
        generation_start = time.time()
        logger.info("Generating answer with Ollama...")
        answer = chain.invoke(question)
        generation_time = time.time() - generation_start

        total_time = time.time() - total_start
        logger.info(f"✅ Done in {total_time:.2f}s")

        return RAGResponse(
            answer=answer,
            sources=self._format_sources(retrieved_docs),
            confidence_score=self._estimate_confidence(answer),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time
        )

    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        """Format documents into source metadata"""
        return [
            {
                "id": i + 1,
                "content": doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", ""),
                    "chunk_id": doc.metadata.get("chunk_id", "")
                }
            }
            for i, doc in enumerate(documents)
        ]

    def _estimate_confidence(self, answer: str) -> float:
        """Simple confidence estimation"""
        uncertainty_phrases = [
            "i don't have", "i'm not sure", "i cannot",
            "unclear", "not enough information", "no information"
        ]
        for phrase in uncertainty_phrases:
            if phrase in answer.lower():
                return 0.3
        return 0.8

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        collection = self.vectorstore._collection
        return {
            "total_documents": collection.count(),
            "embedding_model": "nomic-embed-text via Ollama (FREE)",
            "llm_model": f"{self.ollama_model} via Ollama (FREE)",
            "total_cost": "$0/month"
        }

    def clear_database(self):
        """Clear all documents from vector store"""
        logger.warning("Clearing vector database...")
        self.vectorstore.delete_collection()
        logger.info("Vector database cleared")