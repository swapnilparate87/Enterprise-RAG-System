"""
FREE Version RAG Engine - Zero Cost!
Using: Ollama (local LLM) + HuggingFace (embeddings) + ChromaDB
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import logging
import time

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
    
    Uses:
    - Ollama for LLM (runs locally, completely free)
    - HuggingFace for embeddings (free, runs locally)
    - ChromaDB for vector storage (free, runs locally)
    
    Total Cost: $0/month! 🎉
    """
    
    def __init__(
        self,
        ollama_model: str = "mistral",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize FREE RAG engine
        
        Args:
            ollama_model: Ollama model name (mistral, llama2, phi)
            embedding_model: HuggingFace embedding model
            persist_directory: Directory to persist vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info("Initializing FREE RAG Engine...")
        
        # Initialize FREE embeddings (runs locally)
        logger.info(f"Loading HuggingFace embeddings: {embedding_model}")
        logger.info("First run will download model (~90MB), subsequent runs are instant")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("✅ Embeddings loaded successfully (100% FREE!)")
        
        # Initialize FREE LLM (Ollama - runs locally)
        logger.info(f"Connecting to Ollama model: {ollama_model}")
        logger.info("Make sure Ollama is running: ollama serve")
        
        try:
            self.llm = Ollama(
                model=ollama_model,
                temperature=0
            )
            
            # Test connection
            test_response = self.llm("Hi")
            logger.info("✅ Ollama connected successfully (100% FREE!)")
            
        except Exception as e:
            logger.error(f"❌ Could not connect to Ollama: {e}")
            logger.error("Please install Ollama from: https://ollama.ai")
            logger.error(f"Then run: ollama pull {ollama_model}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize FREE vector store (runs locally)
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        logger.info("✅ ChromaDB initialized (100% FREE!)")
        
        # QA prompt template (optimized for open-source LLMs)
        self.qa_template = """You are a helpful AI assistant. Use the following context to answer the question at the end.

If you don't know the answer based on the context, just say "I don't have enough information to answer this question." Don't make up an answer.

Context:
{context}

Question: {question}

Answer (be concise and cite sources when possible):"""

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        logger.info("🎉 FREE RAG Engine initialized successfully!")
        logger.info("💰 Total cost: $0/month")
    
    def ingest_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Ingest documents into the vector store
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of chunks to process at once
            
        Returns:
            Dict with ingestion statistics
        """
        logger.info(f"Starting document ingestion: {len(documents)} documents")
        start_time = time.time()
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vector store in batches
        logger.info("Creating embeddings and adding to vector store...")
        logger.info("(This runs locally and is completely FREE)")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Persist
        self.vectorstore.persist()
        
        ingestion_time = time.time() - start_time
        logger.info(f"✅ Document ingestion completed in {ingestion_time:.2f}s")
        logger.info("💰 Cost: $0")
        
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
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            RAGResponse with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}")
        total_start = time.time()
        
        # Retrieve relevant documents
        retrieval_start = time.time()
        
        if filter_dict:
            retrieved_docs = self.vectorstore.similarity_search(
                question,
                k=k,
                filter=filter_dict
            )
        else:
            retrieved_docs = self.vectorstore.similarity_search(question, k=k)
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s (FREE)")
        
        if len(retrieved_docs) == 0:
            return RAGResponse(
                answer="I don't have any documents to answer this question.",
                sources=[],
                confidence_score=0.0,
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.time() - total_start
            )
        
        # Generate answer using FREE local LLM
        generation_start = time.time()
        
        # Format context
        context = self._format_context(retrieved_docs)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": self.qa_prompt}
        )
        
        # Generate response
        logger.info("Generating answer with Ollama (running locally, FREE)...")
        result = qa_chain({"query": question})
        answer = result['result']
        
        generation_time = time.time() - generation_start
        
        # Calculate confidence score
        confidence_score = self._estimate_confidence(answer)
        
        # Format sources
        sources = self._format_sources(retrieved_docs)
        
        total_time = time.time() - total_start
        
        logger.info(f"✅ Query completed in {total_time:.2f}s (FREE)")
        logger.info(f"   Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s")
        logger.info("💰 Cost: $0")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time
        )
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            source_info = doc.metadata.get('source', 'Unknown')
            page_info = doc.metadata.get('page', '')
            
            context_parts.append(
                f"[Document {i+1}] (Source: {source_info}, Page: {page_info})\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        """Format documents into source metadata"""
        sources = []
        
        for i, doc in enumerate(documents):
            sources.append({
                "id": i + 1,
                "content": doc.page_content[:300] + "...",
                "metadata": {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', ''),
                    "chunk_id": doc.metadata.get('chunk_id', '')
                }
            })
        
        return sources
    
    def _estimate_confidence(self, answer: str) -> float:
        """Simple confidence estimation"""
        uncertainty_phrases = [
            "i don't have",
            "i'm not sure",
            "i cannot",
            "unclear",
            "not enough information"
        ]
        
        answer_lower = answer.lower()
        
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                return 0.3
        
        return 0.8
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        collection = self.vectorstore._collection
        
        return {
            "total_documents": collection.count(),
            "embedding_model": "HuggingFace (FREE)",
            "llm_model": "Ollama (FREE)",
            "total_cost": "$0/month"
        }
    
    def clear_database(self):
        """Clear all documents from vector store"""
        logger.warning("Clearing vector database...")
        self.vectorstore.delete_collection()
        logger.info("Vector database cleared")


# Example usage
if __name__ == "__main__":
    # Initialize FREE engine
    print("🎉 Initializing 100% FREE RAG Engine...")
    print("💰 Cost: $0/month")
    
    engine = FreeRAGEngine(
        ollama_model="mistral",  # or "llama2", "phi"
        persist_directory="./test_chroma_db"
    )
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={"source": "python_guide.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning is a subset of AI that focuses on learning from data.",
            metadata={"source": "ml_intro.pdf", "page": 1}
        )
    ]
    
    # Ingest (FREE)
    print("\n📚 Ingesting documents (FREE)...")
    stats = engine.ingest_documents(sample_docs)
    print(f"✅ Ingested {stats['num_chunks']} chunks in {stats['ingestion_time']:.2f}s")
    print("💰 Cost: $0")
    
    # Query (FREE)
    print("\n❓ Querying (FREE)...")
    result = engine.query("What is Python?")
    print(f"\n💡 Answer: {result.answer}")
    print(f"⏱️  Time: {result.total_time:.2f}s")
    print(f"💰 Cost: $0")