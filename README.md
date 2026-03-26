# 🧠 Enterprise RAG System

> A 100% FREE, production-ready Retrieval Augmented Generation (RAG) system built with local AI. No API keys, no cloud costs, complete data privacy.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [Supported Models](#supported-models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

Enterprise RAG System lets you upload any document (PDF, TXT, DOCX) or webpage and ask questions about it using a local LLM running entirely on your machine. No data ever leaves your computer.

**Total running cost: $0/month.**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (Mistral, Gemma2, Qwen2, DeepSeek) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Framework | LangChain 0.3 |
| Language | Python 3.14 |

---

## Features

- 📄 **Multi-format ingestion** — PDF, TXT, DOCX, URLs
- 🤖 **Multi-model support** — switch between models at runtime
- 🌊 **Streaming responses** — tokens appear live as generated
- 💬 **Chat history** — full conversation saved in session
- 📥 **Export chat** — download as TXT or JSON
- 📊 **Vector store stats** — live chunk count and model info
- 🗄️ **Document management** — track and clear ingested docs
- 🔍 **MMR retrieval** — diverse, relevant chunk selection
- 🎨 **Dark theme UI** — GitHub-inspired design
- 🔒 **100% private** — everything runs locally

---

## Project Structure

```
Enterprise-RAG-System/
├── backend/
│   └── app/
│       ├── api/v1/
│       │   └── __init__.py
│       ├── core/
│       │   └── rag_engine.py       # RAG pipeline (embed → retrieve → generate)
│       ├── models/
│       │   └── schemas.py          # Pydantic request/response models
│       ├── services/
│       │   └── document_service.py # PDF, TXT, DOCX, URL loaders
│       ├── config.py               # Settings via .env
│       └── main.py                 # FastAPI app + all endpoints
├── data/
│   ├── documents/                  # Place documents here for bulk ingestion
│   └── test_cases/
├── streamlit_app.py                # Streamlit UI
├── requirements.txt
├── .env                            # Environment config (never commit this)
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Git

### Step 1 — Clone the repo

```bash
git clone https://github.com/swapnilparate87/Enterprise-RAG-System.git
cd Enterprise-RAG-System
```

### Step 2 — Create virtual environment

```bash
python -m venv RAG_System
# Windows
RAG_System\Scripts\Activate
# Mac/Linux
source RAG_System/bin/activate
```

### Step 3 — Install dependencies

```bash
# Install in groups to avoid resolver timeout
pip install pydantic pydantic-settings fastapi uvicorn[standard] --no-cache-dir
pip install chromadb==1.5.5 --no-cache-dir
pip install langchain langchain-community langchain-ollama langchain-chroma langchain-text-splitters --no-cache-dir
pip install pypdf python-docx python-multipart beautifulsoup4 requests python-dotenv sqlalchemy httpx pytest --no-cache-dir
pip install sentence-transformers langchain-huggingface streamlit --no-cache-dir
```

### Step 4 — Pull Ollama models

```bash
ollama pull qwen2:1.5b        # Fastest (recommended to start)
ollama pull gemma2:2b         # Balanced
ollama pull mistral           # Best quality
ollama pull nomic-embed-text  # Embedding model (optional)
```

### Step 5 — Configure environment

Create a `.env` file in the root directory:

```env
OLLAMA_MODEL=qwen2:1.5b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_K=4
DEBUG=True
LOG_LEVEL=INFO
```

---

## Running the App

You need **two terminals** running simultaneously.

### Terminal 1 — Start the API

```bash
cd backend
uvicorn app.main:app --reload
```

API will be available at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

### Terminal 2 — Start the UI

```bash
# From project root
streamlit run streamlit_app.py
```

UI will open at: `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/upload` | Upload PDF/TXT/DOCX file |
| POST | `/api/v1/ingest-text` | Ingest raw text |
| POST | `/api/v1/ingest-url` | Scrape and ingest a webpage |
| POST | `/api/v1/query` | Ask a question |
| POST | `/api/v1/query-stream` | Ask with streaming response |
| POST | `/api/v1/switch-model` | Switch LLM at runtime |
| GET | `/api/v1/stats` | Vector store statistics |
| DELETE | `/api/v1/clear` | Clear all documents |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "k": 4}'
```

---

## Supported Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `qwen2:1.5b` | 934MB | ⚡⚡⚡ | ⭐⭐ | Fast Q&A |
| `gemma2:2b` | 1.6GB | ⚡⚡ | ⭐⭐⭐ | Balanced |
| `mistral:latest` | 4.4GB | ⚡ | ⭐⭐⭐⭐ | Best quality |
| `deepseek-r1:8b` | 5.2GB | ⚡ | ⭐⭐⭐⭐ | Reasoning |

Switch models at runtime from the Streamlit sidebar — no restart needed.

---

## Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2:1.5b` | Active LLM model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Vector store location |
| `CHUNK_SIZE` | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of chunks to retrieve per query |

---

## Troubleshooting

**`No module named 'app'`**
→ Make sure you run uvicorn from inside the `backend/` folder.

**`Ollama connection refused`**
→ Ollama is not running. Start it: `ollama serve`

**`Model not found`**
→ Pull the model first: `ollama pull <model-name>`

**Slow responses**
→ Switch to `qwen2:1.5b` in the sidebar. Ensure no other apps are using your GPU.

**`pydantic-core` build error**
→ Install packages in separate groups (see Installation step 3). Avoid pinning pydantic versions.

**Chroma import error on Python 3.14**
→ Use `chromadb==1.5.5` — older versions use `pydantic.v1` which breaks on Python 3.14.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

---

---

*Built with ❤️ using LangChain, Ollama, ChromaDB, FastAPI and Streamlit.*