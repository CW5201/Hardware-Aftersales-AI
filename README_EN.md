
# Hardware-Aftersales-AI

[中文](README.md) | [English](README_EN.md)

> **Production-grade, Multi-route Retrieval & Reranking Graph-driven RAG Knowledge Base Engine**
> 
> Designed for **AI Engineer / LLM Application Development / Production RAG** learning and deployment

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Keywords**: `LangGraph Workflow` `BGE-M3 Hybrid Search` `RRF Fusion` `HyDE Hypothetical Document` `Rerank Truncation` `MinerU PDF Parsing` `Enterprise Knowledge Base` `Intelligent QA System` `SSE Streaming`

## ✨ Key Features

- 🔄 **LangGraph Workflow Orchestration**: Highly extensible DAG-based import/QA Agent workflow
- 🎯 **Entity Recognition Pre-alignment**: Auto-identify and confirm product names before retrieval for precision
- 🔍 **Three-route Parallel Retrieval**:
  - **BGE-M3 Hybrid Search**: Dense + Sparse vector retrieval
  - **HyDE (Hypothetical Document Embeddings)**: LLM generates hypothetical answers for semantic matching
  - **MCP WebSearch**: Auto-supplement with web search when local knowledge base has no matches
- 📊 **Advanced Reranking & Cliff Truncation**: RRF fusion + Qwen3-Rerank cross-encoder, with score gap smart truncation
- 📄 **Deep PDF Parsing**: MinerU integration for precise table, formula, and Markdown image extraction
- 🚀 **SSE Streaming Response**: Real-time progress push and streaming answer output

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend UI                              │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │  import.html        │       │  chat.html          │          │
│  │  (Document Import)  │       │  (Chat QA)          │          │
│  └─────────────────────┘       └─────────────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│  ┌─────────────────────────┐     ┌─────────────────────────┐    │
│  │   import_service        │     │   query_service         │    │
│  │   (port 8000)           │     │   (port 8001)           │    │
│  │   Document Import API   │     │   Query QA API          │    │
│  └─────────────────────────┘     └─────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                    Processing Layer (LangGraph)                  │
│                                                                 │
│  ┌────────────────────────┐     ┌────────────────────────┐      │
│  │    Import Workflow      │     │    Query Workflow       │      │
│  │                        │     │                        │      │
│  │  1. File Type Detection│     │  1. Product Name Confirm│      │
│  │  2. PDF→MD Conversion  │     │     ↓                  │      │
│  │  3. Image Processing   │     │  ┌────┬────┬────┐      │      │
│  │  4. Document Splitting │     │  │Vec │HyDE│Web │      │      │
│  │  5. Product Name Recog │     │  └──┬─┴──┬─┴──┬─┘      │      │
│  │  6. BGE-M3 Embedding   │     │     └──┬─┘   │        │      │
│  │  7. Milvus Storage     │     │     RRF Fusion│        │      │
│  │                        │     │     Rerank    │        │      │
│  │                        │     │     LLM Answer│        │      │
│  └────────────────────────┘     └────────────────────────┘      │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                       Infrastructure                             │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Milvus  │ │ MongoDB  │ │  MinIO   │ │  BGE-M3  │           │
│  │ (Vector) │ │ (History)│ │ (Object) │ │(Embedding)│           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐                               │
│  │Qwen3-Rerank  │ │  DashScope   │                               │
│  │  (Reranker)  │ │    (LLM)     │                               │
│  └──────────────┘ └──────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | FastAPI + Uvicorn | Dual-service architecture (Import/Query) |
| Frontend | HTML/CSS/JS | SSE streaming chat interface |
| Workflow | LangGraph | DAG workflow orchestration |
| LLM | DashScope (Qwen) | QA generation, document understanding |
| Embedding | BGE-M3 (Local) | Dense + Sparse vectors |
| Vector DB | Milvus | Document chunk retrieval |
| Database | MongoDB | Chat history storage |
| Object Storage | MinIO | PDF/image storage |
| PDF Parsing | MinerU | PDF→Markdown conversion |
| Package Manager | UV | Dependency management + CUDA support |

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/CW5201/Hardware-Aftersales-AI.git
cd Hardware-Aftersales-AI

# 2. Start all services
docker-compose up -d

# 3. Access the UI
# Document Import: http://localhost:8000/import.html
# Chat QA: http://localhost:8001/chat.html
```

### Option 2: Manual Installation

```bash
# 1. Install dependencies
uv sync

# 2. Download models
uv run python tool/download_bgem3.py
uv run python tool/download_bge_reranker_large.py

# 3. Start services
# Import service (port 8000)
uv run python -m web.api.import_service

# Query service (port 8001)
uv run python -m web.api.query_service
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# DashScope API
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Model Paths
BGE_M3_PATH=/path/to/bge-m3

# Databases
MILVUS_URL=http://localhost:19530
MONGO_URL=mongodb://localhost:27017
MINIO_ENDPOINT=localhost:9000
```

## 📊 Query Workflow

### 1. Product Name Confirmation (node_item_name_confirm)

- Fetch chat history from MongoDB
- LLM extracts product names and rewrites query
- BGE-M3 vectorization + Milvus retrieval
- Three result branches:
  - **A. Confirmed Match** → Continue to multi-route retrieval
  - **B. Candidate Match** → Ask user to confirm
  - **C. No Match** → Return "not found" message

### 2. Multi-route Parallel Retrieval

| Route | Description |
|-------|-------------|
| Vector Search | BGE-M3 hybrid search (Dense+Sparse) |
| HyDE Search | LLM generates hypothetical document, then vector search |
| Web Search | MCP calls Alibaba Bailian WebSearch |

### 3. Result Fusion & Reranking

- **RRF Fusion**: k=60 score smoothing, merge multi-route results
- **Rerank**: Qwen3-Rerank cross-encoder scoring
- **Cliff Detection**: Absolute gap 0.3 / Relative gap 0.25 truncation
- **Dynamic Top-K**: Select 2-5 documents based on score distribution

### 4. Answer Generation (node_answer_output)

- Concatenate reranked documents as context
- LLM generates summary answer
- Extract image URLs from documents
- Extract reference sources
- Support SSE streaming output

## 📁 Project Structure

```
Hardware-Aftersales-AI/
├── .env                    # Environment variables
├── pyproject.toml          # Project dependencies
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Container image
│
├── config/                 # Configuration modules
│   ├── lm_config.py        # LLM API config
│   ├── embedding_config.py # BGE-M3 config
│   ├── milvus_config.py    # Milvus config
│   ├── minio_config.py     # MinIO config
│   ├── reranker_config.py  # Rerank config
│   └── bailian_mcp_config.py # Web search MCP config
│
├── processor/              # Core processing logic
│   ├── import_processor/   # Document import workflow
│   │   ├── main_graph.py   # Import LangGraph
│   │   └── nodes/          # Import nodes
│   │
│   └── query_processor/    # Query QA workflow
│       ├── main_graph.py   # Query LangGraph
│       └── nodes/          # Query nodes
│
├── utils/                  # Utility modules
│   ├── embedding_utils.py  # BGE-M3 embedding
│   ├── milvus_utils.py     # Milvus operations
│   ├── minio_utils.py      # MinIO operations
│   ├── llm_utils.py        # LLM client
│   ├── reranker_http_utils.py # Rerank API
│   ├── mongo_history_utils.py # MongoDB history
│   ├── sse_utils.py        # SSE streaming tools
│   └── task_utils.py       # Task progress tracking
│
├── web/                    # Web layer
│   ├── api/
│   │   ├── import_service.py  # Import service (port 8000)
│   │   └── query_service.py   # Query service (port 8001)
│   └── page/
│       ├── import.html    # Document import UI
│       └── chat.html      # Chat UI
│
└── tool/                   # Development tools
    └── logger.py           # Logging configuration
```

## 🔧 API Reference

### Import Service (port 8000)

| Method | Path | Description |
|--------|------|-------------|
| POST | /upload | Upload document |
| GET | /status/{task_id} | Query task status |
| GET | /health | Health check |

### Query Service (port 8001)

| Method | Path | Description |
|--------|------|-------------|
| POST | /query | Query QA |
| GET | /stream/{session_id} | SSE streaming output |
| GET | /history/{session_id} | Get chat history |
| DELETE | /history/{session_id} | Clear chat history |
| GET | /sessions | Get all sessions |
| DELETE | /sessions/{session_id} | Delete session |
| GET | /health | Health check |

## 🎯 Design Patterns

- **LangGraph State Graph**: Import/query as DAG workflows with TypedDict state
- **BaseNode Abstract Class**: Unified entry, logging, progress reporting
- **Lazy Singleton**: Milvus/BGE-M3/MinIO/MongoDB global singletons
- **In-memory Task Tracking**: Dict stores node runtime status, frontend polling
- **Idempotent Write**: Dedup by file_title, repeated imports don't create duplicates
- **Multi-source Fusion**: Vector+HyDE+Web three-route retrieval, RRF fusion + Rerank
- **SSE Streaming**: In-memory Queue + async generator, real-time progress and answer push

## 🤝 Contributing

Issues and Pull Requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration framework
- [Milvus](https://milvus.io) - Vector database
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Hybrid retrieval model
- [MinerU](https://github.com/opendatalab/MinerU) - PDF parsing tool
- [DashScope](https://dashscope.aliyuncs.com) - LLM API service

---

## 🏷️ GitHub Topics (Add to repository right side About -> Topics)

```
ai-engineer llm-application agentic-rag advanced-rag knowledge-base
langgraph milvus bge-m3 qwen fastapi python
rag langchain document-parsing sse-streaming
```

If this project helps you, please give it a ⭐ Star!
