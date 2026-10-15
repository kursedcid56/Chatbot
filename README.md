# Jarvis

Jarvis is an agentic AI chatbot application built with FastAPI, LangGraph, and Google Gemini. It supports multi-tool orchestration, persistent conversation memory, real-time SSE streaming, and document retrieval using a hybrid search pipeline (ChromaDB + BM25 with Reciprocal Rank Fusion).

[English](README.md) | [Tiếng Việt](README_vi.md)

## Quick Start

### Option 1: Docker (Recommended)
```bash
# 1. Configure environment variables
cp .env.example .env   # fill in your GOOGLE_API_KEY and TAVILY_API_KEY

# 2. Build and run container
docker build -t jarvis .
docker run -d -p 8080:8080 --env-file .env jarvis
```
Open `http://localhost:8080` in your browser.

### Option 2: Local Python Environment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env   # fill in your GOOGLE_API_KEY and TAVILY_API_KEY

# 3. Start the application
python app.py
```
Open `http://localhost:8080` in your browser.

## Demo

Demo video: [https://drive.google.com/file/d/1-bW6XUzkcUqfO1DndhW0Ava5fuyrm1o3/view?usp=sharing](https://drive.google.com/file/d/1-bW6XUzkcUqfO1DndhW0Ava5fuyrm1o3/view?usp=sharing)

## Features

- Multi-tool agent orchestration (Web Search via Tavily, Calculator, SQLite Memory) using LangGraph
- Hybrid Search RAG combining dense vector search (ChromaDB) and sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF)
- Dual-layer Input/Output Guardrails: prompt injection detection (English & Vietnamese), PII masking (phones, emails, credentials), and secret leak prevention at output
- IP-based rate limiting powered by slowapi to protect system resources and API quotas
- Real-time token streaming with Server-Sent Events (SSE) and live tool execution indicators
- Persistent multi-thread conversation history stored in SQLite
- Clean web interface with real-time tool execution status, document upload (PDF, DOCX, TXT, CSV), and image reasoning
- Docker containerization for local execution and GitHub Actions CI workflow

## Tech Stack

- Backend: FastAPI, Uvicorn
- Agent & LLM: LangGraph, LangChain, Google Gemini
- Guardrails & Security: Custom Security Pipeline, slowapi
- Retrieval & RAG: ChromaDB, BM25 (rank-bm25), PyPDF, docx2txt
- Tools & Storage: Tavily Search, SQLite
- DevOps & CI: Docker, GitHub Actions

## Installation Details

### Prerequisites

- Python 3.11+
- Git

### Detailed Setup

1. Clone the repository:
```bash
git clone https://github.com/kursedcid56/Chatbot.git
cd Chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory (or copy from `.env.example`):
```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
TAVILY_API_KEY=your_tavily_api_key
```

5. Run the application:
```bash
python app.py
```
Open `http://localhost:8080` in your browser.

## Running with Docker (Local)

```bash
docker build -t jarvis .
docker run -d -p 8080:8080 --env-file .env jarvis
```

## License

Apache 2.0