# Local Document Q&A Chatbot (RAG Pipeline)

A lightweight Retrieval-Augmented Generation (RAG) web app where users upload PDF documents and ask questions strictly grounded in the text, complete with page-level source citations.

## Architecture
`PDF Upload` ➔ `Overlapping Text Chunking` ➔ `Gemini Embeddings` ➔ `FAISS Cosine Similarity Index` ➔ `Grounded Prompt Generation`

## Tech Stack
- **Frontend:** Streamlit
- **LLM & Embeddings:** Google GenAI SDK (`gemini-3.5-flash-lite` & `gemini-embedding-001`)
- **Vector Search:** FAISS (`IndexFlatIP` with L2 normalization)
- **PDF Parsing:** PyPDF

## Setup & Run
1. Use VPN if you are not available regions for Google AI Studio and Gemini API

2. Clone the repository and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt

