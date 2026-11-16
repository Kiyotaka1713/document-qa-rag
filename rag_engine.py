import os
import httpx2 as httpx
import faiss
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
# Use a fresh connection per request (limits=max_keepalive_connections=0) and a 120s timeout
http_client = httpx.Client(
    timeout=120.0,
    limits=httpx.Limits(max_keepalive_connections=0),
)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(httpx_client=http_client),
)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash-lite"


def extract_and_chunk_pdf(pdf_file, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Reads a PDF and splits text into overlapping chunks with page metadata."""
    reader = PdfReader(pdf_file)
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # Normalize whitespace
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page_num})
            start += chunk_size - overlap

    return chunks


def get_embeddings(texts: list[str], task_type: str) -> np.ndarray:
    """Generates L2-normalized embeddings for cosine similarity via Inner Product."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_vector_index(chunks: list[dict]):
    """Embeds document chunks and builds an in-memory FAISS index."""
    texts = [c["text"] for c in chunks]
    # Batch in groups of 50 to respect API payload limits
    all_vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        all_vectors.append(get_embeddings(batch, task_type="RETRIEVAL_DOCUMENT"))

    matrix = np.vstack(all_vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])  # Inner Product on normalized vectors = Cosine Similarity
    index.add(matrix)
    return index


def answer_question(question: str, index, chunks: list[dict], top_k: int = 4) -> tuple[str, list[dict]]:
    """Retrieves top-k chunks and generates a strictly grounded answer."""
    q_vec = get_embeddings([question], task_type="RETRIEVAL_QUERY")
    scores, indices = index.search(q_vec, top_k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            retrieved.append({**chunks[idx], "score": float(score)})

    context_block = "\n\n".join(
        [f"[Page {item['page']}]: {item['text']}" for item in retrieved]
    )

    system_prompt = (
        "You are a precise document assistant. Answer the user's question using ONLY "
        "the provided context excerpts. Cite page numbers like (Page X) in your answer. "
        "If the answer is not in the context, say 'I cannot find that information in the uploaded document.'"
    )

    prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    return response.text, retrieved