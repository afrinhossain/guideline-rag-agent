
# Guideline RAG Agent (Embeddings + Retrieval API)

This project implements the **retrieval layer** of RAG from scratch to build a solid understanding of:
- semantic similarity,
- embedding-based search,
- API-driven ML systems.

---

## What’s inside

- **FastAPI** defines the API endpoints and request handling.
- **Uvicorn** runs the application as a high-performance ASGI web server.
- **Sentence-Transformers** generates embeddings locally (no API keys).
- **NumPy** is used for cosine similarity computation.

---

## Endpoints

### `POST /embed`
Returns the embedding vector length and a small preview of the vector.

Example request:
```json
{ "text": "hello world" }
```
Response:
```json
{
  "vector_length": 384,
  "first_5_values": [0.02, -0.11, 0.33, 0.04, -0.07]
}

### `POST /search`
Performs semantic retrieval over a small in-memory document set.

Example request:
```json
{
  "text": "I have a headache and light bothers me"
}

Response:
```json
{
  "query": "I have a headache and light bothers me",
  "best_doc_id": "doc2",
  "best_doc_text": "Migraine is often a severe headache with nausea and sensitivity to light.",
  "score": 0.87
}

## To run the project

### Install dependencies

```bash
{
    pip install fastapi uvicorn sentence-transformers numpy
    }

### Start the FastAPI server


```bash
{
    uvicorn app:app --reload
    }

### Start the FastAPI server
