# CloudSync Support AI (RAG System)

An AI-powered customer support assistant built using **Retrieval-Augmented Generation (RAG)**. It helps support agents quickly answer customer queries by retrieving relevant information from product documentation and historical support tickets, and synthesizing comprehensive answers.

## Features

- **Hybrid Retrieval:** Combines semantic search (ChromaDB + sentence-transformers) with keyword search (TF-IDF) for high recall.
- **Complexity Challenge B (Multi-Document Reasoning):** Synthesizes information across multiple documents (e.g., comparing versions or plan features) and detects contradictions/version mismatches.
- **Smart Query Classification:** Detects troubleshooting, billing, comparison, and factual queries to prioritize sources dynamically.
- **Synonym & Negation Handling:** Expands queries with synonyms and detects negative intent ("can't", "not working") to route to troubleshooting docs.
- **Citations & Confidence:** Every generated response includes a confidence score and citations back to the source documents.
- **Premium Web UI:** Dark mode glassmorphism interface with markdown rendering and response metadata.

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- **Vector Store:** ChromaDB (persistent)
- **Keyword Search:** scikit-learn (TF-IDF)
- **LLM / Generation:** Google Gemini API (`gemini-2.0-flash`)
- **Frontend:** Vanilla HTML/CSS/JS

## Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   Open the `.env` file and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
   *(Get a free key from [Google AI Studio](https://aistudio.google.com/))*

3. **Run the Server:**
   ```bash
   python rag_system/main.py
   ```
   *Note: On the first run, the system will download the embedding model (~80MB) and build the indices. This may take 15-30 seconds. Subsequent starts will be nearly instant.*

4. **Access the App:**
   - Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000) (or `http://localhost:8000`)
   - API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Documentation

### `POST /api/query`
Main endpoint for processing customer queries.

**Request:**
```json
{
  "query": "My files aren't syncing",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "To fix sync issues... [Source: doc_002 - Troubleshooting]",
  "confidence": "HIGH",
  "sources": [
    {
      "source_id": "doc_002",
      "title": "Troubleshooting Sync Issues",
      "type": "product_doc",
      "version": "v2.1",
      "last_updated": "2024-01-20",
      "relevance_score": 0.85
    }
  ],
  "query_type": "troubleshooting",
  "has_conflicts": false,
  "version_warnings": [],
  "source_count": 2,
  "is_multi_source": true,
  "processing_time_ms": 1250.5
}
```

### `POST /api/evaluate`
Runs the 12 predefined test queries and returns precision, recall, and keyword coverage metrics.

## Evaluation

You can run the evaluation suite directly from the Web UI by clicking the **"📊 Run Evaluation"** button in the bottom right corner, or by calling the `/api/evaluate` endpoint.

To run the unit and integration tests:
```bash
pytest tests/test_rag.py -v
```

## Architecture

1. **Document Processor:** Chunks JSON docs intelligently (by markdown sections for docs, semantic sections for tickets).
2. **Indexer:** Embeds chunks using `sentence-transformers` into ChromaDB and builds a TF-IDF index.
3. **Retriever:** Classifies query → expands synonyms → searches ChromaDB & TF-IDF → deduplicates → re-ranks by recency/relevance.
4. **Multi-Doc Reasoner:** Groups retrieved chunks by source, detects cross-document version conflicts or contradictions.
5. **Response Generator:** Constructs a rich prompt with the retrieved context and uses Gemini to generate a professional, cited answer.
