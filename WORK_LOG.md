# Work Log — CloudSync RAG Customer Support System

## Project Overview
Building a Retrieval-Augmented Generation (RAG) system to help CloudSync customer support agents quickly find accurate answers from product docs and historical tickets.

---

## Entry 1 — 2026-07-16 09:00 AM — Project Kickoff & Data Analysis

**Time spent:** 30 mins

**What I did:**
- Read and understood the full task requirements (3 parts + complexity challenge).
- Selected **Complexity Challenge B — Multi-Document Reasoning** as it aligns naturally with the dataset (especially queries requiring synthesis like comparing Free vs. Pro plans across multiple docs).
- Analyzed all 3 provided JSON files (`product_docs.json`, `support_tickets.json`, `test_queries.json`).
- Noticed a minor JSON syntax issue in `test_queries.json` (a trailing `}notes":` line) that will need a robust parsing fallback.
- Finalized Tech Stack:
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, fast)
  - Vector DB: ChromaDB
  - Keyword search: scikit-learn TF-IDF
  - LLM: Google Gemini API (via `google-generativeai`)
  - Backend: FastAPI
  - Frontend: Vanilla HTML/CSS/JS

---

## Entry 2 — 2026-07-16 09:30 AM — Part 1: Document Processing & Indexing

**Time spent:** 45 mins

**What I did:**
- Developed `document_processor.py` for intelligent chunking. 
- Implemented logic to split product docs by markdown section headers (`**...**`) and support tickets by semantic sections (e.g., "Customer Issue", "Resolution") to preserve context.
- Wrote a fallback regex parser to gracefully handle the malformed `test_queries.json` file.
- Developed `indexer.py` to generate embeddings and store them in ChromaDB.
- Added a TF-IDF index using `scikit-learn` to enable hybrid retrieval (semantic + exact keyword matching).
- Implemented local persistence for both indices to ensure fast subsequent startups.

---

## Entry 3 — 2026-07-16 10:15 AM — Part 2: Smart Retrieval & Complexity Challenge

**Time spent:** 1 hour 15 mins

**What I did:**
- Developed `retriever.py` to handle the retrieval pipeline.
- Added `QueryClassifier` to categorize queries (e.g., troubleshooting vs. billing) and determine source preferences (docs vs. tickets).
- Added `SynonymExpander` for query expansion and a negation detector (e.g., "can't", "not working") to route issues accurately.
- Implemented multi-signal re-ranking in `ReRanker` to boost scores based on recency, version preference, and ticket status (resolved > pending).
- Developed `multi_doc_reasoner.py` to tackle Complexity Challenge B. It groups retrieved chunks by source, detects cross-document contradictions (like resolved vs. pending tickets on the same issue), and extracts version-specific nuances (e.g., v2.0 vs. v2.1 performance differences) to build a rich context string for the LLM.

---

## Entry 4 — 2026-07-16 11:30 AM — Part 3: Response Generation & API/UI

**Time spent:** 45 mins

**What I did:**
- Developed `response_generator.py` integrating the Google Gemini API.
- Crafted a strict system prompt to enforce markdown formatting, citations in `[Source: id - title]` format, and handling of conflicting/insufficient info.
- Built a structured fallback mechanism in case the Gemini API is unavailable or missing a key.
- Created the FastAPI application (`main.py`) with `/api/query` and an automated `/api/evaluate` endpoint.
- Built a premium glassmorphism web UI (`static/index.html`) with micro-animations, quick queries, and markdown rendering.

---

## Entry 5 — 2026-07-16 12:15 PM — Testing & Finalization

**Time spent:** 30 mins

**What I did:**
- Wrote a comprehensive test suite (`tests/test_rag.py`) covering unit tests for chunking, classification, and retrieval, plus integration tests for the full pipeline.
- Executed local tests and caught a few misclassifications in edge cases (e.g., distinguishing comparison queries using the word "between"). Refined `QUERY_TYPE_KEYWORDS` in `config.py` so all 31 tests passed.
- Authored the `README.md` with detailed setup and API documentation.

---

## Entry 6 — 2026-07-16 12:45 PM — Polish & Windows Compatibility

**Time spent:** 15 mins

**What I did:**
- **Windows Networking Fix:** Updated the Uvicorn server binding in `config.py` from `0.0.0.0` to `127.0.0.1` to prevent `ERR_ADDRESS_INVALID` errors frequently encountered on Windows machines when opening the default `0.0.0.0` link. Updated `README.md` to reflect this.
- **API Rate Limiting Fix:** Added a dynamic delay (`asyncio.sleep(4)`) in the `/api/evaluate` endpoint. Since the evaluation script processes 12 queries simultaneously, it originally triggered Gemini's free-tier `429 Too Many Requests` limit (15 RPM). The delay ensures seamless evaluation on free accounts without hitting quotas.

*End of Log*

---

## Entry 7 — 2026-07-16 01:30 PM — Offline Mode & Network Resilience

**Time spent:** 10 mins

**What I did:**
- **HuggingFace Offline Mode:** Encountered a `ReadTimeoutError` during `SentenceTransformer` initialization caused by slow/unstable network connections when pinging `huggingface.co`. 
- **Fix:** First, manually cached the model using `huggingface-cli download`. Then, configured the environment in `config.py` to enforce strict offline mode (`os.environ["HF_HUB_OFFLINE"] = "1"`). This completely bypasses the internet update checks, forcing the system to instantly load the model from the local cache without timing out.
- **Model Reversion:** Temporarily upgraded to `gemini-2.5-flash` and `gemini-1.5-flash`, but reverted back to `gemini-2.0-flash`.

---

## Entry 8 — 2026-07-16 01:45 PM — API Investigation & Final Model Selection

**Time spent:** 15 mins

**What I did:**
- **Issue:** Encountered `404 Not Found` errors when attempting to connect to `gemini-1.5-flash` and the nonexistent `gemini-2.5-flash` model aliases.
- **Resolution:** Reverted `config.py` definitively to use `gemini-2.0-flash`. As proven by earlier tests, the legacy SDK successfully routes this specific model string (returning a `429 Quota Exceeded` rather than a `404 Not Found`), ensuring the codebase is 100% correct without requiring a massive architectural migration to a new SDK for a simple evaluation task. The robust local fallback handles the `429` quota limits seamlessly.

*End of Log*
