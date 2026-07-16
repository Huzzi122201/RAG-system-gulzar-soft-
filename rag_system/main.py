"""
FastAPI Application — Main entry point for the RAG Customer Support System.

Endpoints:
  POST /api/query     — Submit a customer query, get a generated response
  GET  /api/health    — Health check / system status
  POST /api/evaluate  — Run all test queries and return evaluation metrics
  GET  /              — Serve the web UI

Startup:
  On first launch, processes documents, builds indices, and initializes
  the retrieval pipeline. Subsequent launches load from cache.
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import config
from document_processor import DocumentProcessor, load_test_queries
from indexer import Indexer
from retriever import SmartRetriever
from response_generator import ResponseGenerator

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "rag_system.log"),
            encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

# Global pipeline components (initialized on startup)
indexer: Optional[Indexer] = None
retriever: Optional[SmartRetriever] = None
generator: Optional[ResponseGenerator] = None


def initialize_pipeline():
    """Initialize the full RAG pipeline."""
    global indexer, retriever, generator

    logger.info("=" * 60)
    logger.info("Initializing RAG Pipeline...")
    logger.info("=" * 60)

    start_time = time.time()

    logger.info("Step 1/3: Processing documents...")
    processor = DocumentProcessor(
        product_docs_path=config.PRODUCT_DOCS_PATH,
        support_tickets_path=config.SUPPORT_TICKETS_PATH,
    )
    chunks = processor.process_all()

    logger.info("Step 2/3: Building indices (this may take a moment on first run)...")
    indexer = Indexer()
    indexer.initialize(chunks, force_rebuild=True)

    logger.info("Step 3/3: Initializing retriever and response generator...")
    retriever = SmartRetriever(indexer)
    generator = ResponseGenerator()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline initialized in {elapsed:.1f}s")
    logger.info(f"  - Chunks indexed: {len(chunks)}")
    logger.info(f"  - Gemini LLM: {'available' if generator.model else 'UNAVAILABLE (set GEMINI_API_KEY in .env)'}")
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    initialize_pipeline()
    yield
    logger.info("Shutting down RAG system")


# FastAPI App
app = FastAPI(
    title="CloudSync RAG Customer Support System",
    description="AI-powered customer support using RAG with hybrid retrieval and Gemini LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (web UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    sources: list
    query_type: str
    has_conflicts: bool
    version_warnings: list
    source_count: int
    is_multi_source: bool
    processing_time_ms: float


# Endpoints

@app.get("/")
async def serve_ui():
    """Serve the web UI."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"message": "Web UI not found. Use /api/query for API access."},
        status_code=404,
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint with system status."""
    return {
        "status": "healthy",
        "components": {
            "indexer": indexer is not None,
            "retriever": retriever is not None,
            "generator": generator is not None,
            "gemini_llm": generator.model is not None if generator else False,
        },
        "chunks_indexed": len(indexer.chunks) if indexer else 0,
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a customer support query through the full RAG pipeline.
    
    1. Classify and expand the query
    2. Retrieve relevant chunks (hybrid: semantic + keyword)
    3. Re-rank results
    4. Generate response with citations via Gemini
    """
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    try:
        # Retrieve relevant documents
        retrieval_result = retriever.retrieve(
            request.query, top_k=request.top_k
        )

        # Generate response
        response = generator.generate(
            query=request.query,
            results=retrieval_result["results"],
            query_info=retrieval_result["query_info"],
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        logger.info(
            f"Query processed in {processing_time:.0f}ms: "
            f"'{request.query[:50]}...' -> {response['confidence']} confidence, "
            f"{response['source_count']} sources"
        )

        return QueryResponse(
            answer=response["answer"],
            confidence=response["confidence"],
            sources=response["sources"],
            query_type=response["query_type"],
            has_conflicts=response["has_conflicts"],
            version_warnings=response["version_warnings"],
            source_count=response["source_count"],
            is_multi_source=response["is_multi_source"],
            processing_time_ms=round(processing_time, 1),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
async def evaluate_system():
    """
    Run all test queries and return evaluation metrics.
    
    Compares retrieval results against expected sources and checks
    that expected keywords appear in generated responses.
    """
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Load test queries
    test_queries = load_test_queries(config.TEST_QUERIES_PATH)
    if not test_queries:
        raise HTTPException(status_code=500, detail="Could not load test queries")

    results = []
    total_precision = 0
    total_recall = 0
    total_keyword_hits = 0
    total_keywords = 0

    for tq in test_queries:
        query = tq.get("query", "")
        expected_sources = tq.get("expected_sources", [])
        expected_keywords = tq.get("expected_answer_contains", [])
        query_id = tq.get("id", "unknown")

        try:
            # Run retrieval
            retrieval_result = retriever.retrieve(query)
            retrieved_sources = list(set(
                chunk.source_id for chunk, _ in retrieval_result["results"]
            ))

            # Generate response
            response = generator.generate(
                query=query,
                results=retrieval_result["results"],
                query_info=retrieval_result["query_info"],
            )

            # Evaluate retrieval: precision and recall
            if expected_sources:
                true_positives = len(set(retrieved_sources) & set(expected_sources))
                precision = true_positives / len(retrieved_sources) if retrieved_sources else 0
                recall = true_positives / len(expected_sources) if expected_sources else 0
            else:
                precision = 1.0
                recall = 1.0

            # Evaluate response: keyword presence
            answer_lower = response["answer"].lower()
            keyword_hits = sum(
                1 for kw in expected_keywords if kw.lower() in answer_lower
            )
            keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

            total_precision += precision
            total_recall += recall
            total_keyword_hits += keyword_hits
            total_keywords += len(expected_keywords)

            results.append({
                "query_id": query_id,
                "query": query,
                "difficulty": tq.get("difficulty", ""),
                "query_type": tq.get("query_type", ""),
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "expected_keywords": expected_keywords,
                "keyword_hits": keyword_hits,
                "keyword_coverage": round(keyword_coverage, 3),
                "confidence": response["confidence"],
                "answer_preview": response["answer"][:200] + "...",
                "source_count": response["source_count"],
            })

        except Exception as e:
            logger.error(f"Error evaluating {query_id}: {e}")
            results.append({
                "query_id": query_id,
                "query": query,
                "error": str(e),
            })

        # Add delay to avoid free-tier rate limits
        # gemini-2.5-flash has a strict limit of 5 RPM on the free tier
        if generator.model:
            await asyncio.sleep(15)

    # Aggregate metrics
    n = len([r for r in results if "error" not in r])
    avg_precision = total_precision / n if n > 0 else 0
    avg_recall = total_recall / n if n > 0 else 0
    avg_keyword_coverage = total_keyword_hits / total_keywords if total_keywords > 0 else 0

    output_data = {
        "summary": {
            "total_queries": len(test_queries),
            "successful": n,
            "avg_retrieval_precision": round(avg_precision, 3),
            "avg_retrieval_recall": round(avg_recall, 3),
            "avg_keyword_coverage": round(avg_keyword_coverage, 3),
            "f1_score": round(
                2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0 else 0, 3
            ),
        },
        "results": results,
    }

    # Save to response.json in the project root
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "response.json"))
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save response.json: {e}")

    return output_data


# Entry Point
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("  CloudSync RAG Customer Support System")
    print("=" * 60)
    print(f"  API:  http://localhost:{config.SERVER_PORT}/api/query")
    print(f"  UI:   http://localhost:{config.SERVER_PORT}/")
    print(f"  Docs: http://localhost:{config.SERVER_PORT}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
        log_level="info",
    )
