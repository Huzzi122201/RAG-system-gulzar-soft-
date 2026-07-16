"""
Unit and Integration Tests for the RAG Customer Support System.

Tests cover:
  - Document processing and chunking
  - Query classification
  - Synonym expansion
  - Negation detection
  - Hybrid retrieval (requires built indices)
  - Response generation structure
  - End-to-end query pipeline
"""

import os
import sys
import json
import pytest

# Ensure rag_system is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_processor import DocumentProcessor, DocumentChunk, load_test_queries
from retriever import QueryClassifier, SynonymExpander, ReRanker
import config


# Document Processor Tests

class TestDocumentProcessor:
    """Tests for document loading and chunking."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(
            product_docs_path=config.PRODUCT_DOCS_PATH,
            support_tickets_path=config.SUPPORT_TICKETS_PATH,
        )

    def test_load_product_docs(self, processor):
        """Product docs should load successfully."""
        docs = processor._load_json(config.PRODUCT_DOCS_PATH, "product_docs")
        assert len(docs) == 7, f"Expected 7 product docs, got {len(docs)}"

    def test_load_support_tickets(self, processor):
        """Support tickets should load successfully."""
        tickets = processor._load_json(config.SUPPORT_TICKETS_PATH, "support_tickets")
        assert len(tickets) == 8, f"Expected 8 tickets, got {len(tickets)}"

    def test_process_all_returns_chunks(self, processor):
        """process_all() should return a non-empty list of chunks."""
        chunks = processor.process_all()
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_has_required_fields(self, processor):
        """Every chunk should have all required metadata."""
        chunks = processor.process_all()
        for chunk in chunks:
            assert chunk.chunk_id, f"Chunk missing chunk_id"
            assert chunk.source_id, f"Chunk missing source_id"
            assert chunk.source_type in ("product_doc", "support_ticket")
            assert chunk.title, f"Chunk {chunk.chunk_id} missing title"
            assert chunk.content, f"Chunk {chunk.chunk_id} has empty content"
            assert len(chunk.content) >= DocumentProcessor.MIN_CHUNK_SIZE or \
                   chunk.total_chunks == 1  # Single-chunk docs can be small

    def test_chunk_ids_unique(self, processor):
        """All chunk IDs should be unique."""
        chunks = processor.process_all()
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"

    def test_product_doc_chunking(self, processor):
        """Product docs with sections should produce multiple chunks."""
        chunks = processor.process_all()
        doc_002_chunks = [c for c in chunks if c.source_id == "doc_002"]
        # doc_002 (Troubleshooting) has clear sections: Basic Checks, Advanced, Contact Support
        assert len(doc_002_chunks) >= 2, \
            f"Expected doc_002 to have multiple chunks, got {len(doc_002_chunks)}"

    def test_support_ticket_chunking(self, processor):
        """Support tickets should be split into semantic sections."""
        chunks = processor.process_all()
        ticket_001_chunks = [c for c in chunks if c.source_id == "ticket_001"]
        # ticket_001 has: Customer Issue, Troubleshooting Steps, Resolution, Follow-up
        assert len(ticket_001_chunks) >= 2, \
            f"Expected ticket_001 to have multiple chunks, got {len(ticket_001_chunks)}"

    def test_metadata_preservation(self, processor):
        """Chunks should preserve parent document metadata."""
        chunks = processor.process_all()
        doc_001_chunks = [c for c in chunks if c.source_id == "doc_001"]
        for chunk in doc_001_chunks:
            assert chunk.version == "v2.1"
            assert chunk.last_updated == "2024-01-15"
            assert "setup" in chunk.tags


# Query Classifier Tests

class TestQueryClassifier:
    """Tests for query classification."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_troubleshooting_classification(self, classifier):
        """Queries with problem indicators should classify as troubleshooting."""
        result = classifier.classify("My files aren't syncing between devices")
        assert result["query_type"] in ("troubleshooting", "performance"), \
            f"Expected troubleshooting type, got {result['query_type']}"

    def test_billing_classification(self, classifier):
        """Billing-related queries should classify correctly."""
        result = classifier.classify("I was charged for Pro but don't remember upgrading")
        assert result["query_type"] == "billing_issue"

    def test_feature_usage_classification(self, classifier):
        """Feature usage queries should classify correctly."""
        result = classifier.classify("How do I share folders with other people?")
        assert result["query_type"] == "feature_usage"

    def test_comparison_classification(self, classifier):
        """Comparison queries should be classified and flagged as multi-doc."""
        result = classifier.classify(
            "What are the differences between Free and Pro accounts?"
        )
        assert result["query_type"] == "comparison"
        assert result["is_multi_doc"] is True

    def test_negation_detection(self, classifier):
        """Queries with negation should be detected."""
        result = classifier.classify("Can't login after password reset")
        assert result["has_negation"] is True

    def test_no_negation(self, classifier):
        """Positive queries should not trigger negation."""
        result = classifier.classify("How do I create an account?")
        assert result["has_negation"] is False

    def test_developer_classification(self, classifier):
        """API/developer queries should classify correctly."""
        result = classifier.classify(
            "Can I use CloudSync API to integrate with my application?"
        )
        assert result["query_type"] == "developer"

    def test_security_classification(self, classifier):
        """Security queries should classify correctly."""
        result = classifier.classify("How secure is my data in CloudSync?")
        assert result["query_type"] == "security"


# Synonym Expander Tests

class TestSynonymExpander:
    """Tests for synonym expansion."""

    @pytest.fixture
    def expander(self):
        return SynonymExpander()

    def test_slow_expansion(self, expander):
        """'slow' should expand with performance-related synonyms."""
        expanded = expander.expand("CloudSync is running very slowly")
        assert "performance" in expanded.lower() or "speed" in expanded.lower()

    def test_login_expansion(self, expander):
        """'login' should expand with authentication synonyms."""
        expanded = expander.expand("I can't login to my account")
        assert "sign in" in expanded.lower() or "authenticate" in expanded.lower()

    def test_no_expansion_needed(self, expander):
        """Queries without synonym matches should remain unchanged."""
        query = "What is the meaning of life?"
        expanded = expander.expand(query)
        assert expanded == query

    def test_crash_expansion(self, expander):
        """'crash' should expand with related terms."""
        expanded = expander.expand("The mobile app keeps crashing")
        assert len(expanded) > len("The mobile app keeps crashing")


# Re-Ranker Tests

class TestReRanker:
    """Tests for the re-ranking system."""

    @pytest.fixture
    def reranker(self):
        return ReRanker()

    def _make_chunk(self, **kwargs):
        """Helper to create a test chunk."""
        defaults = {
            "chunk_id": "test_chunk_0",
            "source_id": "test_001",
            "source_type": "product_doc",
            "title": "Test Doc",
            "section": "test",
            "content": "Test content",
            "version": "v2.1",
            "last_updated": "2024-01-15",
            "tags": ["test"],
            "chunk_index": 0,
            "total_chunks": 1,
        }
        defaults.update(kwargs)
        return DocumentChunk(**defaults)

    def test_recency_boost(self, reranker):
        """More recent documents should get higher boost."""
        recent = reranker._compute_recency_boost("2024-02-01")
        older = reranker._compute_recency_boost("2023-06-01")
        assert recent > older, "Recent docs should have higher recency boost"

    def test_version_boost(self, reranker):
        """Newer versions should get a small boost."""
        v21 = reranker._compute_version_boost("v2.1")
        v20 = reranker._compute_version_boost("v2.0")
        assert v21 >= v20, "v2.1 should have equal or higher boost than v2.0"

    def test_resolved_status_boost(self, reranker):
        """Resolved tickets should score higher than pending."""
        resolved = reranker._compute_status_boost("resolved")
        pending = reranker._compute_status_boost("pending")
        assert resolved > pending

    def test_rerank_preserves_results(self, reranker):
        """Re-ranking should not lose any results."""
        chunk = self._make_chunk()
        results = [(chunk, 0.5)]
        query_info = {"source_preference": "both"}
        reranked = reranker.rerank(results, query_info)
        assert len(reranked) == 1


# Test Query Loading

class TestTestQueries:
    """Tests for loading and parsing test queries."""

    def test_load_test_queries(self):
        """Should successfully load test queries despite JSON formatting issues."""
        queries = load_test_queries(config.TEST_QUERIES_PATH)
        assert len(queries) >= 10, f"Expected at least 10 test queries, got {len(queries)}"

    def test_query_structure(self):
        """Each test query should have required fields."""
        queries = load_test_queries(config.TEST_QUERIES_PATH)
        for q in queries:
            assert "id" in q, f"Query missing 'id' field"
            assert "query" in q, f"Query {q.get('id')} missing 'query' field"


# Integration Test (requires full pipeline)

class TestIntegration:
    """Integration tests that run the full pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Initialize the full RAG pipeline once for all integration tests."""
        from indexer import Indexer
        from retriever import SmartRetriever

        processor = DocumentProcessor(
            product_docs_path=config.PRODUCT_DOCS_PATH,
            support_tickets_path=config.SUPPORT_TICKETS_PATH,
        )
        chunks = processor.process_all()

        indexer = Indexer()
        indexer.initialize(chunks, force_rebuild=True)

        retriever = SmartRetriever(indexer)
        return retriever

    def test_factual_retrieval(self, pipeline):
        """Simple factual query should retrieve the correct document."""
        result = pipeline.retrieve("How do I create a CloudSync account?")
        source_ids = [chunk.source_id for chunk, _ in result["results"]]
        assert "doc_001" in source_ids, \
            f"Expected doc_001 in results, got {source_ids}"

    def test_troubleshooting_retrieval(self, pipeline):
        """Troubleshooting query should retrieve relevant docs and tickets."""
        result = pipeline.retrieve("My files aren't syncing between devices")
        source_ids = [chunk.source_id for chunk, _ in result["results"]]
        assert "doc_002" in source_ids or "ticket_002" in source_ids, \
            f"Expected doc_002 or ticket_002 in results, got {source_ids}"

    def test_multi_doc_retrieval(self, pipeline):
        """Comparison query should retrieve from multiple documents."""
        result = pipeline.retrieve(
            "What are the differences between Free and Pro accounts?"
        )
        source_ids = set(chunk.source_id for chunk, _ in result["results"])
        assert len(source_ids) >= 2, \
            f"Expected multiple sources for comparison query, got {source_ids}"

    def test_billing_retrieval(self, pipeline):
        """Billing query should find billing docs/tickets."""
        result = pipeline.retrieve("I was charged for Pro but don't remember upgrading")
        source_ids = [chunk.source_id for chunk, _ in result["results"]]
        assert "doc_003" in source_ids or "ticket_003" in source_ids, \
            f"Expected billing sources, got {source_ids}"

    def test_empty_query_returns_results(self, pipeline):
        """Even vague queries should return some results."""
        result = pipeline.retrieve("help")
        assert len(result["results"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
