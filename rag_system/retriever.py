"""
Smart Retriever — Part 2 of the RAG Pipeline.

Implements the full retrieval pipeline:
  1. Query Classification — categorize the query type (troubleshooting,
     billing, factual, etc.) to guide source prioritization
  2. Synonym Expansion — expand query with synonyms for better recall
  3. Negation Detection — detect negative intent to route toward
     troubleshooting docs
  4. Hybrid Retrieval — combine semantic (ChromaDB) + keyword (TF-IDF)
     results with configurable weights
  5. Document Prioritization — boost docs vs tickets based on query type
  6. Re-Ranking — final scoring by relevance × recency × source priority
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from document_processor import DocumentChunk
from indexer import Indexer
import config

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classifies user queries into categories to guide retrieval strategy."""

    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify a query and return metadata about it.
        
        Returns:
            dict with keys:
              - query_type: str (primary classification)
              - has_negation: bool
              - is_multi_doc: bool (likely needs multiple sources)
              - source_preference: str ("docs", "tickets", or "both")
        """
        query_lower = query.lower().strip()

        # Detect negation
        has_negation = self._detect_negation(query_lower)

        # Classify query type by keyword matching
        query_type = self._classify_type(query_lower)

        # Determine if this query likely needs multi-document reasoning
        is_multi_doc = self._is_multi_doc_query(query_lower, query_type)

        # Determine source preference
        source_preference = self._get_source_preference(query_type, has_negation)

        result = {
            "query_type": query_type,
            "has_negation": has_negation,
            "is_multi_doc": is_multi_doc,
            "source_preference": source_preference,
        }

        logger.info(f"Query classified: {result}")
        return result

    def _detect_negation(self, query: str) -> bool:
        """Check if the query contains negation patterns."""
        for pattern in config.NEGATION_PATTERNS:
            if pattern in query:
                return True
        return False

    def _classify_type(self, query: str) -> str:
        """Classify query type based on keyword matching. Returns best match."""
        scores = {}
        for qtype, keywords in config.QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                scores[qtype] = score

        if not scores:
            return "factual"  # Default type

        # Return the type with the highest keyword match count
        return max(scores, key=scores.get)

    def _is_multi_doc_query(self, query: str, query_type: str) -> bool:
        """Determine if this query likely needs multiple document sources."""
        multi_doc_indicators = [
            "difference", "compare", "between", "vs", "versus",
            "free and pro", "all features", "overview", "comprehensive",
            "everything about", "complete guide",
        ]
        if query_type == "comparison":
            return True
        return any(indicator in query for indicator in multi_doc_indicators)

    def _get_source_preference(self, query_type: str, has_negation: bool) -> str:
        """Determine whether to prefer docs, tickets, or both."""
        # Troubleshooting and issue-type queries benefit from real ticket resolutions
        ticket_preferred = {"troubleshooting", "known_issue", "performance", "technical_issue"}
        # Pure informational queries are best served by official docs
        doc_preferred = {"factual", "feature_usage", "comparison", "security",
                         "developer", "cancellation"}

        if has_negation:
            return "both"  # Negation suggests a problem — check both
        if query_type in ticket_preferred:
            return "both"  # Want both docs (for steps) and tickets (for real solutions)
        if query_type in doc_preferred:
            return "docs"
        return "both"


class SynonymExpander:
    """Expands queries with synonyms for improved recall."""

    def expand(self, query: str) -> str:
        """
        Expand a query by appending relevant synonyms.
        
        We don't replace words — we append synonyms so the original
        intent is preserved while broadening the search.
        """
        query_lower = query.lower()
        expansions = set()

        for word, synonyms in config.SYNONYM_MAP.items():
            if word in query_lower:
                # Add a subset of synonyms (not all, to avoid noise)
                expansions.update(synonyms[:3])

        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
            return expanded

        return query


class ReRanker:
    """
    Re-ranks retrieval results by combining multiple signals:
      - Base relevance score (from hybrid retrieval)
      - Recency boost (newer documents scored higher)
      - Source type boost (docs vs tickets, based on query type)
      - Version preference (newer versions preferred)
    """

    def rerank(self, results: List[Tuple[DocumentChunk, float]],
               query_info: Dict) -> List[Tuple[DocumentChunk, float]]:
        """
        Re-rank results using multi-signal scoring.
        
        Args:
            results: List of (chunk, base_score) from hybrid retrieval
            query_info: Classification metadata from QueryClassifier
            
        Returns:
            Re-ranked list of (chunk, final_score)
        """
        if not results:
            return []

        reranked = []
        for chunk, base_score in results:
            # 1. Recency boost
            recency_boost = self._compute_recency_boost(chunk.last_updated)

            # 2. Source type boost
            source_boost = self._compute_source_boost(
                chunk.source_type, query_info["source_preference"]
            )

            # 3. Version boost (prefer newer versions)
            version_boost = self._compute_version_boost(chunk.version)

            # 4. Status boost for tickets (resolved > pending)
            status_boost = self._compute_status_boost(chunk.status)

            # Combine signals
            final_score = (
                base_score
                * recency_boost
                * source_boost
                * version_boost
                * status_boost
            )

            reranked.append((chunk, final_score))

        # Sort by final score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(reranked)} results")
        return reranked

    def _compute_recency_boost(self, date_str: str) -> float:
        """Boost newer documents. Returns multiplier 0.8–1.2."""
        if not date_str:
            return 1.0
        try:
            doc_date = datetime.strptime(date_str, "%Y-%m-%d")
            # Use a reference date (latest in our dataset is 2024-02-01)
            reference = datetime(2024, 2, 15)
            days_old = (reference - doc_date).days
            # Linear decay: 0 days old = 1.15x, 365 days old = 0.85x
            boost = 1.15 - (days_old / config.RECENCY_DECAY_DAYS) * 0.3
            return max(0.85, min(1.15, boost))
        except (ValueError, TypeError):
            return 1.0

    def _compute_source_boost(self, source_type: str, preference: str) -> float:
        """Boost based on source type preference."""
        if preference == "both":
            return 1.0
        if preference == "docs" and source_type == "product_doc":
            return config.DOC_BOOST
        if preference == "tickets" and source_type == "support_ticket":
            return config.TICKET_BOOST
        return 1.0

    def _compute_version_boost(self, version: str) -> float:
        """Prefer newer product versions. v2.1 > v2.0."""
        if not version:
            return 1.0
        # Extract version number
        match = re.search(r'v?(\d+\.\d+)', version)
        if match:
            ver = float(match.group(1))
            # Small boost for newer versions
            if ver >= 2.1:
                return 1.05
            elif ver >= 2.0:
                return 1.0
        return 1.0

    def _compute_status_boost(self, status: str) -> float:
        """For tickets: resolved tickets are more useful than pending ones."""
        if status == "resolved":
            return 1.05
        elif status == "pending":
            return 0.95
        return 1.0


class SmartRetriever:
    """
    Orchestrates the full retrieval pipeline:
    classify → expand → retrieve (hybrid) → deduplicate → re-rank.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.classifier = QueryClassifier()
        self.expander = SynonymExpander()
        self.reranker = ReRanker()

    def retrieve(self, query: str, top_k: int = None) -> Dict:
        """
        Full retrieval pipeline.
        
        Args:
            query: User's question
            top_k: Number of final results (default: config.FINAL_TOP_K)
            
        Returns:
            dict with:
              - results: List of (chunk, score) tuples
              - query_info: Classification metadata
              - expanded_query: The synonym-expanded query
        """
        top_k = top_k or config.FINAL_TOP_K

        query_info = self.classifier.classify(query)

        expanded_query = self.expander.expand(query)

        semantic_results = self.indexer.semantic_search(
            expanded_query, n_results=config.INITIAL_RETRIEVAL_K
        )
        keyword_results = self.indexer.keyword_search(
            expanded_query, n_results=config.INITIAL_RETRIEVAL_K
        )

        merged = self._merge_results(semantic_results, keyword_results)

        reranked = self.reranker.rerank(merged, query_info)

        final_results = reranked[:top_k]

        logger.info(
            f"Retrieved {len(final_results)} results for query "
            f"(type={query_info['query_type']}, multi_doc={query_info['is_multi_doc']})"
        )

        return {
            "results": final_results,
            "query_info": query_info,
            "expanded_query": expanded_query,
        }

    def _merge_results(
        self,
        semantic: List[Tuple[DocumentChunk, float]],
        keyword: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Merge semantic and keyword results with weighted scoring.
        
        If a chunk appears in both result sets, its scores are combined
        using the configured weights. This is Reciprocal Rank Fusion lite.
        """
        chunk_scores: Dict[str, Tuple[DocumentChunk, float]] = {}

        # Normalize scores within each result set to [0, 1]
        semantic_scores = self._normalize_scores(semantic)
        keyword_scores = self._normalize_scores(keyword)

        # Add semantic results
        for chunk, score in semantic_scores:
            weighted = score * config.SEMANTIC_WEIGHT
            chunk_scores[chunk.chunk_id] = (chunk, weighted)

        # Add/merge keyword results
        for chunk, score in keyword_scores:
            weighted = score * config.KEYWORD_WEIGHT
            if chunk.chunk_id in chunk_scores:
                existing_chunk, existing_score = chunk_scores[chunk.chunk_id]
                chunk_scores[chunk.chunk_id] = (existing_chunk, existing_score + weighted)
            else:
                chunk_scores[chunk.chunk_id] = (chunk, weighted)

        # Convert to sorted list
        merged = list(chunk_scores.values())
        merged.sort(key=lambda x: x[1], reverse=True)

        return merged

    @staticmethod
    def _normalize_scores(results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not results:
            return []

        scores = [s for _, s in results]
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s

        if range_s == 0:
            # All scores are the same — assign 0.5 to all
            return [(chunk, 0.5) for chunk, _ in results]

        return [(chunk, (score - min_s) / range_s) for chunk, score in results]
