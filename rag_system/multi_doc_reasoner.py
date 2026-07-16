"""
Multi-Document Reasoner — Complexity Challenge B.

Handles queries that require synthesizing information from multiple
source documents. For example:
  - "What are the differences between Free and Pro?" spans doc_001,
    doc_003, and doc_004
  - "My files aren't syncing" benefits from combining the official
    troubleshooting guide (doc_002) with real resolution (ticket_002)

Responsibilities:
  1. Detect whether a query needs multi-doc synthesis
  2. Group retrieved chunks by source document
  3. Identify overlapping/complementary information
  4. Detect cross-document contradictions (e.g., version differences)
  5. Prepare a structured context for the LLM with clear source attribution
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, OrderedDict

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class MultiDocReasoner:
    """
    Aggregates and structures information from multiple retrieved documents
    for the LLM to synthesize into a coherent response.
    """

    def analyze(self, results: List[Tuple[DocumentChunk, float]],
                query_info: Dict) -> Dict:
        """
        Analyze multi-document results and prepare structured context.
        
        Args:
            results: Retrieved (chunk, score) tuples
            query_info: Classification metadata from QueryClassifier
            
        Returns:
            dict with:
              - grouped_sources: chunks grouped by source_id
              - source_summaries: brief info about each source
              - contradictions: detected conflicts between sources
              - version_notes: version-related observations
              - is_multi_source: whether multiple sources are involved
              - context_for_llm: formatted context string for the LLM
        """
        if not results:
            return self._empty_analysis()

        # Group chunks by source document
        grouped = self._group_by_source(results)

        # Extract source-level summaries
        source_summaries = self._build_source_summaries(grouped)

        # Detect contradictions or version differences
        contradictions = self._detect_contradictions(grouped)
        version_notes = self._detect_version_differences(grouped)

        # Build formatted context for LLM
        context = self._build_llm_context(grouped, source_summaries,
                                           contradictions, version_notes,
                                           query_info)

        is_multi = len(grouped) > 1

        analysis = {
            "grouped_sources": grouped,
            "source_summaries": source_summaries,
            "contradictions": contradictions,
            "version_notes": version_notes,
            "is_multi_source": is_multi,
            "context_for_llm": context,
            "source_count": len(grouped),
        }

        logger.info(
            f"Multi-doc analysis: {len(grouped)} sources, "
            f"{len(contradictions)} contradictions, "
            f"{len(version_notes)} version notes"
        )

        return analysis

    def _group_by_source(self, results: List[Tuple[DocumentChunk, float]]) -> OrderedDict:
        """
        Group chunks by their source document, preserving relevance order.
        
        Returns OrderedDict: source_id -> [(chunk, score), ...]
        """
        grouped = OrderedDict()
        for chunk, score in results:
            if chunk.source_id not in grouped:
                grouped[chunk.source_id] = []
            grouped[chunk.source_id].append((chunk, score))
        return grouped

    def _build_source_summaries(self, grouped: OrderedDict) -> Dict[str, Dict]:
        """Build a summary for each source document."""
        summaries = {}
        for source_id, chunk_scores in grouped.items():
            first_chunk = chunk_scores[0][0]
            max_score = max(s for _, s in chunk_scores)
            summaries[source_id] = {
                "title": first_chunk.title,
                "source_type": first_chunk.source_type,
                "version": first_chunk.version,
                "last_updated": first_chunk.last_updated,
                "relevance_score": max_score,
                "chunk_count": len(chunk_scores),
                "category": first_chunk.category,
                "status": first_chunk.status,
            }
        return summaries

    def _detect_contradictions(self, grouped: OrderedDict) -> List[Dict]:
        """
        Detect potential contradictions between sources.
        
        Looks for:
          - Same topic covered by different versions
          - Resolved vs pending status for similar issues
          - Different instructions for the same task
        """
        contradictions = []

        # Check for version mismatches on the same topic
        sources = list(grouped.items())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                id_a, chunks_a = sources[i]
                id_b, chunks_b = sources[j]

                version_a = chunks_a[0][0].version
                version_b = chunks_b[0][0].version

                # Different versions of same-topic documents
                if version_a and version_b and version_a != version_b:
                    contradictions.append({
                        "type": "version_mismatch",
                        "source_a": id_a,
                        "version_a": version_a,
                        "source_b": id_b,
                        "version_b": version_b,
                        "note": f"Information may differ between {version_a} and {version_b}. "
                                f"Prefer the newer version unless the user is on the older one."
                    })

                # Resolved vs pending tickets
                status_a = chunks_a[0][0].status
                status_b = chunks_b[0][0].status
                if (status_a == "pending" and status_b == "resolved") or \
                   (status_a == "resolved" and status_b == "pending"):
                    contradictions.append({
                        "type": "status_conflict",
                        "source_a": id_a,
                        "status_a": status_a,
                        "source_b": id_b,
                        "status_b": status_b,
                        "note": "One source shows this issue as resolved while another "
                                "shows it as pending. The pending ticket may represent "
                                "an ongoing variation of the issue."
                    })

        return contradictions

    def _detect_version_differences(self, grouped: OrderedDict) -> List[Dict]:
        """Flag any version-specific information in the results."""
        version_notes = []
        versions_seen = set()

        for source_id, chunk_scores in grouped.items():
            chunk = chunk_scores[0][0]
            if chunk.version:
                versions_seen.add(chunk.version)

            # Check chunk content for version-specific mentions
            for c, _ in chunk_scores:
                content_lower = c.content.lower()
                if "v2.0" in content_lower and "v2.1" in content_lower:
                    version_notes.append({
                        "source_id": source_id,
                        "note": "This source discusses differences between v2.0 and v2.1",
                    })
                    break
                elif "outdated" in content_lower or "deprecated" in content_lower:
                    version_notes.append({
                        "source_id": source_id,
                        "note": "This source may contain outdated information",
                    })
                    break

        if len(versions_seen) > 1:
            version_notes.append({
                "source_id": "multiple",
                "note": f"Results span multiple versions: {', '.join(sorted(versions_seen))}. "
                        f"Prioritize information from the latest version."
            })

        return version_notes

    def _build_llm_context(self, grouped: OrderedDict,
                           summaries: Dict, contradictions: List,
                           version_notes: List,
                           query_info: Dict) -> str:
        """
        Build a structured context string for the LLM.
        
        This is the key output — it gives the LLM everything it needs
        to generate a well-cited, accurate response.
        """
        parts = []

        # Header with query classification
        parts.append(f"[Query Type: {query_info['query_type']}]")
        if query_info["has_negation"]:
            parts.append("[Note: Query contains negative language — user likely has a problem]")
        if query_info["is_multi_doc"]:
            parts.append("[Note: This query may require synthesizing information from multiple sources]")
        parts.append("")

        # Source documents
        for source_id, chunk_scores in grouped.items():
            summary = summaries[source_id]
            source_label = "Product Documentation" if summary["source_type"] == "product_doc" \
                          else "Support Ticket"
            
            parts.append(f"--- SOURCE: [{source_id}] {summary['title']} ---")
            parts.append(f"Type: {source_label}")
            if summary["version"]:
                parts.append(f"Version: {summary['version']}")
            if summary["last_updated"]:
                parts.append(f"Last Updated: {summary['last_updated']}")
            if summary["status"]:
                parts.append(f"Status: {summary['status']}")
            parts.append("")

            # Add chunk contents
            for chunk, score in chunk_scores:
                parts.append(chunk.content)
                parts.append("")

        # Add contradiction warnings
        if contradictions:
            parts.append("--- CONTRADICTIONS / CONFLICTS ---")
            for c in contradictions:
                parts.append(f"⚠ {c['note']}")
            parts.append("")

        # Add version notes
        if version_notes:
            parts.append("--- VERSION NOTES ---")
            for vn in version_notes:
                parts.append(f"📌 {vn['note']}")
            parts.append("")

        return "\n".join(parts)

    def _empty_analysis(self) -> Dict:
        """Return an empty analysis when no results are available."""
        return {
            "grouped_sources": OrderedDict(),
            "source_summaries": {},
            "contradictions": [],
            "version_notes": [],
            "is_multi_source": False,
            "context_for_llm": "",
            "source_count": 0,
        }
