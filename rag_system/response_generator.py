"""
Response Generator — Part 3 of the RAG Pipeline.

Uses Google Gemini to generate natural language answers from the retrieved
and structured context. Handles:
  - Citation formatting (every claim attributed to a source)
  - Confidence indicators (HIGH / MEDIUM / LOW based on retrieval scores)
  - Insufficient information (graceful "I don't know" responses)
  - Conflicting information (flags contradictions from multi-doc analysis)
  - Outdated information (version-aware warnings)
  - Proper formatting (markdown with headers, steps, bold terms)
"""

import logging
from typing import List, Dict, Tuple, Optional

import google.generativeai as genai

from document_processor import DocumentChunk
from multi_doc_reasoner import MultiDocReasoner
import config

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates LLM-powered responses from retrieved context.
    Falls back to a structured template if the LLM is unavailable.
    """

    def __init__(self):
        self.reasoner = MultiDocReasoner()
        self.model = None
        self._initialize_gemini()

    def _initialize_gemini(self) -> None:
        """Initialize the Gemini API client."""
        if not config.GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY not set. Response generation will use fallback mode. "
                "Set your API key in rag_system/.env"
            )
            return

        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"Gemini model initialized: {config.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def generate(self, query: str,
                 results: List[Tuple[DocumentChunk, float]],
                 query_info: Dict) -> Dict:
        """
        Generate a response for the user query.
        
        Args:
            query: The original user question
            results: Retrieved (chunk, score) tuples
            query_info: Classification metadata
            
        Returns:
            dict with:
              - answer: str (the generated response in markdown)
              - confidence: str (HIGH / MEDIUM / LOW)
              - sources: list of source citations
              - query_type: str
              - has_conflicts: bool
              - version_warnings: list of version notes
        """
        confidence = self._compute_confidence(results)

        analysis = self.reasoner.analyze(results, query_info)

        sources = self._build_citations(analysis["source_summaries"])

        if confidence == "NONE" or not results:
            return self._insufficient_info_response(query, query_info)

        if self.model:
            answer = self._generate_with_gemini(query, analysis, confidence, query_info)
        else:
            answer = self._generate_fallback(query, analysis, confidence)

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "query_type": query_info["query_type"],
            "has_conflicts": len(analysis["contradictions"]) > 0,
            "version_warnings": [vn["note"] for vn in analysis["version_notes"]],
            "source_count": analysis["source_count"],
            "is_multi_source": analysis["is_multi_source"],
        }

    def _generate_with_gemini(self, query: str, analysis: Dict,
                               confidence: str, query_info: Dict) -> str:
        """Generate a response using Google Gemini."""
        system_prompt = self._build_system_prompt(confidence, query_info)
        user_prompt = self._build_user_prompt(query, analysis)

        try:
            response = self.model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,     # Low temp for factual accuracy
                    max_output_tokens=1024,
                    top_p=0.8,
                )
            )
            answer = response.text
            logger.info("Gemini response generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}. Using fallback.")
            return self._generate_fallback(query, analysis, confidence)

    def _build_system_prompt(self, confidence: str, query_info: Dict) -> str:
        """Build the system instruction for Gemini."""
        return f"""You are a helpful CloudSync customer support assistant. Your job is to answer 
customer queries accurately using ONLY the provided source documents.

RULES:
1. ONLY use information from the provided sources. Do NOT make up information.
2. Cite every key fact using the format [Source: source_id - title].
3. If information is insufficient, clearly state what you don't know and suggest contacting support.
4. If sources conflict, mention both versions and recommend the most recent one.
5. Format your response with markdown: use **bold** for key terms, numbered lists for steps, 
   and headers (##) for sections when the answer is long.
6. Keep the tone professional, friendly, and concise.
7. If the query is about troubleshooting, provide step-by-step instructions.
8. Mention version differences when relevant (e.g., v2.0 vs v2.1).

CONFIDENCE LEVEL: {confidence}
QUERY TYPE: {query_info['query_type']}
{"NOTE: The user appears to have a problem (negative language detected)." if query_info['has_negation'] else ""}
{"NOTE: This answer requires synthesizing information from multiple sources." if query_info['is_multi_doc'] else ""}"""

    def _build_user_prompt(self, query: str, analysis: Dict) -> str:
        """Build the user prompt with retrieved context."""
        return f"""CUSTOMER QUESTION: {query}

RETRIEVED CONTEXT:
{analysis['context_for_llm']}

Please provide a helpful, accurate answer to the customer's question based on the above context.
Remember to cite your sources using [Source: source_id - title] format."""

    def _generate_fallback(self, query: str, analysis: Dict,
                           confidence: str) -> str:
        """
        Fallback response generator when Gemini is unavailable.
        Constructs a structured response from the retrieved chunks.
        """
        parts = []

        if confidence == "LOW":
            parts.append(
                "> ⚠️ **Low Confidence**: The following information may not fully "
                "address your question. Consider contacting support for assistance.\n"
            )

        # Add content from each source
        for source_id, chunks_scores in analysis["grouped_sources"].items():
            summary = analysis["source_summaries"][source_id]
            source_label = "📄" if summary["source_type"] == "product_doc" else "🎫"
            
            parts.append(f"### {source_label} From: {summary['title']}")
            if summary["version"]:
                parts.append(f"*Version: {summary['version']}*\n")
            
            for chunk, score in chunks_scores:
                parts.append(chunk.content)
                parts.append("")
            
            parts.append(f"*[Source: {source_id} - {summary['title']}]*\n")

        # Add warnings
        if analysis["contradictions"]:
            parts.append("---\n### ⚠️ Conflicting Information")
            for c in analysis["contradictions"]:
                parts.append(f"- {c['note']}")
            parts.append("")

        if analysis["version_notes"]:
            parts.append("---\n### 📌 Version Notes")
            for vn in analysis["version_notes"]:
                parts.append(f"- {vn['note']}")

        return "\n".join(parts)

    def _insufficient_info_response(self, query: str, query_info: Dict) -> Dict:
        """Generate a response when no relevant information is found."""
        answer = (
            "I don't have enough information in the available documentation to "
            f"fully answer your question about: *\"{query}\"*\n\n"
            "**Here's what I suggest:**\n"
            "1. Check the [CloudSync Help Center](https://help.cloudsync.com) "
            "for the latest documentation\n"
            "2. Contact our support team at **support@cloudsync.com**\n"
            "3. If this is urgent, use the in-app chat for real-time assistance\n\n"
            "*Our support team is available 24/7 and will be happy to help!*"
        )
        return {
            "answer": answer,
            "confidence": "NONE",
            "sources": [],
            "query_type": query_info["query_type"],
            "has_conflicts": False,
            "version_warnings": [],
            "source_count": 0,
            "is_multi_source": False,
        }

    def _compute_confidence(self, results: List[Tuple[DocumentChunk, float]]) -> str:
        """
        Compute confidence level from retrieval scores.
        
        Uses the top result's score as the confidence indicator:
          - HIGH: top score >= 0.55 (strong semantic/keyword match)
          - MEDIUM: top score >= 0.30
          - LOW: top score >= 0.20
          - NONE: no results or all scores < 0.20
        """
        if not results:
            return "NONE"

        top_score = results[0][1]

        if top_score >= config.HIGH_CONFIDENCE_THRESHOLD:
            return "HIGH"
        elif top_score >= config.MEDIUM_CONFIDENCE_THRESHOLD:
            return "MEDIUM"
        elif top_score >= config.NO_RESULT_THRESHOLD:
            return "LOW"
        else:
            return "NONE"

    def _build_citations(self, source_summaries: Dict) -> List[Dict]:
        """Build structured citation objects for the API response."""
        citations = []
        for source_id, summary in source_summaries.items():
            citations.append({
                "source_id": source_id,
                "title": summary["title"],
                "type": summary["source_type"],
                "version": summary["version"],
                "last_updated": summary["last_updated"],
                "relevance_score": round(summary["relevance_score"], 3),
            })
        return citations
