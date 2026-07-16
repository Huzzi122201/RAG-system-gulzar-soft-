"""
Document Processor — Part 1 of the RAG Pipeline.

Loads product_docs.json and support_tickets.json, then performs intelligent
chunking that preserves document structure, metadata, and step-by-step
instructions. Each chunk is a self-contained unit with full provenance info
for downstream retrieval and citation.

Chunking Strategy:
  - Product docs: Split by section headers (bold markdown **...**) to keep
    related info together. Each chunk retains the parent doc title/ID/version.
  - Support tickets: Split into semantic sections (Customer Issue, Resolution,
    Root Cause, etc.) so retrieval can target the relevant part of a ticket.
  - Small documents are kept as single chunks to avoid fragmenting context.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text with full metadata for retrieval and citation."""
    chunk_id: str               # Unique ID: e.g., "doc_001_chunk_0"
    source_id: str              # Original document/ticket ID
    source_type: str            # "product_doc" or "support_ticket"
    title: str                  # Document/ticket title
    section: str                # Section name within the document
    content: str                # The actual text content
    version: str                # Product version (e.g., "v2.1")
    last_updated: str           # ISO date string
    tags: List[str]             # Tags/categories for filtering
    chunk_index: int            # Position within the parent document
    total_chunks: int           # Total chunks from parent document
    category: str = ""          # Category (for tickets: authentication, billing, etc.)
    priority: str = ""          # Priority (for tickets: high, medium, low)
    status: str = ""            # Status (for tickets: resolved, pending)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class DocumentProcessor:
    """
    Loads and chunks product documentation and support tickets.
    
    Produces a flat list of DocumentChunk objects ready for embedding
    and indexing.
    """

    # Minimum chunk size in characters — avoid tiny fragments
    MIN_CHUNK_SIZE = 50

    def __init__(self, product_docs_path: str, support_tickets_path: str):
        self.product_docs_path = product_docs_path
        self.support_tickets_path = support_tickets_path
        self.chunks: List[DocumentChunk] = []

    def process_all(self) -> List[DocumentChunk]:
        """Load and process all documents. Returns list of chunks."""
        logger.info("Starting document processing...")
        
        # Load product docs
        product_docs = self._load_json(self.product_docs_path, "product_docs")
        logger.info(f"Loaded {len(product_docs)} product documents")
        
        # Load support tickets
        support_tickets = self._load_json(self.support_tickets_path, "support_tickets")
        logger.info(f"Loaded {len(support_tickets)} support tickets")

        # Process each document type
        for doc in product_docs:
            doc_chunks = self._chunk_product_doc(doc)
            self.chunks.extend(doc_chunks)
            logger.debug(f"  {doc['id']}: {len(doc_chunks)} chunks")

        for ticket in support_tickets:
            ticket_chunks = self._chunk_support_ticket(ticket)
            self.chunks.extend(ticket_chunks)
            logger.debug(f"  {ticket['id']}: {len(ticket_chunks)} chunks")

        logger.info(f"Total chunks created: {len(self.chunks)}")
        return self.chunks

    def _load_json(self, path: str, key: str) -> List[Dict]:
        """Load a JSON file and extract the array under the given key."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(key, [])
        except json.JSONDecodeError:
            # Handle malformed JSON (like the test_queries.json issue)
            logger.warning(f"JSON decode error in {path}, attempting repair...")
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Try to fix common issues: trailing content after array
            raw = self._repair_json(raw)
            data = json.loads(raw)
            return data.get(key, [])
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            return []

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Attempt to repair common JSON issues."""
        # Fix the test_queries.json issue: '}notes":' -> just remove malformed trailing content
        # Strategy: find the last valid closing of the main array and truncate
        # Find last ']' before the final '}'
        lines = raw.split('\n')
        repaired_lines = []
        for line in lines:
            # Fix lines like: '}notes": "..."' -> just take the notes part out
            if re.match(r'\s*\}.*notes', line):
                # Extract the notes value if possible
                notes_match = re.search(r'"notes"\s*:\s*"([^"]*)"', line)
                if notes_match:
                    repaired_lines.append(f'      "evaluation_notes": "{notes_match.group(1)}"')
                else:
                    repaired_lines.append('      }')
            else:
                repaired_lines.append(line)
        
        repaired = '\n'.join(repaired_lines)
        
        # If still broken, try aggressive fix: just close after last complete object
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            # Find the position of the last complete test query object
            # and truncate everything after
            last_bracket = repaired.rfind(']')
            if last_bracket > 0:
                # Find the matching closing brace
                truncated = repaired[:last_bracket + 1] + "\n}"
                try:
                    json.loads(truncated)
                    return truncated
                except json.JSONDecodeError:
                    pass
            # Last resort: return as-is and let caller handle the error
            return repaired

    def _chunk_product_doc(self, doc: Dict) -> List[DocumentChunk]:
        """
        Chunk a product document by section headers.
        
        Splits on bold markdown headers (**Header:**) to create semantically
        meaningful chunks. Each chunk gets the full document metadata.
        """
        content = doc.get("content", "")
        doc_id = doc["id"]
        
        # Split by bold section headers: **Section Name:**
        sections = self._split_by_sections(content)
        
        if len(sections) <= 1:
            # Document has no clear sections — keep as single chunk
            return [self._create_chunk(
                doc_id=doc_id,
                source_type="product_doc",
                title=doc["title"],
                section="full_document",
                content=content,
                version=doc.get("version", ""),
                last_updated=doc.get("last_updated", ""),
                tags=doc.get("tags", []),
                chunk_index=0,
                total_chunks=1,
                doc_type=doc.get("type", ""),
            )]
        
        chunks = []
        for i, (section_name, section_content) in enumerate(sections):
            # Skip tiny fragments
            if len(section_content.strip()) < self.MIN_CHUNK_SIZE:
                continue
            chunks.append(self._create_chunk(
                doc_id=doc_id,
                source_type="product_doc",
                title=doc["title"],
                section=section_name,
                content=section_content.strip(),
                version=doc.get("version", ""),
                last_updated=doc.get("last_updated", ""),
                tags=doc.get("tags", []),
                chunk_index=i,
                total_chunks=len(sections),
                doc_type=doc.get("type", ""),
            ))
        
        return chunks if chunks else [self._create_chunk(
            doc_id=doc_id,
            source_type="product_doc",
            title=doc["title"],
            section="full_document",
            content=content,
            version=doc.get("version", ""),
            last_updated=doc.get("last_updated", ""),
            tags=doc.get("tags", []),
            chunk_index=0,
            total_chunks=1,
            doc_type=doc.get("type", ""),
        )]

    def _chunk_support_ticket(self, ticket: Dict) -> List[DocumentChunk]:
        """
        Chunk a support ticket by semantic sections.
        
        Support tickets typically have: Customer Issue, Troubleshooting/Diagnosis,
        Resolution, Follow-up. We split on these to allow targeted retrieval
        (e.g., retrieval can find just the resolution for a similar issue).
        """
        content = ticket.get("content", "")
        ticket_id = ticket["id"]
        
        # Split by ticket section patterns
        sections = self._split_ticket_sections(content)
        
        common_kwargs = dict(
            source_type="support_ticket",
            title=ticket["title"],
            version=ticket.get("user_version", ""),
            last_updated=ticket.get("resolved_date") or ticket.get("created_date", ""),
            tags=ticket.get("tags", []),
            category=ticket.get("category", ""),
            priority=ticket.get("priority", ""),
            status=ticket.get("status", ""),
        )
        
        if len(sections) <= 1:
            return [self._create_chunk(
                doc_id=ticket_id, section="full_ticket",
                content=content, chunk_index=0, total_chunks=1,
                **common_kwargs
            )]
        
        chunks = []
        for i, (section_name, section_content) in enumerate(sections):
            if len(section_content.strip()) < self.MIN_CHUNK_SIZE:
                continue
            chunks.append(self._create_chunk(
                doc_id=ticket_id, section=section_name,
                content=section_content.strip(), chunk_index=i,
                total_chunks=len(sections), **common_kwargs
            ))
        
        return chunks if chunks else [self._create_chunk(
            doc_id=ticket_id, section="full_ticket",
            content=content, chunk_index=0, total_chunks=1,
            **common_kwargs
        )]

    @staticmethod
    def _split_by_sections(content: str) -> List[tuple]:
        """
        Split document content by bold markdown headers (**Header:**).
        Returns list of (section_name, section_content) tuples.
        """
        # Pattern: **Section Name:** or **Section Name**
        pattern = r'\*\*([^*]+?)\*\*:?'
        parts = re.split(pattern, content)
        
        if len(parts) <= 1:
            # No sections found — return intro as single section
            return [("introduction", content)]
        
        sections = []
        
        # First part is the intro (before any header)
        intro = parts[0].strip()
        if intro and len(intro) > 30:
            sections.append(("introduction", intro))
        
        # Remaining parts alternate: header, content, header, content...
        for i in range(1, len(parts), 2):
            header = parts[i].strip().lower().replace(" ", "_").replace(":", "")
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if intro and not sections[0][1].endswith(body):
                # Reconstruct with header for context
                full_content = f"**{parts[i].strip()}**: {body}"
                sections.append((header, full_content))
        
        return sections

    @staticmethod
    def _split_ticket_sections(content: str) -> List[tuple]:
        """
        Split support ticket content by semantic section markers.
        
        Common patterns in tickets:
          "Customer Issue:", "Troubleshooting Steps:", "Resolution:",
          "Root Cause:", "Follow-up:", etc.
        """
        # Pattern matches section headers like "Customer Issue:", "Resolution:"
        pattern = r'((?:Customer Issue|Troubleshooting (?:Steps|Process|Attempted)|' \
                  r'Diagnostic Process|Investigation|Initial Troubleshooting|' \
                  r'Root Cause(?: Analysis)?|Diagnosis|Resolution|' \
                  r'Immediate Solution|Optimization Guidance|Code Review|' \
                  r'Customer Response|Outcome|Follow-up|Customer Follow-up|' \
                  r'Verification Steps|Current Status|Temporary Workaround|' \
                  r'Next Steps|Preventive Advice|Additional Actions|' \
                  r'Device Information|Reproduction|Technical Fix|' \
                  r'Manual Setup Success|Security Education|UI Improvement|' \
                  r'Account Analysis|Documentation Update|' \
                  r'Version Comparison|Immediate Solution)\s*:)'
        
        parts = re.split(pattern, content, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            return [("full_ticket", content)]
        
        sections = []
        
        # First part (before any section header)
        if parts[0].strip():
            sections.append(("preamble", parts[0].strip()))
        
        # Pair up headers with their content
        for i in range(1, len(parts), 2):
            header = parts[i].strip().rstrip(':').lower().replace(" ", "_")
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            full_content = f"{parts[i].strip()} {body}"
            sections.append((header, full_content))
        
        return sections

    def _create_chunk(self, doc_id: str, source_type: str, title: str,
                      section: str, content: str, version: str,
                      last_updated: str, tags: List[str], chunk_index: int,
                      total_chunks: int, doc_type: str = "",
                      category: str = "", priority: str = "",
                      status: str = "") -> DocumentChunk:
        """Create a DocumentChunk with a unique ID."""
        chunk_id = f"{doc_id}_chunk_{chunk_index}"
        return DocumentChunk(
            chunk_id=chunk_id,
            source_id=doc_id,
            source_type=source_type,
            title=title,
            section=section,
            content=content,
            version=version,
            last_updated=last_updated,
            tags=tags,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            category=category,
            priority=priority,
            status=status,
            metadata={
                "doc_type": doc_type,
                "content_length": len(content),
            }
        )


def load_test_queries(path: str) -> List[Dict]:
    """
    Load test queries from test_queries.json.
    Handles the known JSON formatting issue in the file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        # Fix known issue: line with '}notes":'
        # The file has a malformed line that breaks JSON parsing
        lines = raw.split('\n')
        fixed_lines = []
        skip_evaluation_block = False
        
        for line in lines:
            # Skip the embedded evaluation_metrics block in query_001
            # and the malformed '}notes":' line
            if '"evaluation_metrics"' in line:
                skip_evaluation_block = True
                continue
            if skip_evaluation_block:
                if '}notes"' in line:
                    # Extract notes value and convert to evaluation_notes
                    notes_match = re.search(r'"([^"]+)"$', line.strip())
                    if notes_match:
                        fixed_lines.append(f'      "evaluation_notes": {json.dumps(notes_match.group(1))}')
                    skip_evaluation_block = False
                    continue
                # Skip lines inside the evaluation_metrics block
                continue
            
            # Remove trailing content after the test_queries array
            if '"evaluation_' in line and line.strip().startswith('"evaluation_'):
                # This might be the trailing evaluation_ at end of file
                if not '"evaluation_notes"' in line:
                    continue
            
            fixed_lines.append(line)
        
        fixed_raw = '\n'.join(fixed_lines)
        
        # Ensure proper JSON closure
        if not fixed_raw.rstrip().endswith('}'):
            # Find last ] and add }
            last_bracket = fixed_raw.rfind(']')
            if last_bracket > 0:
                fixed_raw = fixed_raw[:last_bracket + 1] + '\n}'
        
        data = json.loads(fixed_raw)
        return data.get("test_queries", [])
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading test queries: {e}")
        # Fallback: manually parse what we can
        return _fallback_parse_queries(path)


def _fallback_parse_queries(path: str) -> List[Dict]:
    """Fallback parser that extracts queries using regex if JSON parsing fails."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        queries = []
        # Find all query objects using regex
        pattern = r'"id"\s*:\s*"(query_\d+)".*?"query"\s*:\s*"([^"]+)"'
        for match in re.finditer(pattern, raw, re.DOTALL):
            queries.append({
                "id": match.group(1),
                "query": match.group(2),
            })
        
        logger.warning(f"Fallback parser extracted {len(queries)} queries")
        return queries
    except Exception as e:
        logger.error(f"Fallback parser failed: {e}")
        return []
