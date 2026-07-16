"""
Indexer — Embedding Generation & Vector Store Management.

Responsibilities:
  1. Generate dense embeddings using sentence-transformers (all-MiniLM-L6-v2)
  2. Store embeddings + metadata in ChromaDB for semantic search
  3. Build a TF-IDF index (scikit-learn) for keyword-based search
  4. Persist both indices to disk so subsequent starts are fast

The dual-index approach enables hybrid retrieval: semantic similarity
catches paraphrases and meaning, while TF-IDF catches exact keywords
and technical terms that embeddings might miss.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

from document_processor import DocumentChunk
import config

logger = logging.getLogger(__name__)


class Indexer:
    """
    Manages both the vector store (ChromaDB) and keyword index (TF-IDF)
    for hybrid retrieval.
    """

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.chunks: List[DocumentChunk] = []
        self._chunk_id_to_index: Dict[str, int] = {}

        # Paths for persisted TF-IDF index
        self._tfidf_dir = os.path.join(os.path.dirname(__file__), "tfidf_index")
        self._tfidf_vectorizer_path = os.path.join(self._tfidf_dir, "vectorizer.pkl")
        self._tfidf_matrix_path = os.path.join(self._tfidf_dir, "matrix.pkl")
        self._chunks_path = os.path.join(self._tfidf_dir, "chunks.pkl")

    def initialize(self, chunks: List[DocumentChunk], force_rebuild: bool = False) -> None:
        """
        Initialize both indices. If persisted data exists and force_rebuild
        is False, loads from disk. Otherwise builds from scratch.
        """
        logger.info("Initializing indexer...")

        # Load the embedding model (downloads on first run, ~80MB)
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

        # Check if we can load from cache
        if not force_rebuild and self._indices_exist():
            logger.info("Loading indices from cache...")
            self._load_cached_indices()
            logger.info(f"Loaded {len(self.chunks)} cached chunks")
            return

        # Build from scratch
        self.chunks = chunks
        self._chunk_id_to_index = {c.chunk_id: i for i, c in enumerate(chunks)}

        self._build_chroma_index(chunks)
        self._build_tfidf_index(chunks)
        self._persist_tfidf_index()

        logger.info("Indexer initialization complete")

    def _indices_exist(self) -> bool:
        """Check if persisted indices exist on disk."""
        return (
            os.path.exists(self._tfidf_vectorizer_path)
            and os.path.exists(self._tfidf_matrix_path)
            and os.path.exists(self._chunks_path)
            and os.path.exists(config.CHROMA_PERSIST_DIR)
        )

    # ChromaDB (Semantic Search)

    def _build_chroma_index(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings and store in ChromaDB."""
        logger.info("Building ChromaDB vector index...")

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        # Delete existing collection if rebuilding
        try:
            self.chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Prepare data for batch insertion
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        # Generate embeddings in batch (much faster than one-by-one)
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        raw_embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        for i, chunk in enumerate(chunks):
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "section": chunk.section,
                "version": chunk.version,
                "last_updated": chunk.last_updated,
                "tags": ",".join(chunk.tags),
                "category": chunk.category,
                "priority": chunk.priority,
                "status": chunk.status,
                "chunk_index": chunk.chunk_index,
            })
            embeddings.append(raw_embeddings[i].tolist())

        # Insert in batches (ChromaDB has batch size limits)
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        logger.info(f"ChromaDB index built with {len(ids)} vectors")

    def semantic_search(self, query: str, n_results: int = 10,
                        where_filter: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search ChromaDB for semantically similar chunks.
        
        Returns list of (chunk, similarity_score) tuples sorted by relevance.
        """
        if not self.collection or not self.embedding_model:
            logger.error("Indexer not initialized")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        # Query ChromaDB
        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, len(self.chunks)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        # Convert results to (chunk, score) tuples
        # ChromaDB returns distances; for cosine, distance = 1 - similarity
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                idx = self._chunk_id_to_index.get(chunk_id)
                if idx is not None:
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    output.append((self.chunks[idx], similarity))

        return output

    # TF-IDF (Keyword Search)

    def _build_tfidf_index(self, chunks: List[DocumentChunk]) -> None:
        """Build TF-IDF index for keyword-based search."""
        logger.info("Building TF-IDF keyword index...")

        # Combine content with metadata for richer keyword matching
        corpus = []
        for chunk in chunks:
            # Enrich the text with title and tags for better keyword matching
            enriched = f"{chunk.title} {' '.join(chunk.tags)} {chunk.content}"
            corpus.append(enriched)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),      # Unigrams and bigrams
            sublinear_tf=True,        # Apply log normalization
            min_df=1,
            max_df=0.95,
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        logger.info(f"TF-IDF index built: {self.tfidf_matrix.shape}")

    def keyword_search(self, query: str, n_results: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """
        Search using TF-IDF keyword matching.
        
        Returns list of (chunk, similarity_score) tuples sorted by relevance.
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.error("TF-IDF index not initialized")
            return []

        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])

        # Compute cosine similarity with all chunks
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1][:n_results]

        output = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero scores
                output.append((self.chunks[idx], float(similarities[idx])))

        return output

    # Persistence

    def _persist_tfidf_index(self) -> None:
        """Save TF-IDF index and chunks to disk."""
        os.makedirs(self._tfidf_dir, exist_ok=True)
        with open(self._tfidf_vectorizer_path, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(self._tfidf_matrix_path, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(self._chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("TF-IDF index persisted to disk")

    def _load_cached_indices(self) -> None:
        """Load persisted indices from disk."""
        # Load TF-IDF
        with open(self._tfidf_vectorizer_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(self._tfidf_matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(self._chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self._chunk_id_to_index = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        # Load ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its ID."""
        idx = self._chunk_id_to_index.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    def get_chunks_by_source(self, source_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks from a specific source document/ticket."""
        return [c for c in self.chunks if c.source_id == source_id]
