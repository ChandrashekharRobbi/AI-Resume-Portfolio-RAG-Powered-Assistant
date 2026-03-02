"""
Vector Store Module - Handles FAISS vector database operations with token optimization.

This module manages:
- Loading and processing documents
- Generating embeddings using sentence-transformers
- Creating and managing FAISS index
- Intelligent document retrieval with deduplication
- Chunk ranking and context compression
- Similarity-based caching
"""

import os
import pickle
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from config import (
    EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
    DEDUPLICATION_THRESHOLD,
    ENABLE_DEDUPLICATION,
    ENABLE_EMBEDDING_CACHE,
    ENABLE_CONTEXT_COMPRESSION
)


class VectorStore:
    """
    FAISS-based vector store with token optimization features.
    
    Features:
    - Semantic search with similarity filtering
    - Automatic deduplication of similar chunks
    - Chunk ranking by relevance
    - Context compression
    - Query result caching
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the vector store.
        
        Args:
            model_name: HuggingFace sentence-transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_cache = {}  # Cache embeddings to avoid recomputation
        self.query_cache = {}  # Cache search results
        
    def _get_embedding_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_query_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query results."""
        return f"{query}_{top_k}"
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        if not ENABLE_EMBEDDING_CACHE:
            return None
        
        hash_key = self._get_embedding_hash(text)
        return self.embedding_cache.get(hash_key)
    
    def _deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Remove duplicate or highly similar chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated list of chunks
        """
        if not ENABLE_DEDUPLICATION or len(chunks) < 2:
            return chunks
        
        print("🔍 Deduplicating chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create index for similarity search
        temp_index = faiss.IndexFlatL2(self.embedding_dim)
        temp_index.add(embeddings)
        
        # Find unique chunks
        unique_chunks = []
        unique_indices = set()
        
        for i, embedding in enumerate(embeddings):
            if i in unique_indices:
                continue
            
            unique_chunks.append(chunks[i])
            unique_indices.add(i)
            
            # Find similar chunks
            query_embedding = np.expand_dims(embedding, axis=0).astype('float32')
            distances, indices = temp_index.search(query_embedding, len(chunks))
            
            # Mark similar chunks for removal
            for idx, distance in zip(indices[0], distances[0]):
                similarity = 1 / (1 + distance)
                if similarity > DEDUPLICATION_THRESHOLD and idx != i:
                    unique_indices.add(idx)
        
        print(f"✓ Deduplicated: {len(chunks)} → {len(unique_chunks)} chunks")
        return unique_chunks
    
    def _rank_results_by_relevance(
        self, 
        results: List[Tuple], 
        top_k: int
    ) -> List[Tuple]:
        """
        Rank and filter results by relevance score.
        
        Args:
            results: List of (document, similarity, metadata) tuples
            top_k: Number of results to return
            
        Returns:
            Top-k results filtered by threshold
        """
        # Filter by minimum similarity threshold
        filtered = [
            r for r in results 
            if r[1] >= MIN_SIMILARITY_THRESHOLD
        ]
        
        # Sort by similarity descending
        ranked = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        # Return top k
        return ranked[:top_k]
    
    def _compress_context(self, chunks: List[str]) -> str:
        """
        Compress context by removing redundancy.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Compressed context string
        """
        if not ENABLE_CONTEXT_COMPRESSION or len(chunks) <= 1:
            return "\n\n".join(chunks)
        
        # Remove very short chunks that add no value
        filtered = [c for c in chunks if len(c.strip()) > 20]
        
        # Deduplicate similar content
        if len(filtered) > 1:
            embeddings = self.model.encode(filtered, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            unique = [filtered[0]]
            for i in range(1, len(filtered)):
                curr_emb = np.expand_dims(embeddings[i], axis=0)
                prev_embs = np.array([embeddings[j] for j in range(len(unique))])
                
                # Check similarity to all unique chunks
                distances = np.linalg.norm(prev_embs - curr_emb, axis=1)
                min_distance = distances.min()
                similarity = 1 / (1 + min_distance)
                
                if similarity < DEDUPLICATION_THRESHOLD:
                    unique.append(filtered[i])
            
            filtered = unique
        
        return "\n\n".join(filtered)
    
    def create_index(self, documents: List[str], metadatas: List[dict] = None) -> None:
        """
        Create FAISS index from documents.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts for each document
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        # Deduplicate chunks first
        if ENABLE_DEDUPLICATION:
            documents = self._deduplicate_chunks(documents)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Cache embeddings
        if ENABLE_EMBEDDING_CACHE:
            for doc, emb in zip(documents, embeddings):
                hash_key = self._get_embedding_hash(doc)
                self.embedding_cache[hash_key] = emb
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = metadatas or [{"id": i} for i in range(len(documents))]
        
        print(f"✓ Index created with {len(documents)} documents")
        
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents with optimization.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index first.")
        
        # Check query cache
        cache_key = self._get_query_cache_key(query, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Encode query
        query_embedding = self.model.encode([query])[0].astype('float32')
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.documents)))
        
        # Convert distances to similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                similarity = 1 / (1 + distance)
                results.append((self.documents[idx], similarity, self.metadata[idx]))
        
        # Rank and filter results
        ranked_results = self._rank_results_by_relevance(results, top_k)
        
        # Cache results
        self.query_cache[cache_key] = ranked_results
        
        return ranked_results
    
    def get_compressed_context(self, results: List[Tuple]) -> str:
        """
        Get compressed context from search results.
        
        Args:
            results: List of (document, similarity, metadata) tuples
            
        Returns:
            Compressed context string with separators
        """
        chunks = [doc for doc, _, _ in results]
        
        # Compress if enabled
        if ENABLE_CONTEXT_COMPRESSION:
            return self._compress_context(chunks)
        else:
            return "\n\n---\n\n".join(chunks)
    
    def save(self, path: str) -> None:
        """
        Save index and documents to disk.
        
        Args:
            path: Directory path to save files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss_index"))
        
        # Save documents and metadata
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save embedding cache
        if ENABLE_EMBEDDING_CACHE:
            with open(os.path.join(path, "embedding_cache.pkl"), "wb") as f:
                pickle.dump(self.embedding_cache, f)
        
        print(f"✓ Vector store saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load index and documents from disk.
        
        Args:
            path: Directory path to load files from
        """
        # Load FAISS index
        index_path = os.path.join(path, "faiss_index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load embedding cache if available
        cache_path = os.path.join(path, "embedding_cache.pkl")
        if os.path.exists(cache_path) and ENABLE_EMBEDDING_CACHE:
            with open(cache_path, "rb") as f:
                self.embedding_cache = pickle.load(f)
        
        print(f"✓ Vector store loaded from {path}")
    
    def is_indexed(self) -> bool:
        """Check if index is initialized."""
        return self.index is not None
    
    def clear_cache(self) -> None:
        """Clear query and embedding caches."""
        self.query_cache.clear()
        print("✓ Query cache cleared")
