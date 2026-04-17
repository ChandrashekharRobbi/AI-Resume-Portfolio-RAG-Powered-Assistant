"""
RAG Pipeline Module - Implements token-optimized Retrieval Augmented Generation workflow.

Features:
- Smart query classification (greeting detection, small talk filtering)
- Minimal context retrieval (top-3 by default vs top-5)
- Token-efficient prompt templates
- Context deduplication and compression
- Query caching to avoid repeated API calls
- Token usage monitoring
"""

import os
import glob
import re
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOKENS_PER_RESPONSE,
    MAX_CONTEXT_TOKENS,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    GREETING_QUERIES,
    GREETING_RESPONSE,
    PROFILE_QUESTION_PATTERNS,
    DEBUG_MODE,
    LOG_TOKEN_USAGE,
    EMBEDDING_MODEL,
    CACHE_SIMILARITY_THRESHOLD,
    MAX_CACHE_SIZE,
)


class SemanticCache:
    """
    Semantic caching using embeddings for query similarity.
    
    Instead of exact key matching, finds semantically similar cached queries
    using cosine similarity on embeddings.
    
    Features:
    - Embedding-based query similarity
    - Configurable similarity threshold
    - Memory-efficient with max cache size
    - Fast similarity search using numpy operations
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        """
        Initialize semantic cache.
        
        Args:
            model_name: HuggingFace sentence-transformer model name
            similarity_threshold: Minimum similarity score for cache hit (0-1)
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.cache_queries: List[str] = []
        # Get actual embedding dimension from model
        sample_embedding = self.embedding_model.encode("sample", convert_to_numpy=True)
        embedding_dim = sample_embedding.shape[0]
        self.cache_embeddings: np.ndarray = np.array([]).reshape(0, embedding_dim)
        self.embedding_dim = embedding_dim
        self.cache_responses: List[str] = []
        self.max_size = MAX_CACHE_SIZE
        if DEBUG_MODE:
            print(f"🔧 SemanticCache initialized. Embedding dim: {embedding_dim}, Threshold: {similarity_threshold}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent semantic matching."""
        import string
        # Convert to lowercase
        normalized = query.lower()
        # Remove punctuation
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        # Strip and normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if semantically similar query exists.
        
        Args:
            query: User query
            
        Returns:
            Cached response if found above threshold, else None
        """
        if len(self.cache_queries) == 0:
            if DEBUG_MODE:
                print(f"📭 Cache empty, no match possible")
            return None
        
        # Normalize and embed the query
        normalized_query = self._normalize_query(query)
        query_embedding = self.embedding_model.encode(normalized_query, convert_to_numpy=True)
        
        # Find all similar cached queries
        similarities = np.array([
            self._cosine_similarity(query_embedding, cached_emb)
            for cached_emb in self.cache_embeddings
        ])
        
        max_similarity = np.max(similarities)
        best_idx = np.argmax(similarities)
        
        if DEBUG_MODE:
            print(f"\n🔍 Cache lookup:")
            print(f"   Query: '{query}'")
            print(f"   Normalized: '{normalized_query}'")
            print(f"   Best match: '{self.cache_queries[best_idx]}' (Similarity: {max_similarity:.3f})")
            print(f"   Threshold: {self.similarity_threshold}")
        
        # Return cached response if above threshold
        if max_similarity >= self.similarity_threshold:
            if DEBUG_MODE:
                print(f"   ✅ CACHE HIT!")
            return self.cache_responses[best_idx]
        else:
            if DEBUG_MODE:
                print(f"   ❌ Cache miss (below threshold)")
        
        return None
    
    def put(self, query: str, response: str) -> None:
        """
        Add query-response pair to cache.
        
        Args:
            query: User query
            response: Generated response
        """
        if len(self.cache_queries) >= self.max_size:
            # Remove oldest entry if at max capacity
            self.cache_queries.pop(0)
            self.cache_embeddings = self.cache_embeddings[1:]
            self.cache_responses.pop(0)
        
        # Normalize and embed
        normalized_query = self._normalize_query(query)
        query_embedding = self.embedding_model.encode(normalized_query, convert_to_numpy=True)
        self.cache_queries.append(query)
        self.cache_embeddings = np.vstack([self.cache_embeddings, query_embedding.reshape(1, -1)])
        self.cache_responses.append(response)
        
        if DEBUG_MODE:
            print(f"💾 Cached: {query[:50]}... (Cache size: {len(self.cache_queries)})")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache_queries.clear()
        self.cache_embeddings = np.array([]).reshape(0, self.embedding_dim)
        self.cache_responses.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_queries": len(self.cache_queries),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold
        }


class RAGPipeline:
    """
    RAG Pipeline with aggressive token optimization.
    
    Implements:
    1. Query classification (reduce unnecessary API calls)
    2. Minimal context retrieval (top-3 chunks default)
    3. Context compression and deduplication
    4. Token-efficient prompt construction
    5. Response caching
    6. Token usage monitoring
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize RAG pipeline.
        
        Args:
            groq_api_key: Groq API key
        """
        self.groq_api_key = groq_api_key
        self.vector_store = VectorStore()
        self.semantic_cache = SemanticCache()
        self.llm = None
        self.chunks = []
        self.token_usage_log = []
        self._initialize_llm()
        
    def _initialize_llm(self) -> None:
        """Initialize Groq LLM with token-optimized settings."""
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS_PER_RESPONSE
        )
    
    def _classify_query(self, query: str) -> str:
        """
        Classify query type to optimize processing.
        
        Returns:
            "greeting", "small_talk", "profile_question", or "general"
        """
        query_lower = query.lower().strip().split()
        # Check for greeting
        if any(query in GREETING_QUERIES for query in query_lower):
            return "greeting"
        
        # Check for profile questions (need RAG)
        if any(query in PROFILE_QUESTION_PATTERNS for query in query_lower):
            return "profile_question"
        
        # Default to general (needs LLM but maybe not RAG)
        return "general"
    
    def _is_cached_response(self, query: str) -> Optional[str]:
        """Check if we have a semantically similar cached response."""
        return self.semantic_cache.get(query)
    
    def _log_token_usage(self, query: str, context: str, response: str, query_type: str):
        """Log token usage for monitoring."""
        if not LOG_TOKEN_USAGE:
            return
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        context_tokens = len(context) // 4
        response_tokens = len(response) // 4
        total_tokens = context_tokens + response_tokens
        
        log_entry = {
            "query": query,
            "type": query_type,
            "context_tokens": context_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
        
        self.token_usage_log.append(log_entry)
        
        if DEBUG_MODE:
            print(f"📊 Tokens - Context: {context_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    
    def load_documents(self, data_folder: str = "data") -> List[str]:
        """
        Load all text documents from data folder.
        
        Args:
            data_folder: Path to folder containing text files
            
        Returns:
            List of document contents
        """
        documents = []
        
        # Find all .txt files
        txt_files = glob.glob(os.path.join(data_folder, "*.txt"))
        
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {data_folder}")
        
        print(f"📂 Loading {len(txt_files)} documents...")
        
        for file_path in sorted(txt_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    print(f"  ✓ {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")
        
        return documents
    
    def split_documents(
        self,
        documents: List[str],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split documents into optimized chunks.
        
        Args:
            documents: List of document texts
            chunk_size: Size of each chunk (reduced for efficiency)
            chunk_overlap: Overlap between chunks (minimal)
            
        Returns:
            List of document chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = splitter.split_text(doc)
            chunks.extend(doc_chunks)
        
        # Remove very short chunks (noise)
        chunks = [c for c in chunks if len(c.strip()) > 20]
        
        print(f"✂️ Split into {len(chunks)} chunks (min 20 chars)")
        return chunks
    
    def build_index(self, data_folder: str = "data") -> None:
        """
        Build complete RAG index from documents.
        
        Args:
            data_folder: Path to data folder
        """
        print("\n🔨 Building RAG Index...")
        
        # Load documents
        documents = self.load_documents(data_folder)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        self.chunks = chunks
        
        # Create metadata for each chunk
        metadatas = [
            {"chunk_id": i, "text_preview": chunk[:50]}
            for i, chunk in enumerate(chunks)
        ]
        
        # Build vector index (includes deduplication)
        self.vector_store.create_index(chunks, metadatas)
        
        print(f"\n✅ RAG Index Ready!")
        print(f"  📊 Documents: {len(documents)}")
        print(f"  📝 Chunks: {len(chunks)}")
        print(f"  ⚡ Optimizations: Deduplication, Caching Enabled")
    
    def retrieve_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """
        Retrieve minimal relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve (default: 3, not 5)
            
        Returns:
            Concatenated relevant context
        """
        # Search vector store
        results = self.vector_store.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        # Get compressed context
        context = self.vector_store.get_compressed_context(results)
        
        # Truncate if too long (save tokens!)
        if len(context) > MAX_CONTEXT_TOKENS:
            context = context[:MAX_CONTEXT_TOKENS] + "..."
        
        return context
    
    def _construct_minimal_prompt(self, query: str, context: str = "") -> str:
        """
        Construct minimal prompt to save tokens.
        
        Args:
            query: User query
            context: Retrieved context (optional)
            
        Returns:
            Minimal prompt string
        """
        if context:
            return f"System: {SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            return f"System: {SYSTEM_PROMPT}\n\nQuestion: {query}\n\nAnswer:"
    
    def generate_answer(self, query: str, context: str = "") -> str:
        """
        Generate answer using Groq LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        # Construct minimal prompt
        prompt = self._construct_minimal_prompt(query, context)
        
        try:
            # Get response from Groq
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def query(self, user_query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Complete token-optimized RAG query pipeline.
        
        Args:
            user_query: User's question
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dict with query, context, answer, and metadata
        """
        # Validate index
        if not self.vector_store.is_indexed():
            raise RuntimeError("Index not built. Call build_index() first.")
        
        try:
            # Step 1: Classify query
            query_type = self._classify_query(user_query)
            
            # Step 2: Handle greeting (no API call needed!)
            if query_type == "greeting":
                return {
                    "query": user_query,
                    "answer": GREETING_RESPONSE,
                    "query_type": "greeting",
                    "context_chunks": 0,
                    "status": "success"
                }
            
            # Step 3: Check cache
            cached = self._is_cached_response(user_query)
            if cached:
                return {
                    "query": user_query,
                    "answer": cached,
                    "query_type": query_type,
                    "context_chunks": 0,
                    "cached": True,
                    "status": "success"
                }
            
            # Step 4: Retrieve context (only for profile questions)
            context = ""
            context_chunks = 0
            
            if query_type == "profile_question":
                context = self.retrieve_context(user_query, top_k=top_k)
                context_chunks = len(context.split("\n\n")) if context else 0
            else:
                context = self.retrieve_context(user_query, top_k=top_k)
                context_chunks = len(context.split("\n\n")) if context else 0
            
            # Step 5: Generate answer
            answer = self.generate_answer(user_query, context)
            
            # Step 6: Cache response
            self.semantic_cache.put(user_query, answer)
            
            # Step 7: Log token usage
            self._log_token_usage(user_query, context, answer, query_type)
            
            return {
                "query": user_query,
                "context_chunks": context_chunks,
                "query_type": query_type,
                "answer": answer,
                "status": "success"
            }
        
        except Exception as e:
            return {
                "query": user_query,
                "answer": f"❌ Error: {str(e)}",
                "status": "error"
            }
    
    def get_token_usage_stats(self) -> Dict:
        """Get token usage statistics."""
        if not self.token_usage_log:
            return {"total_queries": 0, "total_tokens": 0}
        
        total_tokens = sum(log["total_tokens"] for log in self.token_usage_log)
        cache_stats = self.semantic_cache.get_stats()
        
        return {
            "total_queries": len(self.token_usage_log),
            "total_tokens": total_tokens,
            "avg_tokens_per_query": total_tokens / len(self.token_usage_log) if self.token_usage_log else 0,
            "semantic_cache": cache_stats
        }
    
    def save_index(self, path: str = ".vector_store") -> None:
        """Save vector store to disk."""
        self.vector_store.save(path)
    
    def load_index(self, path: str = ".vector_store") -> None:
        """Load vector store from disk."""
        self.vector_store.load(path)
