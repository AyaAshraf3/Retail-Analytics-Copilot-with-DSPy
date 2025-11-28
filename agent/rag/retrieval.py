"""
Document Retriever using BM25
- Chunk management with IDs and sources
- Top-K retrieval for RAG
"""

import os
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    def __init__(self, chunk_id: str, content: str, source: str, chunk_index: int = 0):
        self.chunk_id = chunk_id  
        self.content = content
        self.source = source 
        self.chunk_index = chunk_index
        self.score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "score": self.score
        }


class DocumentRetriever:
    """BM25-based document retriever for Northwind documentation."""
    
    def __init__(self, doc_dir: str = "docs", chunk_size: int = 500):
        """
        Initialize retriever.
        
        Args:
            doc_dir: Directory containing markdown documents
            chunk_size: Character size per chunk (approximate)
        """
        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunks: List[DocumentChunk] = []
        self.bm25 = None
        self.tokenized_chunks = []
        self.load_documents()
    
    def load_documents(self):
        """Load and chunk all documents from directory."""
        logger.info(f"Loading documents from {self.doc_dir}")
        
        if not os.path.exists(self.doc_dir):
            logger.warning(f"Document directory not found: {self.doc_dir}")
            return
        
        for filename in os.listdir(self.doc_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.doc_dir, filename)
                self._load_file(filepath, filename)
        
        # Build BM25 index
        self._build_bm25_index()
        logger.info(f"Loaded {len(self.chunks)} chunks from {len(set(c.source for c in self.chunks))} documents")
    
    def _load_file(self, filepath: str, filename: str):
        """Load and chunk a single markdown file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by sections (marked by ##) or by size
            chunks = self._chunk_content(content, filename)
            self.chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
    
    def _chunk_content(self, content: str, source: str) -> List[DocumentChunk]:
        """Split content into chunks with IDs."""
        chunks = []
        
        # Split by markdown headers (##) for semantic chunking
        sections = re.split(r'\n## ', content)
        
        chunk_index = 0
        for section in sections:
            if not section.strip():
                continue
            
            # If section is large, split further
            if len(section) > self.chunk_size:
                sub_chunks = self._split_by_size(section, self.chunk_size)
                for sub_chunk in sub_chunks:
                    chunk_id = f"{source}::chunk{chunk_index}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=sub_chunk.strip(),
                        source=source,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                chunk_id = f"{source}::chunk{chunk_index}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=section.strip(),
                    source=source,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _split_by_size(self, text: str, size: int) -> List[str]:
        """Split text into chunks of approximately given size."""
        chunks = []
        current_chunk = ""
        
        sentences = text.split('. ')
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks."""
        if not self.chunks:
            logger.warning("No chunks to index")
            return
        
        # Tokenize chunks
        self.tokenized_chunks = [
            self._tokenize(chunk.content) for chunk in self.chunks
        ]
        
        # Build BM25
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        logger.info("BM25 index built")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split by whitespace, remove special chars."""
        text = text.lower()
        # Remove punctuation but keep common separators
        text = re.sub(r'[^\w\s-]', ' ', text)
        tokens = text.split()
        return tokens
    
    def retrieve(self, query: str, topk: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks using BM25.
        
        Args:
            query: User query string
            topk: Number of top results to return
        
        Returns:
            List of dicts with id, content, source, score
        """
        if not self.bm25 or not self.chunks:
            logger.warning("Retriever not initialized or no chunks loaded")
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk.to_dict())
        
        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        return results
    
    def retrieve_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks from a specific source document."""
        chunks = [c for c in self.chunks if c.source == source]
        return [c.to_dict() for c in chunks]