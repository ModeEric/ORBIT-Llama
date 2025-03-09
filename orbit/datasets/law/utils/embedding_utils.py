"""
Law Embedding Utilities

This module provides tools for creating and working with embeddings
of law-related text.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import re

class LawEmbedding:
    """
    Specialized embedding utilities for law content.
    
    This class provides methods for embedding law content and
    computing similarities between legal concepts.
    """
    
    def __init__(self, embedding_path: Optional[str] = None):
        """
        Initialize the LawEmbedding.
        
        Args:
            embedding_path: Path to embedding model or vectors
        """
        self.embedding_path = embedding_path
        self.embedding_model = None
        
        # Load embedding model if provided
        if embedding_path and os.path.exists(embedding_path):
            self._load_embedding_model(embedding_path)
    
    def _load_embedding_model(self, path: str):
        """
        Load embedding model from file.
        
        Args:
            path: Path to embedding model
        """
        try:
            if path.endswith('.bin'):
                # Try to load as FastText model
                try:
                    import fasttext
                    self.embedding_model = fasttext.load_model(path)
                    print(f"Loaded FastText model from {path}")
                except ImportError:
                    print("FastText not installed. Install with: pip install fasttext")
                except Exception as e:
                    print(f"Error loading FastText model: {e}")
            else:
                # TODO: Add support for other embedding formats
                print(f"Unsupported embedding format: {path}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using the loaded embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        if not self.embedding_model:
            # Return zeros if no model is loaded
            return np.zeros(300)
        
        try:
            # Preprocess text
            processed_text = ' '.join(text.lower().split())
            
            # Get embedding
            if hasattr(self.embedding_model, 'get_sentence_vector'):
                # FastText model
                return self.embedding_model.get_sentence_vector(processed_text)
            else:
                # TODO: Add support for other embedding models
                return np.zeros(300)
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros(300)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2) 