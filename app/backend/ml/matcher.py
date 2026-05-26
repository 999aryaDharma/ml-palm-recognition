"""
Palm Matching Module
Matches extracted palm embeddings against stored templates
"""

import numpy as np
from typing import Optional, Tuple


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(embedding1, embedding2) / (norm1 * norm2)


def match_embedding(
    query_embedding: np.ndarray,
    user_embeddings: list[np.ndarray],
    top_k: int = 3
) -> Tuple[Optional[int], float]:
    """
    Match query embedding against user templates.
    
    Args:
        query_embedding: Query palm embedding
        user_embeddings: List of stored embeddings for a user
        top_k: Number of top matches to average
        
    Returns:
        Tuple of (user_id or None, match_score)
    """
    if not user_embeddings:
        return None, 0.0
    
    # Calculate similarity to each template
    similarities = [
        cosine_similarity(query_embedding, template)
        for template in user_embeddings
    ]
    
    # Average of top-k scores
    top_scores = sorted(similarities, reverse=True)[:top_k]
    avg_score = np.mean(top_scores)
    
    return None, avg_score
