"""
Stage 3: Matching — cosine similarity between query embedding dan
enrolled templates.

Module ini di-reuse di backend (backend/ml/matcher.py = copy file ini).

Multi-sample matching strategy:
- Setiap user punya 5 templates (dari enrollment) → list of 128-dim
- Untuk query, hitung cosine sim ke SEMUA template milik user X
- Ambil rata-rata top-3 similarity tertinggi sebagai user_score
- Lakukan untuk semua user, pilih user dengan score tertinggi
- Kalau best_score >= threshold → IDENTIFIED
- Kalau < threshold → UNKNOWN

Kenapa top-3 avg? Lebih robust dari max:
- Max bisa "lucky match" dengan satu template outlier
- Avg semua bisa drag down kalau ada template berkualitas rendah
- Top-3 balance: stabil tapi tetap respect best matches
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Import sebagai constants supaya bisa di-tweak per use case
from config import DEFAULT_THRESHOLD, MATCH_TOP_K


@dataclass
class MatchResult:
    """Hasil matching."""
    status: str  # "identified" atau "unknown"
    user_id: Optional[int]
    user_name: Optional[str]
    score: float
    # Optional: untuk debugging
    runner_up_user_id: Optional[int] = None
    runner_up_score: Optional[float] = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity antara dua vector. Untuk vector yang sudah
    L2-normalized, ini setara dengan dot product.

    Args:
        a, b: 1D arrays shape (D,)

    Returns:
        float dalam range [-1, 1]
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def identify(
    query_embedding: np.ndarray,
    enrolled_db: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = MATCH_TOP_K,
) -> MatchResult:
    """
    Multi-sample matching.

    Args:
        query_embedding: shape (128,), L2-normalized (preferably)
        enrolled_db: list of dict, tiap dict:
            {
                "user_id": int,
                "user_name": str,
                "embeddings": list of np.ndarray shape (128,)
            }
        threshold: kalau best_score < threshold, return "unknown"
        top_k: ambil avg top-K similarity per user

    Returns:
        MatchResult
    """
    if not enrolled_db:
        return MatchResult("unknown", None, None, 0.0)

    user_scores: list[tuple[float, int, str]] = []

    for user in enrolled_db:
        user_embs = user.get("embeddings", [])
        if not user_embs:
            continue

        sims = [cosine_similarity(query_embedding, emb) for emb in user_embs]
        sims.sort(reverse=True)
        k = min(top_k, len(sims))
        score = float(sum(sims[:k]) / k)

        user_scores.append((score, user["user_id"], user["user_name"]))

    if not user_scores:
        return MatchResult("unknown", None, None, 0.0)

    # Sort descending
    user_scores.sort(reverse=True, key=lambda x: x[0])
    best = user_scores[0]
    runner_up = user_scores[1] if len(user_scores) > 1 else None

    if best[0] >= threshold:
        return MatchResult(
            status="identified",
            user_id=best[1],
            user_name=best[2],
            score=best[0],
            runner_up_user_id=runner_up[1] if runner_up else None,
            runner_up_score=runner_up[0] if runner_up else None,
        )
    return MatchResult(
        status="unknown",
        user_id=None,
        user_name=None,
        score=best[0],  # tetap return best score untuk debugging
        runner_up_user_id=runner_up[1] if runner_up else None,
        runner_up_score=runner_up[0] if runner_up else None,
    )


def batch_identify(
    query_embeddings: np.ndarray,  # (N, 128)
    enrolled_db: list[dict],
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = MATCH_TOP_K,
) -> list[MatchResult]:
    """
    Batch identify — optimized untuk banyak query sekaligus.
    Untuk single query, pakai `identify()` saja (lebih simple).
    """
    results = []
    for q in query_embeddings:
        results.append(identify(q, enrolled_db, threshold, top_k))
    return results


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)

    def _rand_unit(d=128):
        v = np.random.randn(d).astype(np.float32)
        return v / np.linalg.norm(v)

    # Simulasi: 3 user, masing-masing 5 template
    user_a_center = _rand_unit()
    user_b_center = _rand_unit()
    user_c_center = _rand_unit()

    def _near(center, noise=0.1):
        v = center + noise * np.random.randn(128).astype(np.float32)
        return v / np.linalg.norm(v)

    enrolled = [
        {"user_id": 1, "user_name": "Alice", "embeddings": [_near(user_a_center) for _ in range(5)]},
        {"user_id": 2, "user_name": "Bob",   "embeddings": [_near(user_b_center) for _ in range(5)]},
        {"user_id": 3, "user_name": "Carol", "embeddings": [_near(user_c_center) for _ in range(5)]},
    ]

    # Query: dekat ke user A
    query = _near(user_a_center, noise=0.15)
    result = identify(query, enrolled, threshold=0.5)
    print(f"Query dekat Alice → {result}")

    # Query: random (unknown)
    unknown_query = _rand_unit()
    result = identify(unknown_query, enrolled, threshold=0.5)
    print(f"Query random → {result}")

    # Threshold lebih longgar
    result = identify(unknown_query, enrolled, threshold=0.0)
    print(f"Query random, threshold=0 → {result}")
