"""
Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking

Now includes:
- Cross-encoder re-ranking
- Efficient batch retrieval from Weaviate
- Latency tracking (required for evaluation)
"""

from __future__ import annotations

import time
import weaviate
from sentence_transformers import CrossEncoder

from retrieval_helpers import hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLASS_NAME = "Post"

# Load once globally
ce = CrossEncoder(CROSS_ENCODER_MODEL)


# =========================
# CROSS ENCODER RERANKING
# =========================

def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    k_out: int = 5
) -> list[str]:
    """Re-rank candidates using cross-encoder."""
    
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)

    scored = [
        (c["doc_id"], float(score))
        for c, score in zip(candidates, scores)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in scored[:k_out]]


# =========================
# MAIN RERANK PIPELINE
# =========================

def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> dict:
    """
    Two-stage retrieval:
    1. Hybrid retrieval (fast)
    2. Batch fetch from Weaviate (IMPORTANT FIX)
    3. Cross-encoder reranking

    Returns:
        {
            "results": [doc_id],
            "timing": {
                "hybrid_ms": ...,
                "fetch_ms": ...,
                "rerank_ms": ...,
                "total_ms": ...
            }
        }
    """

    t0 = time.perf_counter()

    # -------------------------
    # Stage 1: Hybrid retrieval
    # -------------------------
    t1 = time.perf_counter()

    candidate_ids = hybrid_search(
        client, query, k=k_in, embedder=embedder, alpha=0.5
    )

    t2 = time.perf_counter()

    # -----------------------------------
    # Stage 2: Batch fetch from Weaviate
    # (FIX: avoids N queries loop problem)
    # -----------------------------------
    candidates = []

    if candidate_ids:
        response = (
            client.query.get(CLASS_NAME, ["doc_id", "text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "ContainsAny",
                "valueText": candidate_ids
            })
            .do()
        )

        rows = (
            response.get("data", {})
            .get("Get", {})
            .get(CLASS_NAME, [])
        )

        # map for fast lookup
        row_map = {r["doc_id"]: r["text"] for r in rows}

        for doc_id in candidate_ids:
            if doc_id in row_map:
                candidates.append({
                    "doc_id": doc_id,
                    "text": row_map[doc_id]
                })

    t3 = time.perf_counter()

    # -------------------------
    # Stage 3: Cross encoder
    # -------------------------
    results = cross_encoder_rerank(
        query=query,
        candidates=candidates,
        k_out=k_out
    )

    t4 = time.perf_counter()

    return {
        "results": results,
        "timing": {
            "hybrid_ms": (t2 - t1) * 1000,
            "fetch_ms": (t3 - t2) * 1000,
            "rerank_ms": (t4 - t3) * 1000,
            "total_ms": (t4 - t0) * 1000,
        }
    }