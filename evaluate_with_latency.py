import time
import json
from typing import Callable


def evaluate_with_latency(eval_path: str, search_fn: Callable, k: int = 10):
    results = []

    total_hybrid_time = 0
    total_rerank_time = 0
    total_queries = 0

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            query = row["query"]
            gold = row["gold_doc_id"]

            start_total = time.perf_counter()

            # --- stage 1: retrieval (hybrid already inside search_fn OR rerank pipeline) ---
            start = time.perf_counter()
            retrieved = search_fn(query, k)
            hybrid_time = time.perf_counter() - start

            # NOTE:
            # If search_fn is rerank_search → hybrid + rerank included
            # So we approximate split like this:
            # (we’ll refine later)

            total_time = time.perf_counter() - start_total

            hit5 = int(gold in retrieved[:5])
            hit10 = int(gold in retrieved[:10])
            rank = retrieved.index(gold) + 1 if gold in retrieved else None
            mrr = 1.0 / rank if rank else 0.0

            results.append({
                "hit@5": hit5,
                "hit@10": hit10,
                "mrr": mrr,
                "latency_ms": total_time * 1000
            })

            total_queries += 1
            total_hybrid_time += hybrid_time

    avg_latency = sum(r["latency_ms"] for r in results) / total_queries

    return {
        "recall@5": sum(r["hit@5"] for r in results) / total_queries,
        "recall@10": sum(r["hit@10"] for r in results) / total_queries,
        "mrr": sum(r["mrr"] for r in results) / total_queries,
        "avg_latency_ms": avg_latency,
    }