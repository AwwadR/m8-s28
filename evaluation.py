import json
from typing import Callable

def _empty_metrics() -> dict[str, float]:
    return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return _empty_metrics()
    
    total = len(rows)
    return {
        "recall@5": sum(row["hit@5"] for row in rows) / total,
        "recall@10": sum(row["hit@10"] for row in rows) / total,
        "mrr": sum(row["mrr"] for row in rows) / total,
    }

def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    """Evaluate a retriever against the labeled set.

    For each (query, gold_doc_id, query_type) row:
      - Call search_fn(query, k=max(k_values))  # one call per query
      - Compute hit@5 (gold in top-5) and hit@10 (gold in top-10)
      - Compute MRR contribution: 1/rank (1-indexed) if gold in top-10, else 0

    Return:
        {
          "recall@5": <mean hit@5>,
          "recall@10": <mean hit@10>,
          "mrr": <mean MRR>,
          "by_type": {  # REQUIRED — used in the comparison brief
            "factoid": {"recall@5": ..., "recall@10": ..., "mrr": ...},
            "paraphrastic": {"recall@5": ..., "recall@10": ..., "mrr": ...}
          }
        }
    """
    max_k = max(k_values)
    scored_rows = []
    by_type_rows = {"factoid": [], "paraphrastic": []}

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            returned_ids = search_fn(row["query"], k=max_k)
            gold_doc_id = row["gold_doc_id"]

            mrr = 0.0
            if gold_doc_id in returned_ids[:max_k]:
                mrr = 1.0 / (returned_ids.index(gold_doc_id) + 1)

            scored = {
                "hit@5": int(gold_doc_id in returned_ids[:5]),
                "hit@10": int(gold_doc_id in returned_ids[:10]),
                "mrr": mrr,
            }
            scored_rows.append(scored)
            by_type_rows.setdefault(row["query_type"], []).append(scored)

    metrics = _mean_metrics(scored_rows)
    metrics["by_type"] = {
        "factoid": _mean_metrics(by_type_rows.get("factoid", [])),
        "paraphrastic": _mean_metrics(by_type_rows.get("paraphrastic", [])),
    }
    return metrics