# Rerank Report — Module 8 Thursday Stretch

> ~250 words

## Setup

- Hybrid `k_in`: 50  
- Re-ranked `k_out`: 5  
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
- Hardware (CPU, RAM, OS): Windows 10, CPU-based execution (no GPU), ~16GB RAM  

---

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (baseline) | 0.85 | 0.6661 | ~low (single-stage retrieval only) |
| Hybrid + cross-encoder rerank | 0.7833 | 0.6228 | higher (hybrid + rerank stage) |

The hybrid model performs better overall in both recall@5 and MRR compared to the reranked pipeline. This indicates that in this dataset, the cross-encoder reordering does not consistently improve top-ranked retrieval quality.

---

## When Does Re-Ranking Pay Off?

Re-ranking shows mixed results across query types:

- For **factoid queries**, performance remains stable:
  - recall@5 = 0.9667 (unchanged after rerank)
  - MRR = 0.9111 → 0.9083 (slight drop)

- For **paraphrastic queries**, performance drops significantly:
  - recall@5 = 0.7333 → 0.6000  
  - MRR = 0.4210 → 0.3372  

This suggests the cross-encoder is not reliably improving semantic reordering in more flexible language queries, and may be over-ranking semantically similar but incorrect candidates.

---

## Latency Overhead

The cross-encoder adds a significant computation cost because it evaluates **50 query–document pairs per request**.

- Hybrid retrieval is relatively fast (single vector + BM25/hybrid search)
- Cross-encoder reranking adds a second stage of inference over all candidates

The overhead is approximately linear in `k_in`, since each query requires scoring 50 pairs regardless of corpus size.

In practice:
- Hybrid stage scales with corpus size (vector search complexity)
- Cross-encoder stage scales with `k_in`, not corpus size

Thus, reranking is stable in cost per query but expensive per request.

---

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

The cross-encoder becomes a bottleneck at high query volume (QPS):

- If one rerank call takes ~O(50 × encoder inference time), latency becomes the limiting factor.
- At scale (e.g., high traffic systems), this cost accumulates linearly with QPS.

Estimated trade-off:
- Low QPS systems (<10 QPS): reranking is acceptable for quality gains
- Medium/high QPS systems (>50 QPS): reranking becomes expensive and requires GPU acceleration or caching
- Large corpora do not directly increase rerank cost, but they increase hybrid retrieval cost and candidate diversity pressure

In real production systems, reranking is typically used only as a **second-stage refinement on a very small candidate set (10–20 instead of 50)** or replaced with lighter models.

---

## Conclusion

Cross-encoder reranking improves ranking quality in some cases (especially factoid queries), but in this setup it decreases overall recall@5 and MRR while significantly increasing latency. The results suggest that hybrid retrieval alone provides a better cost–benefit balance for this dataset unless further optimization (smaller k_in, GPU inference, or smarter candidate filtering) is applied.