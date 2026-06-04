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
|---|---:|---:|---:|
| Hybrid (lab baseline) | 0.8500 | 0.6661 | Not separately measured |
| Hybrid + cross-encoder rerank | 0.7833 | 0.6228 | 4337.57 |

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

The measured average latency for the rerank pipeline was 4337.57 ms per query
(approximately 4.34 seconds).

This overhead comes primarily from the cross-encoder stage, which scores 50
query-document pairs for every query. Unlike the hybrid retrieval stage,
which retrieves candidates efficiently from Weaviate, the cross-encoder must
perform transformer inference on each candidate pair.

The overhead scales approximately linearly with k_in. Doubling k_in from 50
to 100 would roughly double the amount of cross-encoder work. In contrast,
the rerank cost is largely independent of corpus size because it only processes
the retrieved candidates, while the hybrid retrieval stage becomes slower as
the corpus grows.

---

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

With an average latency of 4337.57 ms per query, a single CPU worker could
handle only about 0.23 queries per second.

For low-volume applications where retrieval quality is critical, this cost may
be acceptable. However, for interactive production systems serving many users,
the cross-encoder becomes the bottleneck very quickly.

At moderate traffic levels (10+ QPS), multiple workers or GPU acceleration
would be required. At higher traffic levels (50+ QPS), a full cross-encoder
rerank stage becomes impractical without aggressive caching, batching, or a
smaller candidate set.

For larger corpora, the hybrid retrieval stage will also become slower,
further increasing end-to-end latency. In such cases, lighter rerankers or
learned retrieval approaches may provide a better cost-performance tradeoff.

---

## Conclusion

Cross-encoder reranking improves ranking quality in some cases (especially factoid queries), but in this setup it decreases overall recall@5 and MRR while significantly increasing latency. The results suggest that hybrid retrieval alone provides a better cost–benefit balance for this dataset unless further optimization (smaller k_in, GPU inference, or smarter candidate filtering) is applied.