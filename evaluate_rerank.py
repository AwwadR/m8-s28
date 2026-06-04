import weaviate

from sentence_transformers import SentenceTransformer

from evaluation import evaluate_retriever
from rerank import rerank_search
from retrieval_helpers import hybrid_search

client = weaviate.Client("http://localhost:8080")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("\n=== HYBRID BASELINE ===")

baseline = evaluate_retriever(
    eval_path="data/retrieval_eval.jsonl",
    search_fn=lambda query, k: hybrid_search(
        client,
        query,
        k,
        embedder,
        alpha=0.5,
    ),
)

print(baseline)

print("\n=== HYBRID + CROSS ENCODER ===")

rerank_results = evaluate_retriever(
    eval_path="data/retrieval_eval.jsonl",
    search_fn=lambda query, k: rerank_search(
        client,
        query,
        embedder,
        k_in=50,
        k_out=5,
    ),
)

print(rerank_results)