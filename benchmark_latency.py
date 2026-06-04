from evaluation import evaluate_retriever
from rerank import rerank_search
from retrieval_helpers import hybrid_search
import weaviate
from sentence_transformers import SentenceTransformer

client = weaviate.Client("http://localhost:8080")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("\n=== HYBRID ===")
hybrid = evaluate_retriever(
    "data/retrieval_eval.jsonl",
    lambda q, k: hybrid_search(client, q, k, embedder)
)
print(hybrid)

print("\n=== RERANK ===")
rerank = evaluate_retriever(
    "data/retrieval_eval.jsonl",
    lambda q, k: rerank_search(client, q, embedder, k_in=50, k_out=5)
)
print(rerank)