import time
import weaviate
from sentence_transformers import SentenceTransformer
from rerank import rerank_search

client = weaviate.Client("http://localhost:8080")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "how do I rebase a feature branch"

runs = 10
times = []

for _ in range(runs):
    start = time.perf_counter()

    rerank_search(
        client,
        query,
        embedder,
        k_in=50,
        k_out=5,
    )

    times.append((time.perf_counter() - start) * 1000)

print(f"Average rerank pipeline latency: {sum(times)/len(times):.2f} ms")
