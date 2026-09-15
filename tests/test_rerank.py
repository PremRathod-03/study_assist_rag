import chromadb
from chromadb.utils import embedding_functions
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection(
    name="reinforcement_learning",
    embedding_function=embedding_function,
)

query = "What is the exploration-exploitation tradeoff?"

# Step 1: hybrid search gets a broader shortlist of candidate chunk IDs
candidate_ids = hybrid_search("reinforcement_learning", query, collection, top_k=10)
candidates = collection.get(ids=candidate_ids)["documents"]

print(f"Hybrid search returned {len(candidates)} candidates. Reranking...")

# Step 2: reranker picks the true best few from those candidates
top_chunks = rerank(query, candidates, top_k=3)

for i, chunk in enumerate(top_chunks):
    print(f"--- reranked #{i+1} ---")
    print(chunk[:250])
    print()
