import pickle
import chromadb
from rank_bm25 import BM25Okapi

# Connect to the same Chroma database ingest.py writes to
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection("reinforcement_learning")

# Pull every chunk already stored - no re-embedding, no LLM calls
data = collection.get()
ids = data["ids"]
documents = data["documents"]

print(f"Pulled {len(documents)} chunks from Chroma to index for keyword search.")

# BM25 needs each document tokenized (split into lowercase words)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Save the BM25 index + the ids in the same order, so later we can
# map a BM25 result back to which chunk it actually is
with open("bm25_index/reinforcement_learning.pkl", "wb") as f:
    pickle.dump((bm25, ids), f)

print("Saved BM25 index to bm25_index/reinforcement_learning.pkl")
