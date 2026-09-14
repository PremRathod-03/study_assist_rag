import chromadb
from chromadb.utils import embedding_functions
from app.retrieval.hybrid import hybrid_search

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection(
    name="reinforcement_learning",
    embedding_function=embedding_function,
)  

query = "What is the exploration-exploitation tradeoff?"

fused_ids = hybrid_search("reinforcement_learning", query, collection)
print("Hybrid search results (chunk IDs, best match first):")
for chunk_id in fused_ids[:5]:
    result = collection.get(ids=[chunk_id])
    print(f"Chunk ID: {chunk_id}")
    print(f"Document: {result['documents'][0][:200]}")
    print()