import chromadb
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection(
    name="reinforcement_learning",
    embedding_function=embedding_function,
)

results = collection.query(
    query_texts=["What is a value function?"],
    n_results=3,
)

for i, doc in enumerate(results["documents"][0]):
    print(f"--- match {i+1} ---")
    print(doc)
    print()
