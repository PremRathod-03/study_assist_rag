import chromadb

chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection("reinforcement_learning")
print(f"Total chunks in reinforcement_learning: {collection.count()}")
