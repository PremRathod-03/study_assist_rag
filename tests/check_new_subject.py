import chromadb

chroma_client = chromadb.PersistentClient(path="vectorstore")
for c in chroma_client.list_collections():
    print(c.name, "-", c.count(), "chunks")
