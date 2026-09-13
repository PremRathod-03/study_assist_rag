import os
import chromadb
from chromadb.utils import embedding_functions

from app.loaders import load_file
from app.chunking import chunk_pages

# Better free, local embedding model than the small default one -
# stronger retrieval quality, still zero API cost.
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)

# PersistentClient saves Chroma's database to disk in this folder,
# so the knowledge base survives between runs.
chroma_client = chromadb.PersistentClient(path="vectorstore")


def ingest_subject(subject_name: str):
    """
    Reads every supported file inside data/raw_notes/<subject_name>/,
    chunks it, and stores the chunks into that subject's own Chroma
    collection - creating the collection if it doesn't exist yet.
    """
    folder = f"data/raw_notes/{subject_name}"

    collection = chroma_client.get_or_create_collection(
        name=subject_name,
        embedding_function=embedding_function,
    )

    chunk_id = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if not filename.endswith((".pdf", ".pptx", ".docx")):
            continue

        print(f"Loading {filename}...")
        pages = load_file(file_path)
        chunks = chunk_pages(pages)

        for chunk in chunks:
            unique_id = f"{filename}_{chunk_id}"

            collection.add(
                ids=[unique_id],
                documents=[chunk["text"]],
                metadatas=[{
                    "source_file": filename,
                    "source_type": chunk["source_type"],
                    "page": chunk["page"],
                }],
            )
            chunk_id += 1

    print(f"Ingested {chunk_id} chunks for subject: {subject_name}")


if __name__ == "__main__":
    ingest_subject("reinforcement_learning")
