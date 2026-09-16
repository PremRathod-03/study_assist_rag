import os
import chromadb
from chromadb.utils import embedding_functions

from app.loaders import load_file
from app.chunking import chunk_pages
from app.contextual_chunks import add_context

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
chroma_client = chromadb.PersistentClient(path="vectorstore")


def _file_already_ingested(collection, filename: str) -> bool:
    """
    Checks whether a file with this exact name has already been ingested
    into this collection, by looking at stored metadata. Prevents
    duplicate chunks if the same file gets ingested twice.
    """
    existing = collection.get(where={"source_file": filename})
    return len(existing["ids"]) > 0


def ingest_single_file(subject_name: str, file_path: str) -> int:
    """
    Loads, chunks, enriches, and stores ONE file into a subject's Chroma
    collection. Creates the collection if it doesn't exist yet. APPENDS
    to any existing data - never deletes anything already in the collection.
    Skips the file entirely if it's already been ingested (by filename),
    so re-running this on the same file is always safe.

    This is the building block both bulk folder ingestion AND the future
    upload-in-the-app feature will call - one file in, chunks added out.
    """
    filename = os.path.basename(file_path)

    collection = chroma_client.get_or_create_collection(
        name=subject_name,
        embedding_function=embedding_function,
    )

    if _file_already_ingested(collection, filename):
        print(f"  Skipping {filename} - already ingested for {subject_name}.")
        return 0

    print(f"Loading {filename}...")
    pages = load_file(file_path)
    chunks = chunk_pages(pages)
    full_document_text = "\n".join(page["text"] for page in pages if page.get("text"))

    existing_count = collection.count()

    added = 0
    for i, chunk in enumerate(chunks):
        print(f"  Enriching chunk {i}...")
        enriched_text = add_context(full_document_text, chunk["text"])

        unique_id = f"{filename}_{existing_count + i}"
        collection.add(
            ids=[unique_id],
            documents=[enriched_text],
            metadatas=[{
                "source_file": filename,
                "source_type": chunk["source_type"],
                "page": chunk["page"],
            }],
        )
        added += 1

    print(f"  Added {added} chunks from {filename}.")
    return added


def ingest_subject(subject_name: str):
    """
    Ingests every supported file inside data/raw_notes/<subject_name>/,
    calling ingest_single_file() for each. Safe to re-run - already-
    ingested files are automatically skipped, new files get added.
    """
    folder = f"data/raw_notes/{subject_name}"
    total_added = 0

    for filename in os.listdir(folder):
        if not filename.endswith((".pdf", ".pptx", ".docx")):
            continue
        file_path = os.path.join(folder, filename)
        total_added += ingest_single_file(subject_name, file_path)

    print(f"Done. {total_added} new chunks added for subject: {subject_name}")


if __name__ == "__main__":
    ingest_subject("reinforcement_learning")
