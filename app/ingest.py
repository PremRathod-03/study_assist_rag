import os
import chromadb
from chromadb.utils import embedding_functions

from app.loaders import load_file
from app.chunking import chunk_pages
from app.contextual_chunks import add_context

#Local ,free embedding model - no API cost
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
# This tells Chroma which embedding model to use internally.
# It's a free, local model — no API key or internet call needed once downloaded.


# PersistentClient means Chroma saves its database to disk in this folder,
# so the knowledge base survives between runs (not wiped every time you restart).
chroma_client = chromadb.PersistentClient(path="vectorstore")


def ingest_subject(subject_name: str):
    """
 Reads every file inside data/raw_notes/<subject_name>/, chunks it,
    enriches each chunk with a situating context blurb, embeds it, and
    stores it into that subject's Chroma collection.

    If a collection for this subject already exists, it's deleted first
    and rebuilt fresh - this avoids ending up with both old (un-enriched)
    and new (enriched) versions of the same chunks sitting side by side.
    """
    folder = f"data/raw_notes/{subject_name}"
    #wipe any existing collection for a clean rebuild
    try:
        chroma_client.delete_collection(subject_name)
        print(f"Deleted existing collection for {subject_name},rebuilding fresh")
    except Exception:
        #no existing collection yet - nothing to delete, thats fine
        pass
    # get_or_create_collection: if this subject already has a collection,
    # reuse it (so re-running ingestion adds to it); otherwise make a new one.
    collection = chroma_client.get_or_create_collection(
        name=subject_name,
        embedding_function=embedding_function,
    )

    chunk_id = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # Skip anything that isn't a file we know how to read
        if not filename.endswith((".pdf", ".pptx", ".docx")):
            continue

        print(f"Loading {filename}...")
        pages = load_file(file_path)
        chunks = chunk_pages(pages)

        #Build one big string of the whole document's text, used to give
        # the LLm enough surrounding context when writing each chunk's brub

        full_document_text="\n".join(page["text"] for page in pages if page.get("text"))

        for chunk in chunks:
            print(f"Enriching chunk {chunk_id} with context blurb...")
            enriched_text = add_context(full_document_text, chunk["text"])
            # Each chunk needs a unique ID within the collection.
            # We include the filename so IDs stay unique even across
            # multiple files ingested into the same subject.
            unique_id = f"{filename}_{chunk_id}"

            collection.add(
                ids=[unique_id],
                documents=[enriched_text],
                metadatas=[{
                    "source_file": filename,
                    "source_type": chunk["source_type"],
                    "page": chunk["page"],
                }],
            )
            chunk_id += 1

    print(f"Ingested {chunk_id} chunks for subject: {subject_name}")


if __name__ == "__main__":
    # Running this file directly ingests just Reinforcement Learning for now.
    # We'll loop over all SUBJECTS once we have more than one subject's notes ready.
    ingest_subject("reinforcement_learning")