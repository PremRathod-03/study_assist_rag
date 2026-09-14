from app.loaders import load_file
from app.chunking import chunk_pages

pages=load_file("data/raw_notes/reinforcement_learning/unit1_r1.pdf")
chunks=chunk_pages(pages)
print("total chunks:", len(chunks))
print("chunk:", chunks[10])