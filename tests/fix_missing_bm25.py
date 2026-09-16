from app.ingest import _rebuild_bm25_index

_rebuild_bm25_index("autoencoders_and_generative_ai")
_rebuild_bm25_index("reinforcement_learning")
print("BM25 indexes rebuilt for both subjects.")
