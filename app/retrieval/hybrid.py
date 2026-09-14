import pickle


def reciprocal_rank_fusion(vector_ids: list, bm25_ids: list, k: int = 60) -> list:
    """
    Combines two separately-ranked lists of chunk IDs into one ranking.
    A chunk that ranks well in BOTH lists ends up ranked highest overall.
    'k' is a smoothing constant (60 is the standard default from the
    original RRF research) - it just softens how much a #1 rank versus
    a #5 rank matters, so one list doesn't completely dominate the other.
    """
    scores = {}

    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # sort all seen chunk ids by their combined score, highest first
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(subject: str, query: str, chroma_collection, top_k: int = 10) -> list:
    """
    Runs both a keyword search (BM25) and a meaning search (vector/Chroma)
    for the same query, then fuses the two rankings together.
    Returns a list of chunk IDs, best match first.
    """
    # --- Keyword search (BM25) ---
    with open(f"bm25_index/{subject}.pkl", "rb") as f:
        bm25, bm25_ids = pickle.load(f)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:top_k]
    bm25_ranked_ids = [bm25_ids[i] for i in ranked_indices]

    # --- Meaning search (vector, via Chroma) ---
    vector_results = chroma_collection.query(query_texts=[query], n_results=top_k)
    vector_ranked_ids = vector_results["ids"][0]

    # --- Combine both rankings ---
    fused_ids = reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids)
    return fused_ids[:top_k]
