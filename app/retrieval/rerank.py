from sentence_transformers import CrossEncoder

# This model is specifically trained to score how well a (question, passage)
# pair match - much more precise than the fast search used in hybrid_search,
# but too slow to run against every chunk, which is why it only runs on
# the ~10 candidates hybrid_search already narrowed things down to.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidate_texts: list[str], top_k: int = 5) -> list[str]:
    """
    Takes a question and a list of candidate chunk texts (already narrowed
    down by hybrid search), and returns the best few, reordered by how
    well each one actually answers the question.
    """
    # The model scores each (question, chunk) pair together
    pairs = [(query, text) for text in candidate_texts]
    scores = reranker.predict(pairs)

    # Sort candidates by score, highest first, and keep only the top ones
    ranked = sorted(zip(candidate_texts, scores), key=lambda pair: pair[1], reverse=True)
    return [text for text, score in ranked[:top_k]]
