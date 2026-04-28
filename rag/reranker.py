from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list[dict], top_n: int = 10) -> list[dict]:
    """Cross-encoder reranking of retriever output. Returns top_n with rerank_score."""
    if not candidates:
        return []

    pairs = [(query, c["description"]) for c in candidates]
    scores = _reranker.predict(pairs)

    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{"rerank_score": float(score), **doc} for doc, score in ranked[:top_n]]
