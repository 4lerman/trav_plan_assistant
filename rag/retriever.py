from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, MatchValue, MatchAny, Range, Filter
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

from models.profile import ConstraintProfile, MobilityLevel

class NoFeasibleResultsError(Exception):
    def __init__(self, profile: ConstraintProfile, query: str):
        self.profile = profile
        self.query = query
        super().__init__(f"No feasible results found for query '{query}' matching the constraints.")

_client = QdrantClient("http://localhost:6333")
_embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def retrieve(query: str, profile: ConstraintProfile, request_id: str, top_k: int = 50) -> list[dict]:
    must = []
    if profile.mobility_level == MobilityLevel.FULL:
        must.append(FieldCondition(key="accessibility.wheelchair_accessible", match=MatchValue(value=True)))
        must.append(FieldCondition(key="accessibility.step_free_routes", match=MatchValue(value=True)))
    
    if profile.dietary_tags:
        must.append(FieldCondition(key="dietary_tags", match=MatchAny(any=profile.dietary_tags)))
        
    must.append(FieldCondition(key="avg_cost_per_person", range=Range(lte=float(profile.daily_budget))))
    
    qdrant_filter = Filter(must=must)
    
    dense_vec = _embed_model.encode([query], return_dense=True)["dense_vecs"][0]
    
    dense_hits = _client.search(
        collection_name="destinations",
        query_vector=dense_vec.tolist(),
        query_filter=qdrant_filter,
        limit=top_k,
    )
    
    records, _ = _client.scroll(
        collection_name="destinations",
        scroll_filter=qdrant_filter,
        limit=10000,
        with_payload=True
    )
    
    if not records:
        raise NoFeasibleResultsError(profile=profile, query=query)
        
    corpus_tokens = [doc.payload["description"].split() for doc in records]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query.split())
    
    bm25_ranked = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    rrf_scores = {}
    docs_by_id = {}
    
    for rank, hit in enumerate(dense_hits):
        doc_id = hit.payload["doc_id"]
        rrf_scores[doc_id] = _rrf(rank + 1)
        docs_by_id[doc_id] = hit.payload
        
    for rank, (rec, bm25_score) in enumerate(bm25_ranked):
        doc_id = rec.payload["doc_id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            docs_by_id[doc_id] = rec.payload
        rrf_scores[doc_id] += _rrf(rank + 1)
        
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for doc_id, rrf_score in fused:
        doc = docs_by_id[doc_id].copy()
        doc["rrf_score"] = rrf_score
        results.append(doc)
        
    if not results:
        raise NoFeasibleResultsError(profile=profile, query=query)
        
    return results
