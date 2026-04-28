import pytest
from unittest.mock import patch, MagicMock
from rag.retriever import retrieve, NoFeasibleResultsError
from models.profile import MobilityLevel

@pytest.fixture
def mock_qdrant(mocker):
    return mocker.patch("rag.retriever._client")

@pytest.fixture
def mock_embed(mocker):
    embed = mocker.patch("rag.retriever._embed_model")
    embed.encode.return_value = {"dense_vecs": [[0.1] * 1024]}
    return embed

def test_hard_filter_blocks_inaccessible_docs(sample_profile, mock_qdrant, mock_embed):
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)
    
    with pytest.raises(NoFeasibleResultsError):
        retrieve("halal restaurant", sample_profile, "req_1")
        
    call_kwargs = mock_qdrant.search.call_args[1]
    q_filter = call_kwargs["query_filter"]
    must_conditions = q_filter.must
    keys = [cond.key for cond in must_conditions if hasattr(cond, "key")]
    assert "accessibility.wheelchair_accessible" in keys
    assert "accessibility.step_free_routes" in keys

def test_empty_result_raises_no_feasible_error(sample_profile, mock_qdrant, mock_embed):
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)
    
    with pytest.raises(NoFeasibleResultsError) as exc:
        retrieve("impossible query", sample_profile, "req_2")
        
    assert "impossible query" in str(exc.value)

def test_rrf_fusion_weights_both_signals(sample_profile, mock_qdrant, mock_embed):
    from qdrant_client.models import ScoredPoint
    
    p1 = ScoredPoint(id="1", version=1, score=0.9, payload={"doc_id": "doc_1", "description": "a great place"})
    p2 = ScoredPoint(id="2", version=1, score=0.8, payload={"doc_id": "doc_2", "description": "another place"})
    mock_qdrant.search.return_value = [p1, p2]
    
    class MockRecord:
        def __init__(self, payload):
            self.payload = payload
    
    r1 = MockRecord(p1.payload)
    r2 = MockRecord({"doc_id": "doc_2", "description": "exactly matching another place"})
    mock_qdrant.scroll.return_value = ([r1, r2], None)
    
    results = retrieve("matching", sample_profile, "req_3")
    
    assert len(results) == 2
    assert "rrf_score" in results[0]

def test_reranker_reorders_candidates():
    from rag.reranker import rerank
    candidates = [
        {"doc_id": "1", "description": "poor match"},
        {"doc_id": "2", "description": "perfect match for query"},
    ]
    with patch("rag.reranker._reranker.compute_score") as mock_score:
        mock_score.return_value = [-1.0, 5.0]
        
        ranked = rerank("perfect match", candidates, top_n=2)
        assert ranked[0]["doc_id"] == "2"
        assert ranked[1]["doc_id"] == "1"
        assert "rerank_score" in ranked[0]
