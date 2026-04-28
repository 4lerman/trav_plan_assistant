import pytest
from unittest.mock import patch
from rag.retriever import retrieve, NoFeasibleResultsError
from models.profile import MobilityLevel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant(mocker):
    return mocker.patch("rag.retriever._client")


@pytest.fixture
def mock_embed(mocker):
    embed = mocker.patch("rag.retriever._embed_model")
    # encode() returns dict with dense_vecs; each vec needs .tolist() (numpy array API)
    import numpy as np; embed.encode.return_value = {"dense_vecs": [np.array([0.1] * 1024)]}
    return embed


def _make_scored_point(doc_id, description="some place", score=0.9):
    from qdrant_client.models import ScoredPoint
    return ScoredPoint(
        id=doc_id, version=1, score=score,
        payload={"doc_id": doc_id, "description": description},
    )


class _Record:
    def __init__(self, payload):
        self.payload = payload


# ---------------------------------------------------------------------------
# Hard constraint filter — verified by inspecting the Filter object built
# ---------------------------------------------------------------------------

def test_hard_filter_blocks_inaccessible_docs(sample_profile, mock_qdrant, mock_embed):
    """MobilityLevel.FULL must add wheelchair + step_free conditions to the Qdrant filter."""
    # Return nothing from both search and scroll so we hit NoFeasibleResultsError
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)

    with pytest.raises(NoFeasibleResultsError):
        retrieve("halal restaurant", sample_profile, "req_1")

    # Inspect the Filter passed to client.scroll (called before search, always fires)
    call_kwargs = mock_qdrant.scroll.call_args[1]
    q_filter = call_kwargs["scroll_filter"]
    keys = [cond.key for cond in q_filter.must if hasattr(cond, "key")]
    assert "accessibility.wheelchair_accessible" in keys
    assert "accessibility.step_free_routes" in keys


def test_budget_filter_is_included(sample_profile, mock_qdrant, mock_embed):  # noqa: ARG001
    """avg_cost_per_person range condition must always be present in the filter."""
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)

    with pytest.raises(NoFeasibleResultsError):
        retrieve("hotel", sample_profile, "req_budget")

    call_kwargs = mock_qdrant.scroll.call_args[1]
    q_filter = call_kwargs["scroll_filter"]
    keys = [cond.key for cond in q_filter.must if hasattr(cond, "key")]
    assert "avg_cost_per_person" in keys


# ---------------------------------------------------------------------------
# Empty result → NoFeasibleResultsError
# ---------------------------------------------------------------------------

def test_empty_result_raises_no_feasible_error(sample_profile, mock_qdrant, mock_embed):  # noqa: ARG001
    """scroll returning empty list must raise NoFeasibleResultsError with correct query."""
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)

    with pytest.raises(NoFeasibleResultsError) as exc:
        retrieve("impossible query", sample_profile, "req_2")

    assert "impossible query" in str(exc.value)
    assert exc.value.profile is sample_profile


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def test_rrf_fusion_weights_both_signals(sample_profile, mock_qdrant, mock_embed):
    """Docs appearing in both dense and BM25 results should have higher RRF scores."""
    p1 = _make_scored_point("doc_1", "a great place halal", score=0.9)
    p2 = _make_scored_point("doc_2", "another place", score=0.5)

    mock_qdrant.search.return_value = [p1, p2]
    # doc_1 appears in both scroll results (double signal), doc_2 only in scroll
    mock_qdrant.scroll.return_value = (
        [_Record(p1.payload), _Record(p2.payload)], None
    )

    results = retrieve("great place halal", sample_profile, "req_3")

    assert len(results) == 2
    assert "rrf_score" in results[0]
    # doc_1 ranks first: it appears first in dense AND BM25 matches "great place halal"
    assert results[0]["doc_id"] == "doc_1"


def test_results_ordered_by_rrf_score_descending(sample_profile, mock_qdrant, mock_embed):  # noqa: ARG001
    """Results must be sorted by rrf_score descending."""
    p1 = _make_scored_point("doc_1", "halal Paris restaurant", score=0.9)
    p2 = _make_scored_point("doc_2", "museum visit", score=0.7)

    mock_qdrant.search.return_value = [p1, p2]
    mock_qdrant.scroll.return_value = (
        [_Record(p1.payload), _Record(p2.payload)], None
    )

    results = retrieve("halal Paris restaurant", sample_profile, "req_order")
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_results_carry_rrf_score(sample_profile, mock_qdrant, mock_embed):  # noqa: ARG001
    """Every result must have a positive numeric rrf_score field."""
    p = _make_scored_point("doc_1", "Paris halal")
    mock_qdrant.search.return_value = [p]
    mock_qdrant.scroll.return_value = ([_Record(p.payload)], None)

    results = retrieve("halal Paris", sample_profile, "req_score")
    assert len(results) > 0
    for doc in results:
        assert isinstance(doc["rrf_score"], float)
        assert doc["rrf_score"] > 0


# ---------------------------------------------------------------------------
# Reranker (no Qdrant needed)
# ---------------------------------------------------------------------------

def test_reranker_reorders_candidates():
    """Cross-encoder scores must determine final ordering."""
    from rag.reranker import rerank
    candidates = [
        {"doc_id": "1", "description": "poor match"},
        {"doc_id": "2", "description": "perfect match for query"},
    ]
    with patch("rag.reranker._reranker.predict") as mock_score:
        mock_score.return_value = [-1.0, 5.0]
        ranked = rerank("perfect match", candidates, top_n=2)

    assert ranked[0]["doc_id"] == "2"
    assert ranked[1]["doc_id"] == "1"
    assert "rerank_score" in ranked[0]


def test_reranker_preserves_original_fields():
    """Reranker must not drop any fields from the input dict."""
    from rag.reranker import rerank
    candidates = [{"doc_id": "x", "description": "test", "rrf_score": 0.5, "name": "X"}]
    with patch("rag.reranker._reranker.predict") as mock_score:
        mock_score.return_value = [1.0]
        result = rerank("test", candidates, top_n=1)

    assert result[0]["doc_id"] == "x"
    assert result[0]["rrf_score"] == 0.5
    assert result[0]["name"] == "X"


def test_reranker_empty_input():
    from rag.reranker import rerank
    assert rerank("query", []) == []


def test_reranker_top_n_respected():
    from rag.reranker import rerank
    candidates = [{"doc_id": str(i), "description": f"place {i}"} for i in range(8)]
    with patch("rag.reranker._reranker.predict") as mock_score:
        mock_score.return_value = list(range(8))
        result = rerank("query", candidates, top_n=3)
    assert len(result) == 3
