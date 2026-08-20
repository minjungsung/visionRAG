"""Search and query endpoint tests with mocked retrieval."""
import sys
from unittest.mock import MagicMock, patch

import pytest

MOCK_SEARCH_RESULTS = [
    {
        "text": "FastAPI는 현대적인 Python 웹 프레임워크입니다.",
        "score": 0.95,
        "doc_id": "doc-001",
        "page_num": 1,
    },
    {
        "text": "비동기 처리를 기본으로 지원합니다.",
        "score": 0.87,
        "doc_id": "doc-001",
        "page_num": 2,
    },
    {
        "text": "Pydantic을 활용한 데이터 검증이 내장되어 있습니다.",
        "score": 0.82,
        "doc_id": "doc-002",
        "page_num": 5,
    },
]


@pytest.fixture
def mock_retrieval_module():
    """Mock the src.retrieval.pipeline module for search endpoints."""
    mock_module = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.search.return_value = MOCK_SEARCH_RESULTS
    mock_module.RetrievalPipeline.return_value = mock_pipeline_instance

    with patch.dict(sys.modules, {"src.retrieval.pipeline": mock_module}):
        yield mock_pipeline_instance


@pytest.fixture
def mock_rag_module():
    """Mock the src.retrieval.rag module for query endpoints."""
    mock_module = MagicMock()
    mock_rag_instance = MagicMock()
    mock_rag_instance.answer.return_value = {
        "answer": "FastAPI는 현대적인 Python 웹 프레임워크입니다.",
        "sources": MOCK_SEARCH_RESULTS,
    }
    mock_module.RAGPipeline.return_value = mock_rag_instance

    with patch.dict(sys.modules, {"src.retrieval.rag": mock_module}):
        yield mock_rag_instance


# --- POST /search Tests ---


@pytest.mark.asyncio
async def test_search_returns_results(client, mock_retrieval_module):
    """POST /search returns search results."""
    resp = await client.post("/search", json={"query": "FastAPI란?", "top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["text"] == MOCK_SEARCH_RESULTS[0]["text"]
    assert data["results"][0]["score"] == MOCK_SEARCH_RESULTS[0]["score"]


@pytest.mark.asyncio
async def test_search_respects_top_k(client, mock_retrieval_module):
    """POST /search passes top_k to the retrieval pipeline."""
    await client.post("/search", json={"query": "test query", "top_k": 3})
    mock_retrieval_module.search.assert_called_once_with("test query", top_k=3)


@pytest.mark.asyncio
async def test_search_default_top_k(client, mock_retrieval_module):
    """POST /search uses default top_k=5 when not specified."""
    await client.post("/search", json={"query": "default top_k"})
    mock_retrieval_module.search.assert_called_once_with("default top_k", top_k=5)


@pytest.mark.asyncio
async def test_search_empty_results(client):
    """POST /search returns empty results when nothing matches."""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.search.return_value = []
    mock_module.RetrievalPipeline.return_value = mock_instance

    with patch.dict(sys.modules, {"src.retrieval.pipeline": mock_module}):
        resp = await client.post("/search", json={"query": "nonexistent topic"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


@pytest.mark.asyncio
async def test_search_missing_query_returns_422(client):
    """POST /search without query field returns validation error."""
    resp = await client.post("/search", json={"top_k": 5})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_invalid_body_returns_422(client):
    """POST /search with invalid JSON body returns validation error."""
    resp = await client.post(
        "/search", content=b"not json", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_result_structure(client, mock_retrieval_module):
    """POST /search results contain expected fields."""
    resp = await client.post("/search", json={"query": "test"})
    data = resp.json()
    result = data["results"][0]

    assert "text" in result
    assert "score" in result
    assert "doc_id" in result
    assert "page_num" in result
    assert isinstance(result["score"], float)


@pytest.mark.asyncio
async def test_search_large_top_k(client):
    """POST /search with large top_k still works."""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.search.return_value = MOCK_SEARCH_RESULTS[:1]
    mock_module.RetrievalPipeline.return_value = mock_instance

    with patch.dict(sys.modules, {"src.retrieval.pipeline": mock_module}):
        resp = await client.post("/search", json={"query": "test", "top_k": 100})

    assert resp.status_code == 200
    mock_instance.search.assert_called_once_with("test", top_k=100)


# --- POST /query Tests ---


@pytest.mark.asyncio
async def test_query_returns_answer_and_sources(client, mock_rag_module):
    """POST /query returns an answer with source documents."""
    resp = await client.post("/query", json={"query": "FastAPI의 특징은?", "top_k": 3})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


@pytest.mark.asyncio
async def test_query_passes_parameters(client, mock_rag_module):
    """POST /query passes query and top_k to RAGPipeline."""
    await client.post("/query", json={"query": "test query", "top_k": 3})
    mock_rag_module.answer.assert_called_once_with("test query", top_k=3)


@pytest.mark.asyncio
async def test_query_no_results_returns_fallback(client):
    """POST /query returns fallback message when no documents match."""
    mock_module = MagicMock()
    mock_rag_instance = MagicMock()
    mock_rag_instance.answer.return_value = {
        "answer": "관련 문서를 찾지 못했습니다.",
        "sources": [],
    }
    mock_module.RAGPipeline.return_value = mock_rag_instance

    with patch.dict(sys.modules, {"src.retrieval.rag": mock_module}):
        resp = await client.post("/query", json={"query": "completely unknown"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "관련 문서를 찾지 못했습니다."
    assert data["sources"] == []


@pytest.mark.asyncio
async def test_query_missing_query_returns_422(client):
    """POST /query without query field returns validation error."""
    resp = await client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_default_top_k(client, mock_rag_module):
    """POST /query uses default top_k=5 when not specified."""
    await client.post("/query", json={"query": "test"})
    mock_rag_module.answer.assert_called_once_with("test", top_k=5)


@pytest.mark.asyncio
async def test_query_response_contains_sources_list(client, mock_rag_module):
    """POST /query response sources is a list of dicts."""
    resp = await client.post("/query", json={"query": "test"})
    data = resp.json()

    assert isinstance(data["sources"], list)
    if data["sources"]:
        source = data["sources"][0]
        assert "text" in source
        assert "score" in source
        assert "doc_id" in source
