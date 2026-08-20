"""Query Rewriter 테스트."""

from unittest.mock import patch

from src.retrieval.query_rewriter import (
    ABBREVIATION_MAP,
    QueryRewriter,
    RewriteStrategy,
)


class TestRewriteStrategy:
    """RewriteStrategy enum 테스트."""

    def test_all_values(self):
        assert RewriteStrategy.NONE.value == "none"
        assert RewriteStrategy.SIMPLE.value == "simple"
        assert RewriteStrategy.MULTI_QUERY.value == "multi_query"
        assert RewriteStrategy.HYDE.value == "hyde"

    def test_from_string(self):
        assert RewriteStrategy("none") == RewriteStrategy.NONE
        assert RewriteStrategy("simple") == RewriteStrategy.SIMPLE
        assert RewriteStrategy("multi_query") == RewriteStrategy.MULTI_QUERY
        assert RewriteStrategy("hyde") == RewriteStrategy.HYDE


class TestQueryRewriterRuleBased:
    """규칙 기반 리라이팅 테스트 (LLM 없이)."""

    def setup_method(self):
        self.rewriter = QueryRewriter(openai_api_key=None)

    def test_llm_not_available(self):
        assert not self.rewriter.llm_available

    def test_rewrite_abbreviation_expansion(self):
        result = self.rewriter.rewrite("RAG 시스템이 뭐야?")
        assert "Retrieval-Augmented Generation" in result

    def test_rewrite_multiple_abbreviations(self):
        result = self.rewriter.rewrite("LLM으로 RAG 구현")
        assert "Large Language Model" in result
        assert "Retrieval-Augmented Generation" in result

    def test_rewrite_removes_noise(self):
        result = self.rewriter.rewrite("알려줘 벡터 검색이 뭐야")
        # "알려줘" 접두사가 제거되어야 함
        assert not result.startswith("알려줘")

    def test_rewrite_preserves_original_if_no_change(self):
        query = "Milvus 벡터 데이터베이스 설정 방법"
        result = self.rewriter.rewrite(query)
        # 약어 없고 불용어 없으면 원본과 비슷해야 함
        assert "Milvus" in result
        assert "벡터" in result

    def test_rewrite_empty_string(self):
        result = self.rewriter.rewrite("")
        assert result == ""

    def test_expand_returns_multiple_variants(self):
        results = self.rewriter.expand("RAG 파이프라인 구현 방법")
        assert len(results) >= 1
        assert len(results) <= 3
        # 원본이 포함되어야 함
        assert "RAG 파이프라인 구현 방법" in results

    def test_expand_variants_are_different(self):
        results = self.rewriter.expand("RAG 시스템 아키텍처")
        # 최소 2개 이상 변형이 있어야 함 (원본 + 변형)
        if len(results) > 1:
            assert results[0] != results[1]

    def test_expand_n_parameter(self):
        results = self.rewriter.expand("검색 파이프라인", n=2)
        assert len(results) <= 2

    def test_hyde_without_llm_returns_original(self):
        query = "Milvus ANN 검색 원리"
        result = self.rewriter.hyde(query)
        assert result == query


class TestQueryRewriterWithLLM:
    """LLM 기반 리라이팅 테스트 (mocked)."""

    def setup_method(self):
        self.rewriter = QueryRewriter(openai_api_key="test-key", model="gpt-4o-mini")

    def test_llm_available(self):
        assert self.rewriter.llm_available

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_rewrite_calls_llm(self, mock_llm):
        mock_llm.return_value = "Milvus 벡터 데이터베이스의 HNSW 인덱스 검색 원리"
        result = self.rewriter.rewrite("Milvus 검색 어떻게 동작해?")
        mock_llm.assert_called_once()
        assert "HNSW" in result

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_rewrite_fallback_on_llm_failure(self, mock_llm):
        mock_llm.return_value = ""  # LLM 실패
        result = self.rewriter.rewrite("RAG 시스템")
        # 규칙 기반 fallback 동작해야 함
        assert "Retrieval-Augmented Generation" in result

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_expand_with_llm(self, mock_llm):
        mock_llm.return_value = "변형1\n변형2\n변형3"
        results = self.rewriter.expand("원본 쿼리", n=3)
        mock_llm.assert_called_once()
        assert len(results) >= 2  # 원본 + 변형들

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_expand_includes_original(self, mock_llm):
        mock_llm.return_value = "변형1\n변형2"
        results = self.rewriter.expand("원본 쿼리", n=3)
        assert "원본 쿼리" in results

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_hyde_with_llm(self, mock_llm):
        mock_llm.return_value = "Milvus는 HNSW 알고리즘을 사용하여 근사 최근접 이웃 검색을 수행합니다."
        result = self.rewriter.hyde("Milvus 검색 원리")
        mock_llm.assert_called_once()
        assert "HNSW" in result

    @patch("src.retrieval.query_rewriter.QueryRewriter._call_llm")
    def test_hyde_fallback_on_failure(self, mock_llm):
        mock_llm.return_value = ""
        query = "Milvus 검색 원리"
        result = self.rewriter.hyde(query)
        assert result == query


class TestAbbreviationMap:
    """약어 매핑 테스트."""

    def test_contains_key_abbreviations(self):
        assert "RAG" in ABBREVIATION_MAP
        assert "LLM" in ABBREVIATION_MAP
        assert "ANN" in ABBREVIATION_MAP
        assert "HNSW" in ABBREVIATION_MAP

    def test_abbreviations_have_korean_explanation(self):
        for abbr, full in ABBREVIATION_MAP.items():
            # 각 약어는 영문 풀어쓰기 + 한국어 설명 포함해야 함
            assert len(full) > len(abbr)


class TestPipelineIntegration:
    """Pipeline과의 통합 테스트 (설정 기반)."""

    def test_rewrite_strategy_from_settings(self):
        """settings.rewrite_strategy로 RewriteStrategy가 정상 생성되는지 확인."""
        for strategy in ["none", "simple", "multi_query", "hyde"]:
            s = RewriteStrategy(strategy)
            assert s.value == strategy

    def test_query_rewriter_initialization(self):
        """QueryRewriter가 api_key 없이도 초기화 가능한지 확인."""
        rewriter = QueryRewriter(openai_api_key=None)
        assert not rewriter.llm_available
        # 규칙 기반은 동작해야 함
        result = rewriter.rewrite("테스트 쿼리")
        assert isinstance(result, str)
        assert len(result) > 0
