"""프롬프트 템플릿 시스템 테스트."""

import pytest

from src.retrieval.prompts import PromptConfig, PromptManager, QueryType


class TestQueryType:
    """QueryType enum 테스트."""

    def test_all_values_exist(self):
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.ANALYTICAL.value == "analytical"
        assert QueryType.COMPARATIVE.value == "comparative"
        assert QueryType.HOW_TO.value == "how_to"
        assert QueryType.DEFAULT.value == "default"

    def test_from_string(self):
        assert QueryType("factual") == QueryType.FACTUAL
        assert QueryType("analytical") == QueryType.ANALYTICAL

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            QueryType("invalid_type")


class TestPromptConfig:
    """PromptConfig dataclass 테스트."""

    def test_format_prompt_basic(self):
        config = PromptConfig(
            system_prompt="시스템 프롬프트",
            user_template="[문서]\n{context}\n\n[질문]\n{query}",
            require_citation=False,
            chain_of_thought=False,
        )
        result = config.format_prompt(query="테스트 질문", context="테스트 문서")
        assert "시스템 프롬프트" in result
        assert "테스트 질문" in result
        assert "테스트 문서" in result

    def test_format_prompt_with_citation(self):
        config = PromptConfig(
            system_prompt="시스템",
            user_template="{context}\n{query}",
            require_citation=True,
        )
        result = config.format_prompt(query="q", context="c")
        assert "[출처]" in result

    def test_format_prompt_without_citation(self):
        config = PromptConfig(
            system_prompt="시스템",
            user_template="{context}\n{query}",
            require_citation=False,
        )
        result = config.format_prompt(query="q", context="c")
        assert "[출처]" not in result

    def test_format_prompt_with_cot(self):
        config = PromptConfig(
            system_prompt="시스템",
            user_template="{context}\n{query}",
            chain_of_thought=True,
            require_citation=False,
        )
        result = config.format_prompt(query="q", context="c")
        assert "단계별로 생각" in result

    def test_format_prompt_with_few_shot(self):
        config = PromptConfig(
            system_prompt="시스템",
            user_template="{context}\n{query}",
            require_citation=False,
            few_shot_examples=[
                {"question": "예시 질문", "answer": "예시 답변"},
            ],
        )
        result = config.format_prompt(query="q", context="c")
        assert "예시 질문" in result
        assert "예시 답변" in result
        assert "[예시]" in result


class TestPromptManager:
    """PromptManager 클래스 테스트."""

    def setup_method(self):
        self.manager = PromptManager()

    def test_get_config_default_when_none(self):
        config = self.manager.get_config(None)
        assert config.query_type == QueryType.DEFAULT

    def test_get_config_by_enum(self):
        config = self.manager.get_config(QueryType.FACTUAL)
        assert config.query_type == QueryType.FACTUAL

    def test_get_config_by_string(self):
        config = self.manager.get_config("analytical")
        assert config.query_type == QueryType.ANALYTICAL

    def test_get_config_invalid_string_returns_default(self):
        config = self.manager.get_config("nonexistent")
        assert config.query_type == QueryType.DEFAULT

    def test_build_prompt_contains_query_and_context(self):
        prompt = self.manager.build_prompt(
            query="Milvus란?",
            context="Milvus는 벡터 데이터베이스입니다.",
        )
        assert "Milvus란?" in prompt
        assert "Milvus는 벡터 데이터베이스입니다." in prompt

    def test_build_prompt_with_query_type(self):
        prompt = self.manager.build_prompt(
            query="q",
            context="c",
            query_type="how_to",
        )
        # HOW_TO has chain_of_thought=True
        assert "단계별" in prompt

    def test_build_prompt_factual_no_cot(self):
        prompt = self.manager.build_prompt(
            query="q",
            context="c",
            query_type="factual",
        )
        # FACTUAL has chain_of_thought=False
        assert "단계별로 생각" not in prompt

    def test_build_prompt_comparative_has_few_shot(self):
        prompt = self.manager.build_prompt(
            query="A vs B?",
            context="문서 내용",
            query_type="comparative",
        )
        assert "HNSW" in prompt  # few-shot example contains HNSW

    def test_list_types(self):
        types = self.manager.list_types()
        assert "factual" in types
        assert "analytical" in types
        assert "comparative" in types
        assert "how_to" in types
        assert "default" in types
        assert len(types) == 5

    def test_custom_templates(self):
        custom = {
            QueryType.DEFAULT: PromptConfig(
                system_prompt="커스텀",
                user_template="{context} {query}",
                require_citation=False,
            ),
        }
        manager = PromptManager(templates=custom)
        config = manager.get_config(QueryType.DEFAULT)
        assert config.system_prompt == "커스텀"


class TestPromptManagerIntegration:
    """API 연동 관점의 통합 테스트."""

    @pytest.mark.asyncio
    async def test_query_endpoint_accepts_prompt_type(self, client):
        """prompt_type 파라미터가 API에서 수용되는지 확인."""
        from unittest.mock import MagicMock, patch

        mock_rag = MagicMock()
        mock_rag.answer.return_value = {"answer": "test", "sources": []}
        mock_module = MagicMock()
        mock_module.RAGPipeline.return_value = mock_rag

        with patch.dict("sys.modules", {"src.retrieval.rag": mock_module}):
            resp = await client.post(
                "/query",
                json={"query": "테스트", "prompt_type": "factual"},
            )
        assert resp.status_code == 200
        mock_rag.answer.assert_called_once_with("테스트", top_k=5, query_type="factual")

    @pytest.mark.asyncio
    async def test_query_endpoint_accepts_null_prompt_type(self, client):
        """prompt_type=null이어도 정상 동작."""
        from unittest.mock import MagicMock, patch

        mock_rag = MagicMock()
        mock_rag.answer.return_value = {"answer": "test", "sources": []}
        mock_module = MagicMock()
        mock_module.RAGPipeline.return_value = mock_rag

        with patch.dict("sys.modules", {"src.retrieval.rag": mock_module}):
            resp = await client.post(
                "/query",
                json={"query": "테스트", "prompt_type": None},
            )
        assert resp.status_code == 200
        mock_rag.answer.assert_called_once_with("테스트", top_k=5, query_type=None)

    @pytest.mark.asyncio
    async def test_query_endpoint_without_prompt_type(self, client):
        """prompt_type 없이도 기존처럼 동작."""
        from unittest.mock import MagicMock, patch

        mock_rag = MagicMock()
        mock_rag.answer.return_value = {"answer": "test", "sources": []}
        mock_module = MagicMock()
        mock_module.RAGPipeline.return_value = mock_rag

        with patch.dict("sys.modules", {"src.retrieval.rag": mock_module}):
            resp = await client.post(
                "/query",
                json={"query": "테스트"},
            )
        assert resp.status_code == 200
        mock_rag.answer.assert_called_once_with("테스트", top_k=5, query_type=None)
