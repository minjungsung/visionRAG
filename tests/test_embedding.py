"""Embedding model tests including Triton/local fallback logic."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.embedding import EmbeddingModel


class TestEmbeddingModelInit:
    """Test EmbeddingModel initialization and lazy loading."""

    def test_model_lazy_initialization(self):
        """Models are not loaded until first use."""
        model = EmbeddingModel()
        assert model._text_model is None
        assert model._image_model is None
        assert model._triton_client is None

    def test_text_model_loaded_on_property_access(self):
        """text_model property triggers lazy loading of SentenceTransformer."""
        mock_st_cls = MagicMock()
        mock_st_cls.return_value = MagicMock()

        model = EmbeddingModel()
        with patch(
            "src.models.embedding.SentenceTransformer",
            mock_st_cls,
            create=True,
        ):
            # Access via the property forces the import inside the property body
            with patch.dict(
                sys.modules,
                {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)},
            ):
                _ = model.text_model

        mock_st_cls.assert_called_once_with("BAAI/bge-m3")

    def test_text_model_cached_after_first_access(self):
        """text_model is only loaded once, subsequent accesses return cached instance."""
        mock_st_cls = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_cls.return_value = mock_st_instance

        model = EmbeddingModel()
        with patch.dict(
            sys.modules,
            {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)},
        ):
            first = model.text_model
            second = model.text_model

        assert first is second
        mock_st_cls.assert_called_once()


class TestEmbeddingModelEncode:
    """Test text encoding functionality."""

    def test_encode_uses_local_model_when_triton_disabled(self):
        """When use_triton=False, encode() uses local SentenceTransformer."""
        fake_embeddings = np.random.rand(2, 1024).astype(np.float32)

        model = EmbeddingModel()
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = fake_embeddings
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            result = model.encode(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 1024)
        mock_text_model.encode.assert_called_once_with(
            ["hello", "world"], normalize_embeddings=True
        )

    def test_encode_returns_float32(self):
        """encode() always returns float32 array."""
        # Even if model returns float64, encode should cast to float32
        fake_embeddings = np.random.rand(1, 1024).astype(np.float64)

        model = EmbeddingModel()
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = fake_embeddings
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            result = model.encode(["test"])

        assert result.dtype == np.float32

    def test_encode_single_text(self):
        """encode() works with a single text input."""
        fake_embedding = np.random.rand(1, 1024).astype(np.float32)

        model = EmbeddingModel()
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = fake_embedding
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            result = model.encode(["single text"])

        assert result.shape == (1, 1024)

    def test_encode_empty_list(self):
        """encode() handles empty input list gracefully."""
        fake_result = np.array([], dtype=np.float32).reshape(0, 1024)

        model = EmbeddingModel()
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = fake_result
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            result = model.encode([])

        assert result.shape[0] == 0


class TestTritonFallback:
    """Test Triton → local model fallback behavior."""

    def test_encode_falls_back_to_local_when_triton_fails(self):
        """When Triton encoding raises, encode() falls back to local model."""
        model = EmbeddingModel()

        # Set up local model as fallback
        fake_local_result = np.random.rand(1, 1024).astype(np.float32)
        mock_text_model = MagicMock()
        mock_text_model.encode.return_value = fake_local_result
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = True
            mock_settings.text_embed_model = "bge-m3"

            # Mock the Triton encode to fail
            with patch.object(
                model, "_triton_encode_text", side_effect=ConnectionError("No Triton")
            ):
                result = model.encode(["test"])

        # Should fall back to local model
        assert result.shape == (1, 1024)
        mock_text_model.encode.assert_called_once()

    def test_encode_uses_triton_when_available(self):
        """When use_triton=True and server is live, Triton is used."""
        model = EmbeddingModel()
        fake_triton_result = np.random.rand(1, 1024).astype(np.float32)

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = True

            with patch.object(model, "_triton_encode_text", return_value=fake_triton_result):
                result = model.encode(["test"])

        assert np.array_equal(result, fake_triton_result)

    def test_triton_client_raises_on_dead_server(self):
        """triton_client property raises when server is not live."""
        model = EmbeddingModel()

        mock_grpc = MagicMock()
        mock_client = MagicMock()
        mock_client.is_server_live.return_value = False
        mock_grpc.InferenceServerClient.return_value = mock_client

        # The import `import tritonclient.grpc as grpcclient` resolves via
        # sys.modules['tritonclient'].grpc, so we need to wire the parent mock.
        mock_tritonclient = MagicMock()
        mock_tritonclient.grpc = mock_grpc

        original_tc = sys.modules.get("tritonclient")
        original_grpc = sys.modules.get("tritonclient.grpc")
        sys.modules["tritonclient"] = mock_tritonclient
        sys.modules["tritonclient.grpc"] = mock_grpc
        try:
            with patch("src.models.embedding.settings") as mock_settings:
                mock_settings.triton_url = "localhost:8001"
                with pytest.raises(ConnectionError, match="not live"):
                    _ = model.triton_client
        finally:
            if original_tc is not None:
                sys.modules["tritonclient"] = original_tc
            if original_grpc is not None:
                sys.modules["tritonclient.grpc"] = original_grpc

    def test_triton_client_lazy_initialization(self):
        """triton_client is None until first access."""
        model = EmbeddingModel()
        assert model._triton_client is None


class TestEmbeddingModelImage:
    """Test image encoding."""

    def test_encode_image_uses_local_when_triton_disabled(self):
        """encode_image() uses local model when use_triton=False."""
        model = EmbeddingModel()
        fake_embeddings = np.random.rand(1, 1152).astype(np.float32)

        mock_image_model = MagicMock()
        mock_image_model.encode.return_value = fake_embeddings
        model._image_model = mock_image_model
        model._image_preprocess = None  # sentence-transformers path

        mock_image = MagicMock()

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            result = model.encode_image([mock_image])

        assert result.shape == (1, 1152)

    def test_encode_image_falls_back_on_triton_failure(self):
        """encode_image() falls back to local model when Triton fails."""
        model = EmbeddingModel()
        fake_embeddings = np.random.rand(1, 1152).astype(np.float32)

        mock_image_model = MagicMock()
        mock_image_model.encode.return_value = fake_embeddings
        model._image_model = mock_image_model
        model._image_preprocess = None

        mock_image = MagicMock()

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = True

            with patch.object(
                model,
                "_triton_encode_image",
                side_effect=ConnectionError("Triton down"),
            ):
                result = model.encode_image([mock_image])

        # Should fall back to local
        assert result.shape == (1, 1152)
        mock_image_model.encode.assert_called_once()


class TestEmbeddingModelResilience:
    """Test resilience and error handling."""

    def test_encode_failure_does_not_corrupt_state(self):
        """If encode fails, subsequent calls can still succeed."""
        model = EmbeddingModel()
        mock_text_model = MagicMock()
        mock_text_model.encode.side_effect = [
            RuntimeError("GPU OOM"),
            np.random.rand(1, 1024).astype(np.float32),
        ]
        model._text_model = mock_text_model

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False

            with pytest.raises(RuntimeError, match="GPU OOM"):
                model.encode(["first attempt"])

            # Second call should work
            result = model.encode(["second attempt"])
            assert result.shape == (1, 1024)

    def test_model_load_failure_propagates(self):
        """If SentenceTransformer can't be loaded, error propagates."""
        model = EmbeddingModel()

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = OSError("Model not found")

        with patch("src.models.embedding.settings") as mock_settings:
            mock_settings.use_triton = False
            with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
                with pytest.raises(OSError, match="Model not found"):
                    model.encode(["test"])
