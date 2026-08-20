"""Embedding model with Triton inference fallback.

When settings.use_triton is True, embeddings are computed via gRPC calls to
NVIDIA Triton Inference Server. When False (default), local SentenceTransformer
/ open_clip models are used instead.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.config.settings import settings

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Unified text + image embedding interface with Triton/local backends."""

    def __init__(self) -> None:
        self._text_model = None
        self._image_model = None
        self._image_preprocess = None
        self._triton_client = None

    # ------------------------------------------------------------------
    # Triton gRPC client (lazy)
    # ------------------------------------------------------------------

    @property
    def triton_client(self):
        """Lazy-initialised Triton gRPC client."""
        if self._triton_client is None:
            try:
                import tritonclient.grpc as grpcclient

                self._triton_client = grpcclient.InferenceServerClient(
                    url=settings.triton_url
                )
                if not self._triton_client.is_server_live():
                    raise ConnectionError("Triton server is not live")
                logger.info("Connected to Triton at %s", settings.triton_url)
            except Exception as exc:
                logger.warning(
                    "Failed to connect to Triton (%s), falling back to local models",
                    exc,
                )
                self._triton_client = None
                raise
        return self._triton_client

    # ------------------------------------------------------------------
    # Local text model (lazy)
    # ------------------------------------------------------------------

    @property
    def text_model(self):
        """Lazy-loaded local SentenceTransformer for text embeddings."""
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer("BAAI/bge-m3")
            logger.info("Loaded local text model: BAAI/bge-m3")
        return self._text_model

    # ------------------------------------------------------------------
    # Local image model (lazy)
    # ------------------------------------------------------------------

    @property
    def image_model(self):
        """Lazy-loaded local SigLIP model for image embeddings."""
        if self._image_model is None:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-SO400M-14-SigLIP-384", pretrained="webli"
                )
                model.eval()
                self._image_model = model
                self._image_preprocess = preprocess
                logger.info("Loaded local image model: SigLIP via open_clip")
            except ImportError:
                from sentence_transformers import SentenceTransformer

                self._image_model = SentenceTransformer(
                    "google/siglip-so400m-patch14-384"
                )
                self._image_preprocess = None
                logger.info(
                    "Loaded local image model: SigLIP via sentence-transformers"
                )
        return self._image_model

    # ------------------------------------------------------------------
    # Triton inference helpers
    # ------------------------------------------------------------------

    def _triton_encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode texts via Triton gRPC."""
        import tritonclient.grpc as grpcclient

        input_data = np.array([[t] for t in texts], dtype=object)
        inp = grpcclient.InferInput("TEXT", input_data.shape, "BYTES")
        inp.set_data_from_numpy(input_data)

        output = grpcclient.InferRequestedOutput("EMBEDDINGS")
        result = self.triton_client.infer(
            model_name=settings.text_embed_model,
            inputs=[inp],
            outputs=[output],
        )
        embeddings = result.as_numpy("EMBEDDINGS").astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    def _triton_encode_image(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode images via Triton gRPC."""
        import tritonclient.grpc as grpcclient

        # Stack images into batch: (B, H, W, C) as uint8
        batch = np.stack(images, axis=0).astype(np.uint8)
        inp = grpcclient.InferInput("IMAGE", batch.shape, "UINT8")
        inp.set_data_from_numpy(batch)

        output = grpcclient.InferRequestedOutput("EMBEDDINGS")
        result = self.triton_client.infer(
            model_name=settings.image_embed_model,
            inputs=[inp],
            outputs=[output],
        )
        embeddings = result.as_numpy("EMBEDDINGS").astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Uses Triton when settings.use_triton is True and the server is
        reachable; falls back to local SentenceTransformer otherwise.
        """
        if settings.use_triton:
            try:
                return self._triton_encode_text(texts)
            except Exception as exc:
                logger.warning(
                    "Triton text encoding failed (%s), falling back to local model",
                    exc,
                )
        return self.text_model.encode(texts, normalize_embeddings=True).astype(
            np.float32
        )

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images to embeddings.

        Uses Triton when settings.use_triton is True and the server is
        reachable; falls back to local SigLIP model otherwise.

        Args:
            images: List of PIL Image objects.

        Returns:
            numpy array of shape (N, image_embed_dim) with normalized embeddings.
        """
        if settings.use_triton:
            try:
                # Convert PIL images to numpy for Triton
                np_images = [
                    np.array(img.convert("RGB").resize((384, 384))) for img in images
                ]
                return self._triton_encode_image(np_images)
            except Exception as exc:
                logger.warning(
                    "Triton image encoding failed (%s), falling back to local model",
                    exc,
                )

        # Local fallback
        import torch

        model = self.image_model

        if self._image_preprocess is not None:
            # open_clip path
            with torch.no_grad():
                batch = torch.stack([self._image_preprocess(img) for img in images])
                embeddings = model.encode_image(batch)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            # sentence-transformers path
            embeddings = model.encode(images).astype(np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype(np.float32)


embedding_model = EmbeddingModel()
