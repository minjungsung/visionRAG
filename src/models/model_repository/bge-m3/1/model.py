"""BGE-M3 텍스트 임베딩 모델."""
import numpy as np
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer


class TritonPythonModel:
    def initialize(self, args):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = [
                t.decode("utf-8")
                for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().flatten()
            ]
            embeddings = self.model.encode(texts, normalize_embeddings=True).astype(np.float32)
            out = pb_utils.Tensor("embedding", embeddings)
            responses.append(pb_utils.InferenceResponse([out]))
        return responses

    def finalize(self):
        pass
