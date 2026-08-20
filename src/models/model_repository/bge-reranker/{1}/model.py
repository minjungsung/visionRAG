import numpy as np
import triton_python_backend_utils as pb_utils
from sentence_transformers import CrossEncoder


class TritonPythonModel:
    def initialize(self, args):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def execute(self, requests):
        responses = []
        for request in requests:
            queries = [
                q.decode("utf-8")
                for q in pb_utils.get_input_tensor_by_name(request, "query")
                .as_numpy()
                .flatten()
            ]
            passages = [
                p.decode("utf-8")
                for p in pb_utils.get_input_tensor_by_name(request, "passage")
                .as_numpy()
                .flatten()
            ]
            pairs = list(zip(queries, passages))
            scores = self.model.predict(pairs).astype(np.float32).reshape(-1, 1)
            out_tensor = pb_utils.Tensor("score", scores)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        pass
