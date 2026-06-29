"""SigLIP 이미지 임베딩 모델."""
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import SiglipModel, SiglipProcessor


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda"
        self.model = SiglipModel.from_pretrained("google/siglip-base-patch16-384").to(self.device)
        self.processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-384")

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            pixel_values = torch.tensor(images, dtype=torch.float32).to(self.device) / 255.0
            with torch.no_grad():
                embeddings = self.model.get_image_features(pixel_values=pixel_values)
            out = pb_utils.Tensor("embedding", embeddings.cpu().numpy().astype(np.float32))
            responses.append(pb_utils.InferenceResponse([out]))
        return responses

    def finalize(self):
        pass
