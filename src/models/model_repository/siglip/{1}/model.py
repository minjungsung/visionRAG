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
            # shape: (batch, 3, 384, 384) uint8 → PIL-like tensor
            pixel_values = torch.tensor(images, dtype=torch.float32).to(self.device) / 255.0
            with torch.no_grad():
                outputs = self.model.get_image_features(pixel_values=pixel_values)
            embeddings = outputs.cpu().numpy().astype(np.float32)
            out_tensor = pb_utils.Tensor("embedding", embeddings)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        pass
