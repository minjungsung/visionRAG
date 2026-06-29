import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import io


class TritonPythonModel:
    def initialize(self, args):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt = (
                pb_utils.get_input_tensor_by_name(request, "prompt")
                .as_numpy()
                .flatten()[0]
                .decode("utf-8")
            )
            image_bytes = pb_utils.get_input_tensor_by_name(request, "image").as_numpy().tobytes()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=512)
            result = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

            out_tensor = pb_utils.Tensor("response", np.array([result.encode("utf-8")]))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def finalize(self):
        pass