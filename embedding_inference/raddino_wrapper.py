import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor


class RadDINOWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output


def build_raddino(device):
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(repo)
    return RadDINOWrapper(model, processor)
