import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DinoV3(nn.Module):
    def __init__(self, device):
        super().__init__()
        model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name, do_resize=False)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return outputs.pooler_output

def build_model(device):
    return DinoV3(device)
