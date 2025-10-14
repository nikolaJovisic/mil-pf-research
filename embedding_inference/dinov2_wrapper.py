import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DinoV2(nn.Module):
    def __init__(self, device):
        super().__init__()
        model_name = "dinov2_vitb14"
        self.device = torch.device(device)
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device).eval()

        
    def forward(self, images):
        inputs = torch.stack(images).to(self.device)

        with torch.inference_mode():
            outputs = self.model(inputs)
        
        return outputs

def build_model(device):
    return DinoV2(device)
