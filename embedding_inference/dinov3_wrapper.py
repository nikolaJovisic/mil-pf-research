import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DinoV3(nn.Module):
    def __init__(self, device, save_all):
        super().__init__()
        model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        self.device = torch.device(device)
        self.base_model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name, do_resize=False)
        self.save_all = save_all
        
    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.base_model(**inputs)

        pooler_output = outputs.pooler_output.unsqueeze(1)
        cls_ = outputs.last_hidden_state[:, 0:1, :]
        rest = outputs.last_hidden_state[:, 1:, :]

        if self.save_all:
            return torch.cat([pooler_output, cls_, rest], dim=1)
        return torch.cat([pooler_output, cls_], dim=1)


def build_model(device, cls_only):
    return DinoV3(device, cls_only)
