import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModel

class MedSigLIP(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = "google/medsiglip-448"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def denormalize_imagenet(self, tensor):
        if tensor.min() < 0 or tensor.max() > 1:
            return (tensor * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
        return tensor.clamp(0, 1)

    def prepare_images(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = self.denormalize_imagenet(images)
                images = [to_pil_image(img.cpu()) for img in images]
            elif images.dim() == 3:
                images = [to_pil_image(self.denormalize_imagenet(images).cpu())]
            else:
                raise ValueError("Expected images to be 3D or 4D tensor with 3 channels.")
        return images

    def forward(self, images):
        images = self.prepare_images(images)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            vision_outputs = self.model.vision_model(**{k: v for k, v in inputs.items() if "pixel_values" in k})
            if hasattr(self.model, "visual_projection"):
                embeds = self.model.visual_projection(vision_outputs.pooler_output)
            else:
                embeds = vision_outputs.pooler_output  # already normalized feature
        return embeds

def build_model(device="cuda"):
    return MedSigLIP(device)
