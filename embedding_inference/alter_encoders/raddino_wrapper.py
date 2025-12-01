import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel, AutoImageProcessor


class RadDINOWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _maybe_denormalize(self, img):
        if isinstance(img, torch.Tensor):
            # ensure it’s on CPU and detached
            img = img.detach().to("cpu")
            # detect normalization by range
            if img.min() < 0 or img.max() > 1:
                img = (img * self.std + self.mean).clamp(0, 1)
            return to_pil_image(img)
        return img

    def forward(self, images):
        # convert all to PIL, handling cuda tensors automatically
        images = [self._maybe_denormalize(img) for img in images]

        # process and run inference
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.pooler_output


def build_model(device):
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(repo)
    return RadDINOWrapper(model, processor)
