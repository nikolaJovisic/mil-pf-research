import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/nikola.jovisic.ivi/nj/Mammo-CLIP/src/codebase/')
from breastclip.model.modules import load_image_encoder
from breastclip.data.data_utils import load_tokenizer
from breastclip.model import build_model as build_mammoclip


class MammoCLIPEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)

        ckpt_path = "/lustre/mammo_clip/model_ckpt/b2-model-best-epoch-10.tar"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt["config"]

        ckpt_config["tokenizer"]['cache_dir'] = './outputs/huggingface/tokenizers'
        ckpt_config["model"]['text_encoder']['cache_dir'] = './outputs/huggingface/'

        tokenizer = load_tokenizer(**ckpt_config["tokenizer"])
        model = build_mammoclip(
            model_config=ckpt_config["model"],
            loss_config=ckpt_config["loss"],
            tokenizer=tokenizer
        ).to(self.device)

        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        self.model = model

        # normalization constants
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean = 0.3089279
        self.std = 0.25053555408335154

        # target image size
        self.target_size = (912, 1520)

    def forward(self, imgs):
        # resize to 1520x912
        imgs = F.interpolate(imgs, size=self.target_size, mode='bilinear', align_corners=False)

        # denormalize from ImageNet
        imgs = imgs * self.imagenet_std.to(imgs.device) + self.imagenet_mean.to(imgs.device)

        # normalize with custom stats
        imgs = (imgs - self.mean) / self.std

        with torch.no_grad():
            img_emb = self.model.encode_image(imgs)
            img_emb = self.model.image_projection(img_emb) if self.model.projection else img_emb
        return img_emb.detach().cpu()


def build_model(device):
    return MammoCLIPEncoder(device)
