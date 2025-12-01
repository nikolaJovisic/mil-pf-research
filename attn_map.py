from mammo_datasets import MammoDataset, ReturnMode, siar, DatasetEnum, ConvertTo
import torch
from embedding_inference.dinov2_wrapper import build_model
from icecream import ic
import torch.nn as nn
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add, scatter_max
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np

import os 
from PIL import Image
import torchvision.transforms as T
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def get_imagenet_normalization():
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def _rgb_tensor_normalized(x, normalization):
    x = np.asarray(x)
    img = torch.from_numpy(np.stack([x]*3) / 255).float()
    mean = torch.tensor(normalization[0]).view(3, 1, 1)
    std = torch.tensor(normalization[1]).view(3, 1, 1)
    return (img - mean) / std

import inspect
from PIL import Image

save_dir_name = 'attn_visual_test'

def save_image_list():
    for idx, (images, tiles, _) in enumerate(MammoDataset(DatasetEnum.VINDR, labels=['BI-RADS 1'], return_mode=ReturnMode.BREAST_TILES_LABEL, convert_to=ConvertTo.PIL, tile_size=518, read_window=True)):
        print(f'img {idx} num of tiles: {len(tiles)}')
        save_dir = f'./{save_dir_name}/img_{str(idx)}'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/images', exist_ok=True)
        os.makedirs(f'{save_dir}/top_attn', exist_ok=True)

        for i, img in enumerate(images):
            img.save(f"{save_dir}/images/img_{i}.png")

        for j, img in enumerate(tiles):
            img.save(f"{save_dir}/images/tile_{j}.png")
            
#         if idx == 20:
#             break


class Trex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.global_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.gl_hidden_dim), nn.ReLU())
        self.global_proj_1 = nn.Sequential(nn.Linear(2 * cfg.gl_hidden_dim, cfg.gl_hidden_dim), nn.ReLU())

        self.local_proj_0 = nn.Sequential(nn.Linear(cfg.input_dim, 2 * cfg.lc_hidden_dim), nn.ReLU())
        self.local_proj_1 = nn.Sequential(nn.Linear(2 * cfg.lc_hidden_dim, cfg.lc_hidden_dim), nn.ReLU())

        self.k = nn.Linear(cfg.lc_hidden_dim, cfg.lc_hidden_dim)
        self.v = nn.Linear(cfg.lc_hidden_dim, cfg.lc_hidden_dim)

        self.latent = nn.Parameter(torch.randn(cfg.num_latents, cfg.lc_hidden_dim))

        fused_dim = cfg.lc_hidden_dim * cfg.num_latents + cfg.gl_hidden_dim

        if cfg.mlp_out:
            self.linear_out = nn.Sequential(
                nn.Linear(fused_dim, fused_dim// 2),
                nn.ReLU(),
                nn.Linear(fused_dim// 2, 1)
            )
        else:
            self.linear_out = nn.Linear(fused_dim, 1)

    def forward(self, x, group, instance_type, return_attn_maps=False):
        is_whole = instance_type == 0
        is_tile = instance_type == 1

        x_whole, group_whole = x[is_whole], group[is_whole]
        x_tile, group_tile = x[is_tile], group[is_tile]

        x_whole = self.global_proj_0(x_whole)
        x_whole = self.global_proj_1(x_whole)
        whole_agg = scatter_max(x_whole, group_whole, dim=0)[0]

        x_tile = self.local_proj_0(x_tile)
        x_tile = self.local_proj_1(x_tile)

        k = self.k(x_tile)
        v = self.v(x_tile)

        G = int(group_tile.max().item()) + 1

        scores = (k @ self.latent.t()) / (self.latent.size(-1) ** 0.5)
        attn = segment_softmax(scores, group_tile, num_nodes=G)
        if return_attn_maps:
            return attn.reshape(G, self.latent.size(0), -1).cpu().detach(), scores
        out_group = scatter_add(attn.unsqueeze(-1) * v.unsqueeze(1), group_tile, dim=0, dim_size=G)
        out_group = rearrange(out_group, 'g l d -> g (l d)')

        fused = torch.cat([whole_agg, out_group], dim=-1)
        out = self.linear_out(fused)
        return out

def load_cfg(cfg_path=None):
    if cfg_path is None:
        cfg_path = '/home/nikola.jovisic.ivi/nj/mammo_filter/head_training/config.yaml'
    cfg = OmegaConf.load(cfg_path)
    return cfg


embedding_model = build_model(device='cuda')
embedding_model.eval()

cfg = load_cfg()

state = torch.load(cfg.load_path, map_location='cpu')
head = Trex(cfg).to('cuda')
head.load_state_dict(torch.load(cfg.load_path, map_location='cuda'), strict=True)
head.eval()

def get_attn_scores(idx, images, tiles):

    
#     print('Checkpoint keys')
    
#     for k in state.keys():
#         print(k)
        
    images = [_rgb_tensor_normalized(img, get_imagenet_normalization()) for img in images]
    tiles = [_rgb_tensor_normalized(img, get_imagenet_normalization()) for img in tiles]
#     images = torch.stack(images).to('cuda')
#     tiles = torch.stack(tiles).to('cuda')
    
    save_dir = f'./{save_dir_name}/img_{str(idx)}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    os.makedirs(f'{save_dir}/top_attn', exist_ok=True)


#     print("Images shape:", images.shape)
#     print("Tiles shape:", tiles.shape)
#     x = [_rgb_tensor_normalized(img, get_imagenet_normalization()) for img in images]
#     x.extend(_rgb_tensor_normalized(img, get_imagenet_normalization()) for img in tiles)
#     x = torch.stack(x).to('cuda')

    # DINO
    x = []
    x.extend(images)
    x.extend(tiles)
    
    # MedSigLIP
#     x = torch.cat([images, tiles], dim=0)  # [25, 3, 518, 518]


#     for img_idx, img in enumerate(x):
#         img = img.detach().cpu()

#         # If values are NOT in [0,1], normalize them
#         img = img - img.min()
#         img = img / img.max()
#         pil_img = T.ToPILImage()(img)
#         pil_img.save(f"{save_dir}/images/{str(img_idx)}.png")

    instance_type = torch.cat([
        torch.zeros(len(images), dtype=torch.long, device='cuda'),  # 0 for images
        torch.ones(len(tiles), dtype=torch.long, device='cuda')     # 1 for tiles
    ])

    group = torch.zeros(len(x), dtype=torch.long, device='cuda')

    with torch.no_grad():
        embeddings = embedding_model(x)  
        
    x = torch.stack(x).to('cuda')
    ic(x.shape)
    ic(group.shape)
    ic(instance_type.shape)

    ic(embeddings.shape)

    with torch.no_grad():
        attn_map_, scores = head(embeddings, group, instance_type, return_attn_maps=True)
        attn_map_ = attn_map_.squeeze()
        sorted_values, indices = torch.sort(attn_map_)
    print('Len(indices), len(sorted_values): ', len(indices), len(sorted_values))
    print('Scores: ', sorted(scores))

    k = 0
    for idx, val in zip(indices, sorted_values):
        print(f"({idx.item()}, {val.item()})")
        img = tiles[idx].detach().cpu()

        img = img - img.min()
        img = img / img.max()
        pil_img = T.ToPILImage()(img)
        pil_img.save(f"{save_dir}/top_attn/attn_{str(k)}_img_{str(idx.item())}_val_{str(val.item())}.png")
        k += 1


if __name__ == '__main__':
    save_image_list()
#     for idx, (images, tiles, _) in enumerate(MammoDataset(DatasetEnum.VINDR, labels=['BI-RADS 1'], return_mode=ReturnMode.BREAST_TILES_LABEL, convert_to=ConvertTo.RGB_TENSOR_IMGNET_NORM, tile_size=518, final_resize=518)):
    for idx, (images, tiles, _) in enumerate(MammoDataset(DatasetEnum.VINDR, labels=['BI-RADS 5'], split='test', return_mode=ReturnMode.BREAST_TILES_LABEL, convert_to=ConvertTo.PIL, tile_size=518, final_resize=518, read_window=True)):
        print(f'img {idx} num of tiles: {len(tiles)}')
        get_attn_scores(idx, images, tiles)
#         if idx == 20:
#             break