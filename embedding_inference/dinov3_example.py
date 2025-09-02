import sys
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from mammo_datasets import *

args = {
    'convert_to': ConvertTo.RGB_TENSOR,
    'tile_size': 4096, 
    'final_resize': 1024}

ds = MammoDataset(
    DatasetEnum.EMBED,
    labels=[1, 4, 5, 6],
    split='train',
    **args
)

pretrained_model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name, do_resize=False)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

batch = np.array([ds[0]])
print(batch.shape)

inputs = processor(images=batch, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

from icecream import ic
ic(outputs)
pooled_output = outputs.pooler_output
print(outputs.last_hidden_state.shape)

print("Pooled output shape:", pooled_output.shape)



patch_size = 16
batch_size, _, img_height, img_width = inputs.pixel_values.shape
num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
num_patches_flat = num_patches_height * num_patches_width

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)
print(last_hidden_states.dtype)

cls_token = last_hidden_states[:, 0, :]
register_tokens = last_hidden_states[:, 1:1 + model.config.num_register_tokens, :]
patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
