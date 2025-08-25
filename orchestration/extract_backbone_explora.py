import torch
import sys
from collections import OrderedDict

def extract_backbone(input_path, output_path):
    state_dict = torch.load(input_path, map_location='cpu', weights_only=False)

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue
        if ".lora_A" in k or ".lora_B" in k:
            continue
        if ".q_proj." in k or ".k_proj." in k or ".v_proj." in k:
            continue
        new_state_dict[k[len("backbone."):]] = v

    torch.save(new_state_dict, output_path)
    print(f"Extracted backbone saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_backbone_explora.py <input_pth> <output_pth>")
    else:
        input_pth = sys.argv[1]
        output_pth = sys.argv[2]
        extract_backbone(input_pth, output_pth)
