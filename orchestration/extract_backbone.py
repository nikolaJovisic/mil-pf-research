import torch
import sys
from collections import OrderedDict

def extract_backbone(input_path, output_path, use_teacher=False):
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    model_dict = ckpt["model"]

    prefix = "teacher.backbone." if use_teacher else "student.backbone." #for ddp use student.backbone.module.
    new_state_dict = OrderedDict()

    for k, v in model_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v

    torch.save(new_state_dict, output_path)
    print(f"Extracted {'teacher' if use_teacher else 'student'} backbone saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_backbone.py <input_pth> <output_pth> [--teacher]")
    else:
        use_teacher = "--teacher" in sys.argv
        input_pth = sys.argv[1]
        output_pth = sys.argv[2]
        extract_backbone(input_pth, output_pth, use_teacher)

