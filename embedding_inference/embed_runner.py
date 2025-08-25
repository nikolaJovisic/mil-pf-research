import argparse
import sys
import uuid

sys.path.append('..')

from shim import *
from embedding_inference import EmbeddingInference
from omegaconf import OmegaConf


def get_embedding_cfg():
    cfg_path = f'{REPOS_DIR}/mammo_filter/embedding_inference/config.yaml'
    return OmegaConf.load(cfg_path)


def run(split, weights_path, convert_to, gpu_id, run_name_prefix, description):
    args = {'return_mode': ReturnMode.BREAST_TILES_LABEL, 'tile_size': 518, 'final_resize': 4096}
    ds = MammoDataset(
        DatasetEnum.EMBED,
        labels=[1, 4, 5, 6],
        convert_to=convert_to,
        split=split,
        **args
    )
    cfg = get_embedding_cfg()
    if weights_path is not None:
        cfg.model.weights = weights_path
    cfg.run_name = f'{run_name_prefix}-{split}'
    cfg.run_description = description
    inference = EmbeddingInference(ds, cfg, gpu_id)
    inference.run_images()
#     inference.run_tiles()


def run_embedding_pipeline(use_imagenet=False, weights=None, description=None, gpu=None):
    embedding_id = 'imagenet' #str(uuid.uuid4())[:8]
    dataset_id = 'embed'
    run_name_prefix = f'{dataset_id}-{embedding_id}'

    if gpu is None:
        gpu_id = 'cuda'
    else:
        gpu_id = f'cuda:{gpu}'

    if use_imagenet:
        weights_path = None
        convert_to = ConvertTo.RGB_TENSOR_IMGNET_NORM
        run_description = 'EMBED dataset according to csv split, imagenet pretrained model.'
    else:
        if not weights or not description:
            raise ValueError("Both weights and description must be provided when not using ImageNet weights.")
        weights_path = weights
        convert_to = ConvertTo.RGB_TENSOR_NORM
        run_description = description

    for split in ['train', 'valid', 'test']:
        run(split, weights_path, convert_to, gpu_id, run_name_prefix, run_description)

    print(embedding_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet', dest='imagenet', action='store_true', help='Use ImageNet-pretrained DINOv2 weights')
    parser.add_argument('--weights', type=str, help='Path to custom finetuned weights')
    parser.add_argument('--description', type=str, help='Run description for finetuned model')
    parser.add_argument('--gpu', type=int, help='GPU index (e.g., 0 or 1). If not set, defaults to "cuda"')

    args = parser.parse_args()

    run_embedding_pipeline(
        use_imagenet=args.imagenet,
        weights=args.weights,
        description=args.description,
        gpu=args.gpu
    )

if __name__ == '__main__':
    main()
