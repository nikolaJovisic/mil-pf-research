from omegaconf import OmegaConf
from mammo_datasets import *
from embedding_inference_ import EmbeddingInference
from ds_loader import ShardedMammoIterable
# from mammo_iterable import MammoIterable

# def run_raw(split, rank=0): 
#     cfg.run_name = f'{split}-gpu{rank}'

#     ds_params = {
#         "dataset": DatasetEnum.VINDR,
#         "labels": [0, 1],
#         "convert_to": ConvertTo.RGB_TENSOR_IMGNET_NORM,
#         "split": split,
#         "tile_size": 518,
#         "final_resize": 518,
#     }

#     ds_images = MammoDataset(**{**ds_params, "return_mode": ReturnMode.BREAST_LABEL})
#     ds_images_iter = MammoIterable(ds_images, batch_size=1, prefetch=4)
#     #ds_tiles  = MammoDataset(**{**ds_params, "return_mode": ReturnMode.BREAST_TILES_LABEL})

#     embedding_inference = EmbeddingInference(ds_images_iter, None, cfg, device='cuda')
#     embedding_inference.run_images()
#     #embedding_inference.run_tiles()

def run(split, rank, cfg):
    # ds = MammoDataset(
    #     DatasetEnum.VINDR,
    #     return_mode=ReturnMode.BREAST_TILES_LABEL,
    #     labels=[0, 1, 2, 3],
    #     convert_to=ConvertTo.RGB_TENSOR_IMGNET_NORM,
    #     split=split,
    #     tile_size=518,
    #     final_resize=518
    # )

    cfg.run_name = f'{split}-gpu{rank}'

    ds_images = ShardedMammoIterable(split, rank, base_path=cfg.base_path, tiles=False, num_splits=4, prefetch=4096)
    ds_tiles = ShardedMammoIterable(split, rank, base_path=cfg.base_path, tiles=True, num_splits=32)

    embedding_inference = EmbeddingInference(ds_images, ds_tiles, cfg, device='cuda')
    embedding_inference.run_images()
    embedding_inference.run_tiles()

if __name__ == "__main__":
    cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    for split in ['train', 'valid', 'test']:
        for rank in range(6):
            run(split, rank, cfg)
