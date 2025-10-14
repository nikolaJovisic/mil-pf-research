from omegaconf import OmegaConf
from mammo_datasets import *
from embedding_inference_ import EmbeddingInference
from ds_loader import ShardedMammoIterable

def run(split, rank):
    # ds = MammoDataset(
    #     DatasetEnum.EMBED,
    #     return_mode=ReturnMode.BREAST_TILES_LABEL,
    #     labels=[1, 4, 5, 6],
    #     convert_to=ConvertTo.RGB_TENSOR_IMGNET_NORM,
    #     split=split,
    #     tile_size=518,
    #     final_resize=518
    # )

    cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    cfg.run_name = f'{split}-gpu{rank}'

    ds_images = ShardedMammoIterable(split, rank, tiles=False, num_splits=4, prefetch=4096)
    ds_tiles = ShardedMammoIterable(split, rank, tiles=True, num_splits=32)

    embedding_inference = EmbeddingInference(ds_images, ds_tiles, cfg, device='cuda')
    embedding_inference.run_images()
    embedding_inference.run_tiles()


for split in ['train', 'valid', 'test']:
    for rank in range(6):
        run(split, rank)
