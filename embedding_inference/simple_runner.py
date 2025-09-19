from omegaconf import OmegaConf
from mammo_datasets import *
from embedding_inference_ import EmbeddingInference

def run(split):
    ds = MammoDataset(
        DatasetEnum.EMBED,
        return_mode=ReturnMode.BREAST_TILES_LABEL,
        labels=[1, 4, 5, 6],
        convert_to=ConvertTo.RGB_TENSOR_IMGNET_NORM,
        split=split,
        tile_size=518,
        final_resize=518
    )

    cfg = OmegaConf.load('/home/nikola.jovisic.ivi/nj/mammo_filter/embedding_inference/config.yaml')
    cfg.run_name = split
    inference = EmbeddingInference(ds, cfg) 
    inference.run_images()
    inference.run_tiles()


for split in ['train', 'valid', 'test']:
    run(split)
