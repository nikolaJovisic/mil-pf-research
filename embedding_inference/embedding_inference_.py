import os
import uuid
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import h5py
from pathlib import Path
from omegaconf import OmegaConf
#for custom model
#from model.build import build_model
#for dinov2 model:
#from dinov2_wrapper import build_model
#from medsiglip_wrapper import build_model
# from eval_gmic_v2 import build_model
#from bioclip_wrapper import build_model
from mammoclip_wrapper import build_model
# from dinov3_wrapper import build_model
from utils_.serialization import save_embedding_inference
#from batched_dataloader import get_batched_dataloader
import numpy as np
from itertools import islice
import warnings


class EmbeddingInference:
    def __init__(self, ds_images, ds_tiles, cfg=None, device='cuda'):        
        if cfg is None:
            cfg_path = Path(__file__).parent / "config.yaml"
            self.cfg = OmegaConf.load(cfg_path)
        else:
            self.cfg = cfg
        
        self.device = device
        self.model = self._build_model()
        self.ds_images = ds_images
        self.ds_tiles = ds_tiles
        self.output_dir = os.path.join(self.cfg.embeddings_root, self.cfg.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.hdf5_out_path = os.path.join(self.output_dir, 'embeddings.hdf5')
        save_embedding_inference(self, os.path.join(self.output_dir, 'config.yaml'))
 
        
    def run_images(self):
        # dataloader = get_batched_dataloader(
        #     self.ds_images,
        #     batch_size=self.cfg.batch_size,
        #     num_workers=self.cfg.loader_workers,
        #     tiles=False
        # )
        # self._run(dataloader, 'images')
        self._run(self.ds_images, 'images')

    def run_tiles(self):
        # dataloader = get_batched_dataloader(
        #     self.ds_tiles,
        #     batch_size=self.cfg.batch_size,
        #     num_workers=self.cfg.loader_workers,
        #     tiles=True
        # )
        # self._run(dataloader, 'tiles')
        self._run(self.ds_tiles, 'tiles')

    def _run(self, loader, subgroup_name):
        with h5py.File(self.hdf5_out_path, 'a') as h5f:
            with torch.no_grad():
                for batch_images, batch_i, batch_labels in tqdm(loader):
                    #for custom model:
                    #print("$$$$$$$$$$$$$4")
                    #print("Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
                    #print("Reserved:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
                    images = torch.stack(batch_images).to(self.device) 
                    #print("^^^^^^^^^^^^^^^")
                    #print("Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
                    #print("Reserved:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
                    #for dinov2 model:

                    embeddings = self.model(images)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.cpu().numpy()

                    labels_np = np.array(batch_labels)
                    indices = np.array(batch_i)

                    unique_i = np.unique(indices)

                    for ui in unique_i:
                        mask = indices == ui
                        group = h5f.require_group(str(ui))

                        embeddings_to_append = embeddings[mask]
                        label_to_save = labels_np[mask][0]

                        self._append_to_dataset(group, subgroup_name, embeddings_to_append, 'f4')
                        self._save_label_once(group, label_to_save)

    def _append_to_dataset(self, group, name, data, dtype):
        if name in group:
            dset = group[name]
            old_size = dset.shape[0]
            new_size = old_size + data.shape[0]
            dset.resize((new_size,) + dset.shape[1:])
            dset[old_size:new_size] = data
        else:
            group.create_dataset(
                name,
                data=data,
                maxshape=(None,) + data.shape[1:],
                dtype=dtype
            )

    def _save_label_once(self, group, label):
        name = 'label'
        if name not in group:
            group.create_dataset(
                name,
                data=np.array([label]),
                dtype='i8'
            )

    def _build_model(self):
        #for dinov2 model:
        return build_model(self.device) #, self.cfg.save_all)
    
        #for custom model: 
        # model = build_model(self.cfg.model)
        # state_dict = torch.load(self.cfg.model.weights, map_location="cpu")
        # model.load_state_dict(state_dict)
        # model.eval().to(self.device)
        # return model
