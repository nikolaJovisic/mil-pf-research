import os
import uuid
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import h5py
from pathlib import Path
from omegaconf import OmegaConf
from embedding_inference.model.build import build_model
from embedding_inference.utils.serialization import save_embedding_inference
from embedding_inference.batched_dataloader import get_batched_dataloader, BatchEnum
import numpy as np
from itertools import islice

class EmbeddingInference:
    def __init__(self, dataset, cfg=None, device='cuda'):        
        if cfg is None:
            cfg_path = Path(__file__).parent / "config.yaml"
            self.cfg = OmegaConf.load(cfg_path)
        else:
            self.cfg = cfg
        
        self.device = device
        self.model = self._build_model()
        self.dataset = dataset
        self.output_dir = os.path.join(self.cfg.embeddings_root, self.cfg.run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.hdf5_out_path = os.path.join(self.output_dir, 'embeddings.hdf5')
        save_embedding_inference(self, os.path.join(self.output_dir, 'config.yaml'))
        
    def run_images(self):
        dataloader = get_batched_dataloader(
            self.dataset,
            BatchEnum.IMAGE,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.img_loader_workers
        )
        self._run(dataloader, 'images')

    def run_tiles(self):
        dataloader = get_batched_dataloader(
            self.dataset,
            BatchEnum.TILE,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.tile_loader_workers
        )
        self._run(dataloader, 'tiles')

    def _run(self, loader, subgroup_name):
        with h5py.File(self.hdf5_out_path, 'a') as h5f:
            with torch.no_grad():
                for batch_images, batch_i, batch_labels in loader:
                    images = torch.stack(batch_images).to(self.device)
                    embeddings = self.model(images).cpu().numpy()
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
        model = build_model(self.cfg.model)
        state_dict = torch.load(self.cfg.model.weights, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
    
    def _check_img_size_match():
        model_img_size = self.cfg.model.img_size
        final_resize = self.dataset.format_transform.final_resize
        resize = self.dataset.resize
        tile_size = self.tile_size
        
        if (final_resize not in [None, model_img_size]) or \
           (final_resize is None and \
           (resize != model_img_size or tile_size not in [None, model_img_size])):
            raise ValueError("Image size mismatch between dataset and model.")
