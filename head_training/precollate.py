import torch
import uuid
from torch.nn.functional import normalize 
import h5py
from einops import rearrange    
from icecream import ic
from collections import Counter
from utils.collate import collate
import os

def precollate(h5_path, save_dir, pos_labels, neg_labels, pos_weight=1.0, max_gb=10):
    pos_labels = set(pos_labels)
    neg_labels = set(neg_labels)
    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as file:
        # COMMENTED OUT FOR DISTRIBUTED PRECOLLATION
        # labels = [int(group['label'][()]) for group in file.values()]
        # ic(file['0']['embeddings'][()].shape)

        # counter = Counter(labels)
        # ic(counter)

        # unexpected = set(counter) - (pos_labels | neg_labels)
        # if unexpected:
        #     ic(f"Warning: unexpected labels {unexpected}")

        # def compute_weights(counter, pos_weight):
        #     pos_count = sum(counter[l] for l in pos_labels)
        #     neg_count = sum(counter[l] for l in neg_labels)
        #     ic(pos_count, neg_count)
        #     total = pos_count + neg_count
        #     weights = {0: (total / (2 * neg_count)),
        #                1: pos_weight * (total / (2 * pos_count))}
        #     return weights

        weights = {0: 0.528, 1: 9.448} #compute_weights(counter, pos_weight)
        ic(weights)

        items = []
        total_bytes = 0
        batches = []

        for group in file.values():
            label = int(group['label'][()])
            if label not in pos_labels | neg_labels:
                continue
            label = int(label in pos_labels)
            embeddings = torch.from_numpy(group['embeddings'][()])
            emb_bytes = embeddings.numel() * embeddings.element_size()
            weight = torch.tensor([weights[label]], dtype=torch.float32)
            label_tensor = torch.tensor([label], dtype=torch.float32)

            items.append((embeddings, label_tensor, weight))
            total_bytes += emb_bytes

            if total_bytes >= max_gb * (1024**3):  
                file_index = str(uuid.uuid4())[:8]
                big_batch = [batch for batch in collate(items, float('inf'))][0]
                torch.save(big_batch, os.path.join(save_dir, f"batch_{file_index}.pt"))
                print(f"Saved batch_{file_index}.pt with {len(items)} breasts.")
                items.clear()
                total_bytes = 0

        if items:
            file_index = str(uuid.uuid4())[:8]
            big_batch = [batch for batch in collate(items, float('inf'))][0]
            torch.save(big_batch, os.path.join(save_dir, f"batch_{file_index}.pt"))
            print(f"Saved final batch_{file_index}.pt with {len(items)} breasts.")
            batches.extend(big_batch)

    return batches

if __name__ == "__main__":
    root = '/lustre/nj/dinov3-embeddings/dinov3-s-512-embed'
    for split in ['train', 'valid', 'test']:
        path = f'{root}/{split}/embeddings.hdf5'
        precollate(path, f'{root}/precollated/{split}', pos_labels=[4,5,6], neg_labels=[1], pos_weight=1.0, max_gb=10)

