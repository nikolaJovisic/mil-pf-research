import threading
import queue
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

class MammoIterable(IterableDataset):
    def __init__(self, dataset, batch_size=1, prefetch=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch = prefetch

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch)

        def producer():
            batch_images, batch_ids, batch_labels = [], [], []
            for i, item in enumerate(self.dataset):
                if isinstance(item, tuple) and len(item) == 3:
                    img, img_id, label = item
                elif isinstance(item, tuple) and len(item) == 2:
                    img, label = item
                    img_id = i
                else:
                    raise ValueError(f"Unexpected item format: {type(item)}")

                # Flatten if the dataset item is already a list of tensors
                if isinstance(img, (list, tuple)) and isinstance(img[0], torch.Tensor):
                    batch_images.extend(img)
                    if isinstance(img_id, (list, tuple)):
                        batch_ids.extend(img_id)
                    else:
                        batch_ids.extend([img_id] * len(img))
                    if isinstance(label, (list, tuple)):
                        batch_labels.extend(label)
                    else:
                        batch_labels.extend([label] * len(img))
                else:
                    batch_images.append(img)
                    batch_ids.append(img_id)
                    batch_labels.append(label)

                if len(batch_images) >= self.batch_size:
                    q.put((batch_images, batch_ids, batch_labels))
                    batch_images, batch_ids, batch_labels = [], [], []

            if batch_images:
                q.put((batch_images, batch_ids, batch_labels))
            q.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

def get_iter_dataloader(dataset):
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
