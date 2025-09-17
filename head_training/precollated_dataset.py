import os
import torch
from torch.utils.data import Dataset
from threading import Thread
from queue import Queue
import time
from icecream import ic 


class PrecollatedDataset(Dataset):
    def __init__(self, pt_dir, device="cuda", buffer_size=4):
        self.pt_dir = pt_dir
        self.device = device
        self.batch_files = sorted(
            [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
        )
        if not self.batch_files:
            raise ValueError(f"No .pt batches found in {pt_dir}")

        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.stop_signal = False

        self.prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()

        while self.queue.qsize() < min(self.buffer_size, len(self.batch_files)):
            ic(self.queue.qsize())
            time.sleep(1)
            pass

        print(f"Initialized PrecollatedDataset with {len(self.batch_files)} batches.")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        ic(self.queue.qsize())
        item_idx, batch = self.queue.get()
        if item_idx != idx:
            raise RuntimeError(
                f"Out-of-order access: requested {idx}, got {item_idx}"
           )
        return tuple(x.to(self.device) for x in batch)

    def _prefetch_loop(self):
        while not self.stop_signal:
            for j, path in enumerate(self.batch_files):
                if self.stop_signal:
                    return
                batch = torch.load(path, map_location='cpu')
                self.queue.put((j, batch))

    def shutdown(self):
        self.stop_signal = True
        self.prefetch_thread.join()
