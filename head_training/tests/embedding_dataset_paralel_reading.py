import sys
from icecream import ic
from itertools import islice
import multiprocessing as mp

sys.path.append('..')
from shim import *

from embedding_inference import EmbeddingsDataset, EmbeddingsDatasetOld
from head_training.utils.collate import collate

# path = '/home/nikola.jovisic.ivi/nj/lustre_mock/vindr-imagenet-test/embeddings.hdf5'
path = '/lustre/nj/dinov3-embeddings/dinov3-s-512-embed/test/embeddings.hdf5'
# old = EmbeddingsDatasetOld({path: {'pos': [4, 5, 6], 'neg': [1]}})
# new = EmbeddingsDataset(path, [4, 5, 6], [1])

def _worker(proc_id, dataset_ctor, ctor_kwargs, samples_per_proc):
    ds = dataset_ctor(**ctor_kwargs)
    seen = 0
    for sample in islice(ds, samples_per_proc):
        seen += 1
    print(f"[proc={proc_id}] seen={seen}", flush=True)

def spawn_parallel_probe(dataset_ctor, ctor_kwargs, procs=2, samples_per_proc=100):
    mp.set_start_method("spawn", force=True)
    ps = []
    for i in range(procs):
        p = mp.Process(target=_worker, args=(i, dataset_ctor, ctor_kwargs, samples_per_proc))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
        
if __name__ == "__main__":
    spawn_parallel_probe(
        EmbeddingsDataset,
        {"h5_path": path, "pos_labels": [4,5,6], "neg_labels": [1]},
        procs=4,
        samples_per_proc=10
    )

