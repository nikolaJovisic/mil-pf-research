import os
import multiprocessing as mp
from precollate import precollate

def worker(args):
    h5_path, save_dir, pos_labels, neg_labels, pos_weight, max_gb = args
    precollate(
        h5_path,
        save_dir,
        pos_labels=pos_labels,
        neg_labels=neg_labels,
        pos_weight=pos_weight,
        max_gb=max_gb,
    )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    root = '/lustre/nj/dinov3-embeddings'


    jobs = [
        (f"{root}/b-2048/train-gpu0/embeddings.hdf5", f"{root}/b-2048/precollated/train"),
        (f"{root}/b-2048/train-gpu1/embeddings.hdf5", f"{root}/b-2048/precollated/train"),
        (f"{root}/b-2048/train-gpu2/embeddings.hdf5", f"{root}/b-2048/precollated/train"),
        (f"{root}/b-2048/train-gpu3/embeddings.hdf5", f"{root}/b-2048/precollated/train"),
        (f"{root}/b-2048/train-gpu4/embeddings.hdf5", f"{root}/b-2048/precollated/train"),
        (f"{root}/b-2048/train-gpu5/embeddings.hdf5", f"{root}/b-2048/precollated/train"),

        (f"{root}/b-2048/valid-gpu0/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),
        (f"{root}/b-2048/valid-gpu1/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),
        (f"{root}/b-2048/valid-gpu2/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),
        (f"{root}/b-2048/valid-gpu3/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),
        (f"{root}/b-2048/valid-gpu4/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),
        (f"{root}/b-2048/valid-gpu5/embeddings.hdf5", f"{root}/b-2048/precollated/valid"),

        (f"{root}/b-2048/test-gpu0/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
        (f"{root}/b-2048/test-gpu1/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
        (f"{root}/b-2048/test-gpu2/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
        (f"{root}/b-2048/test-gpu3/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
        (f"{root}/b-2048/test-gpu4/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
        (f"{root}/b-2048/test-gpu5/embeddings.hdf5", f"{root}/b-2048/precollated/test"),
    ]

    pos_labels = [4, 5, 6]
    neg_labels = [1]
    pos_weight = 1.0
    max_gb = 8 

    args_list = [(h5, out, pos_labels, neg_labels, pos_weight, max_gb) for h5, out in jobs]

    with mp.Pool(processes=4) as pool:
        #doesn't really work, processes die out, better to run sequentially for now
        pool.map(worker, args_list)
