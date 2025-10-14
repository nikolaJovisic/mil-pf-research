import pickle
from embeddings_dataset import EmbeddingsDataset

batch_size = 2**18
pos_weight = 1.0

def get_dataset_cfg():
    embeddings_root = '/lustre/nj/cvpr2026' 

    def get_path(split):
        return f'{embeddings_root}/vitb-explora-baseline/{split}/embeddings.hdf5'

    pos_labels = [4, 5, 6]
    neg_labels = [1]

    return {
        'train': ([get_path(f'train-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'valid': ([get_path(f'valid-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'test': ([get_path(f'test-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels)
    }

train_ds = EmbeddingsDataset(*get_dataset_cfg()['train'], batch_size, pos_weight)
valid_ds = EmbeddingsDataset(*get_dataset_cfg()['valid'], batch_size, pos_weight)
test_ds = EmbeddingsDataset(*get_dataset_cfg()['test'], batch_size, pos_weight)
datasets = (train_ds, valid_ds, test_ds)

pickle.dump(datasets, open('vitb-explora-baseline.pkl', 'wb'))
