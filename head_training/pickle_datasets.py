import pickle
from embeddings_dataset import EmbeddingsDataset

train_batch_size =  float('inf') #500000 #float('inf') #200000  #2**18
valid_batch_size = float('inf')
test_batch_size = float('inf')

def get_dataset_cfg():
    embeddings_root = '/lustre/nj/cvpr2026/embeddings' 

    def get_path(split):
        return f'{embeddings_root}/vindr-mammoclip/{split}/embeddings.hdf5'
    
    #embed, vindr
    pos_labels = [4, 5, 6]
    neg_labels = [1]

    #rsna
    # pos_labels = [3]
    # neg_labels = [0, 1]

    #csaw
    # pos_labels = [1]
    # neg_labels = [0]
   

    return {
        'train': ([get_path(f'train-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'valid': ([get_path(f'valid-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels),
        'test': ([get_path(f'test-gpu{gpu}') for gpu in range(6)], pos_labels, neg_labels)
    }

train_ds = EmbeddingsDataset(*get_dataset_cfg()['train'], train_batch_size)
valid_ds = EmbeddingsDataset(*get_dataset_cfg()['valid'], valid_batch_size) 
test_ds = EmbeddingsDataset(*get_dataset_cfg()['test'], test_batch_size)
datasets = (train_ds, valid_ds, test_ds)

pickle.dump(datasets, open('/lustre/nj/cvpr2026/pickles/vindr-mammoclip.pkl', 'wb'))
