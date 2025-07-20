from torch.utils.data import IterableDataset

class FlattenGroup(IterableDataset):
    def __init__(self, groupwise_dataset):
        self.dataset = groupwise_dataset

    def __iter__(self):
        for group, label, weight in self.dataset:
            for i in range(group.size(0)):
                yield group[i].unsqueeze(0), label, weight
