import torch 

class FlattenGroup(torch.utils.data.IterableDataset):
    def __init__(self, groupwise_dataset):
        self.dataset = groupwise_dataset

    def __iter__(self):
        for x, y, w, group, instance_type in self.dataset:
            # Select only whole instances
            mask = instance_type == 0
            if not mask.any():
                continue  # skip group if none are whole
            x, group, instance_type = x[mask], group[mask], instance_type[mask]

            for i in range(x.size(0)):
                yield (
                    x[i].unsqueeze(0),
                    y if y.ndim == 0 else y[i].unsqueeze(0),
                    w if w.ndim == 0 else w[i].unsqueeze(0),
                    group[i].unsqueeze(0),
                    instance_type[i].unsqueeze(0),
                )
