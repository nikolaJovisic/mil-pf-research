import pickle
import torch

from collections import defaultdict

def sample_batch(x, y, batch_size, device=None):
    classes = torch.unique(y)
    n_classes = len(classes)
    per_class = batch_size // n_classes

    indices_by_class = defaultdict(list)
    for idx, label in enumerate(y):
        indices_by_class[int(label)].append(idx)

    while True:
        idx1 = []
        idx2 = []
        by = []

        for c in classes:
            idxs = indices_by_class[int(c)]
            idxs = torch.tensor(idxs)

            perm1 = torch.randperm(len(idxs))
            perm2 = torch.randperm(len(idxs))

            if len(idxs) == 1:
                raise ValueError("Need at least 2 samples per class")

            perm2[perm2 == perm1] = perm2.roll(1)[perm2 == perm1]

            sel1 = idxs[perm1[:per_class]]
            sel2 = idxs[perm2[:per_class]]

            idx1.append(sel1)
            idx2.append(sel2)
            by.append(torch.full((per_class,), c))

        idx1 = torch.cat(idx1)
        idx2 = torch.cat(idx2)
        by = torch.cat(by)

        perm = torch.randperm(len(by))
        idx1 = idx1[perm]
        idx2 = idx2[perm]
        by = by[perm]

        bx1 = x[idx1]
        bx2 = x[idx2]

        if device is not None:
            bx1 = bx1.to(device)
            bx2 = bx2.to(device)
            by = by.to(device)

        yield bx1, bx2, by


if __name__ == "__main__":
    with open('msl_gl.pkl', 'rb') as f:
        x, y = pickle.load(f)

    for x_a, x_b, c in sample_batch(x, y, batch_size=4):
        print(x_a.shape, x_b.shape, c)