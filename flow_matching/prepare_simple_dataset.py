import pickle
import sys
sys.path.append("../head_training")

train_ds, _, _ = pickle.load(open('/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl', 'rb'))
train_batch = train_ds[0]
x_batch, y, w, group, instance_type = train_batch
print(y.shape, group.shape, instance_type.shape)

# mask = instance_type == 0

# x = x_batch[mask]
# y = y[group[mask]]

# pickle.dump((x, y), open('msl_gl.pkl', 'wb'))
