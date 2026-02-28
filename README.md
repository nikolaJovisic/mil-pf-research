# mil-pf

**Multiple Instance Learning on Precomputed Features (MIL-PF)** for high-resolution mammography classification.

This repository implements a scalable two-stage pipeline:

1. **Feature precomputation** with frozen foundation encoders  
2. **Lightweight MIL head training** (~40k trainable parameters)

---

## Pipeline

### 1. Preprocess & Save Images
```
ds_prep/save_ds.py
```
Uses  [`mammo-datasets`](https://github.com/nikolaJovisic/mammo-datasets) to preprocess raw 16bit/DICOM images and prepare inputs for encoders.

### 2. Compute Embeddings
```
embeddings_inference/simple_runner.py
```
Encodes global views and tiles with frozen backbones (e.g., DINOv2, MedSigLIP) and saves the outputs.

### 3. Prepare MIL Bags
```
head_training/pickle_datasets.py
```
Builds breast-level `(G_i, T_i, y_i)` embedding batches.

### 4. Train MIL Head
```
head_training/manual_grid_search.py
```
Trains and evaluates, searching through different head configurations.

---

## Key Ideas

- Frozen encoder, train only aggregation head  
- Global + tile embeddings per breast  
- Attention-based aggregation for sparse ROIs  
- Fast iteration in a fixed embedding space  

Designed for weakly labeled, multi-view, high-resolution mammography.
