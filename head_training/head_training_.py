from tqdm import tqdm
import pickle
from icecream import ic
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from omegaconf import OmegaConf
from pathlib import Path
from precollated_dataset import PrecollatedDataset
from utils.flatten_group import FlattenGroup
from utils.evaluate import evaluate
from model.aggregation import Aggregation
from model.model_ import build_model
import os
import copy
from icecream import ic
import math


def merge_embedding_datasets_in_memory(ds1, ds2):
    assert len(ds1) == len(ds2), "Datasets must have the same number of batches."

    merged_batches = []
    for (x1, y1, g1, w1, t1), (x2, y2, g2, w2, t2) in zip(ds1.batches, ds2.batches):
        ic(len(y1), len(y2), x1.shape, x2.shape, g1.shape, g2.shape, w1.shape, w2.shape, t1.shape, t2.shape)
        assert torch.allclose(y1, y2), "Mismatched labels."
        assert torch.all(t1 == t2), "Mismatched instance types."

        # Concatenate embeddings along feature dimension
        merged_x = torch.cat([x1, x2], dim=1)

        merged_batches.append((merged_x, y1, g1, w1, t1))

    merged_dataset = ds1
    merged_dataset.batches = merged_batches
    return merged_dataset

def train_head(run_id, cfg=None, gpu_id=None, just_evaluate=False):
    if cfg is None:
        cfg = load_cfg()
        
    device = 'cuda'
    
    if gpu_id is not None: 
        torch.cuda.set_device(gpu_id)
        device = f'{device}:{gpu_id}'
    
    log_dir = os.path.join(cfg.logs_path, run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{gpu_id}.csv")
    config_file = os.path.join(log_dir, f"{gpu_id}.yaml")
    
    OmegaConf.save(cfg, config_file)

    # datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/rsna-v2.pkl', 'rb'))
    # datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/bsexp/v2-giant-inf.pkl', 'rb'))
    datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl', 'rb'))
    # datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/vindr-msl.pkl', 'rb'))
    train_ds, valid_ds, test_ds = datasets

    # v2_datasets = pickle.load(open('/lustre/nj/cvpr2026/pickles/bsexp/v2-giant-inf.pkl', 'rb'))
    # msl_datasets= pickle.load(open('/lustre/nj/cvpr2026/pickles/bsexp/medsiglip-inf.pkl', 'rb'))

    # v2_train_ds, v2_valid_ds, v2_test_ds = v2_datasets
    # msl_train_ds, msl_valid_ds, msl_test_ds = msl_datasets

    # train_ds = merge_embedding_datasets_in_memory(v2_train_ds, msl_train_ds)
    # valid_ds = merge_embedding_datasets_in_memory(v2_valid_ds, msl_valid_ds)
    # test_ds = merge_embedding_datasets_in_memory(v2_test_ds, msl_test_ds)

    #root = '/lustre/nj/dinov3-embeddings/dinov3-s-512-embed'
    # root = '/home/nikola.jovisic.ivi/nj/mammo_filter'
    # train_ds = PrecollatedDataset(f'{root}/precollated/train', device)
    # valid_ds = PrecollatedDataset(f'{root}/precollated/valid', device)
    # test_ds = PrecollatedDataset(f'{root}/precollated/test', device)

    # if cfg.flatten:
    #     train_ds = FlattenGroup(train_ds)
    #     valid_ds = FlattenGroup(valid_ds)
    
    model = _train(train_ds, valid_ds, cfg, device, log_file, just_evaluate, run_id)
    return evaluate(model, test_ds, device), evaluate(model, valid_ds, device), evaluate(model, train_ds, device)
    
        
def _train(train_dataset, valid_dataset, cfg, device, log_file, just_evaluate, run_id):
    model = build_model(cfg, device)

    if cfg.load_path is not None:
        model.load_state_dict(torch.load(cfg.load_path, map_location=device), strict=False)
        print(f'Model loaded from {cfg.load_path}.')
        
        if cfg.freeze_whole_image_branch:
            model.freeze_whole_image_branch()
            print(f'Whole image branch freezed.')

    if just_evaluate:
        if cfg.load_path is None:
            raise ValueError("Cannot evaluate without load_path!")
        return model

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')


    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,train_auc,val_auc,train_spec@85,val_spec@85\n')

    best_val_auc = float('-inf')
    patience_counter = 0

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        model.train()
        train_loss = 0
        ic('Training:')
        for x, y, w, group, instance_type in tqdm(train_dataset):
            x, y, w, group, instance_type = x.to(device), y.to(device), w.to(device), group.to(device), instance_type.to(device)

            logits = model(x, group, instance_type)
            loss = criterion(logits, y)
            loss = (loss * w).mean()
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            ic('Validating:')
            for x, y, w, group, instance_type in tqdm(valid_dataset):
                x, y, w, group, instance_type = x.to(device), y.to(device), w.to(device), group.to(device), instance_type.to(device)

                logits = model(x, group, instance_type)
                loss = criterion(logits, y)
                loss = (loss * w).mean()
                val_loss += loss.item()

        train_report = evaluate(model, train_dataset, device)
        train_auc = train_report.auc()
        train_spec_at_85 = train_report.specificity_at(0.85)

        val_report = evaluate(model, valid_dataset, device)
        val_auc = val_report.auc()
        val_spec_at_85 = val_report.specificity_at(0.85)

        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{train_auc},{val_auc},{train_spec_at_85},{val_spec_at_85}\n')

        if val_auc - best_val_auc > 0.001:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if cfg.save_path is not None:
        torch.save(best_model_state, os.path.join(cfg.save_path, f'{run_id}.pth'))
        print(f'Model saved to {cfg.save_path}.')

    model.load_state_dict(best_model_state)
    return model


def load_cfg(cfg_path=None):
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    cfg.aggregation = Aggregation(cfg.aggregation)
    
    return cfg
