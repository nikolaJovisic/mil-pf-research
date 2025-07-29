import sys

sys.path.append('../embedding_inference')
sys.path.append('..')
from shim import *

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from omegaconf import OmegaConf
from pathlib import Path
from embedding_inference import EmbeddingsDataset
from utils.flatten_group import FlattenGroup
from model import Aggregation
from utils.evaluate import evaluate
from utils.collate import collate
from model import Head
import os

def train_head(dataset_cfg, run_id, cfg=None, gpu_id=None, just_evaluate=False):
    if cfg is None:
        cfg = load_cfg()
        
    device = 'cuda'
    
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = f'{device}:{gpu_id}'
    
    log_dir = os.path.join(REPOS_DIR, cfg.logs_path, run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{gpu_id}.csv")
    config_file = os.path.join(log_dir, f"{gpu_id}.yaml")
    
    OmegaConf.save(cfg, config_file)
    
    train_ds = EmbeddingsDataset(dataset_cfg['train'], cfg.use_tiles, cfg.pos_weight)
    valid_ds = EmbeddingsDataset(dataset_cfg['valid'], cfg.use_tiles, cfg.pos_weight)
    test_ds = EmbeddingsDataset(dataset_cfg['test'], cfg.use_tiles, cfg.pos_weight)
    
    if cfg.flatten:
        train_ds = FlattenGroup(train_ds)
        valid_ds = FlattenGroup(valid_ds)
    
    model = _train(train_ds, valid_ds, cfg, device, log_file, just_evaluate)
    return evaluate(model, test_ds, cfg.batch_size, device)
    
        
def _train(train_dataset, valid_dataset, cfg, device, log_file, just_evaluate):
    model = Head(cfg).to(device)

    if cfg.load_path is not None:
        model.load_state_dict(torch.load(cfg.load_path), strict=False)
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
        f.write('epoch,train_loss,val_loss,val_specificity,val_sensitivity\n')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0

        for x, y, w, group, instance_type in collate(train_dataset, cfg.batch_size):
            x, y, w, group, instance_type = x.to(device), y.to(device), w.to(device), group.to(device), instance_type.to(device)

            logits = model(x, group, instance_type)
            loss = criterion(logits, y)
            loss = (loss * w).mean()
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w, group, instance_type in collate(valid_dataset, cfg.batch_size):
                x, y, w, group, instance_type = x.to(device), y.to(device), w.to(device), group.to(device), instance_type.to(device)

                logits = model(x, group, instance_type)
                loss = criterion(logits, y)
                loss = (loss * w).mean()
                val_loss += loss.item()

        if (epoch % cfg.eval_every) == 0:
            report = evaluate(model, valid_dataset, cfg.batch_size, device)
            specificity, sensitivity = report.specificity(0.5), report.sensitivity(0.5)
        else:
            specificity, sensitivity = '', ''

        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{specificity},{sensitivity}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if cfg.save_path is not None:
        torch.save(best_model_state, cfg.save_path)
        print(f'Model saved to {cfg.save_path}.')

    model.load_state_dict(best_model_state)
    return model


def load_cfg(cfg_path=None):
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    cfg.aggregation = Aggregation(cfg.aggregation)
    
    return cfg
