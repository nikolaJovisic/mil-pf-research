import sys

sys.path.append('../embedding_inference')

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
from utils.split import split
from model import Aggregation
from utils.evaluate import evaluate
from utils.collate import collate
from model import Head

def train_head(dataset_config, cfg=None, gpu_id=None, just_evaluate=False):
    if cfg is None:
        cfg = load_cfg()
        
    device = 'cuda'
    
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        device = f'{device}:{gpu_id}'
        
    dataset = EmbeddingsDataset(dataset_config, cfg.pos_weight)
    
    train_ds, valid_ds, test_ds = split(dataset, cfg.valid_size, cfg.test_size, cfg.flatten)
    
    model = _train(train_ds, valid_ds, cfg, device, just_evaluate)
    return evaluate(model, test_ds, cfg.batch_size, device)
    
        
def _train(train_dataset, valid_dataset, cfg, device, just_evaluate):
    model = Head(cfg).to(device)
    
    if cfg.load_path is not None:
        model.load_state_dict(torch.load(cfg.load_path))
        print(f'Model loaded from {cfg.load_path}.')
    
    if just_evaluate:
        if cfg.load_path is None:
            raise ValueError("Cannot evaluate without load_path!")
        return model
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.epochs):
        print('Epoch:', epoch)
        
        model.train()
        train_loss = 0

        for x, y, w, group in collate(train_dataset, cfg.batch_size):
            x, y, w, group = x.to(device), y.to(device), w.to(device), group.to(device)

            logits = model(x, group)
            loss = criterion(logits, y)
            loss = (loss * w).mean()
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        print('Train loss:', train_loss)
        

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w, group in collate(valid_dataset, cfg.batch_size):
                x, y, w, group = x.to(device), y.to(device), w.to(device), group.to(device)
                
                logits = model(x, group)
                loss = criterion(logits, y)
                loss = (loss * w).mean()
                val_loss += loss.item()

        print('Val loss:', val_loss)
        
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
