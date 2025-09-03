import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.evaluation_report import EvaluationReport

def evaluate(model, dataset, batch_size, device='cuda'):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y, _, group in dataset:
            #x, group = x.to(device), group.to(device)
            
            logits = model(x, group)

            all_logits.append(logits.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)
    probs = torch.sigmoid(logits)

    return EvaluationReport(probs, labels)


    