import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.evaluation_report import EvaluationReport

def evaluate(model, dataset, device='cuda'):
    model.eval()
    all_logits = []
    all_labels = []
    # all_fused = []

    with torch.no_grad():
        for x, y, _, group, instance_type in dataset:
            x, group, instance_type = x.to(device), group.to(device), instance_type.to(device)
            
            # logits, fused = model(x, group, instance_type)
            logits = model(x, group, instance_type)


            all_logits.append(logits.cpu())
            # all_fused.append(fused.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1)
    # fused = torch.cat(all_fused).squeeze(1)
    probs = torch.sigmoid(logits)

    # torch.save({
    #         'pred_logits': logits,
    #         'labels': labels,
    #         'features': fused
    # }, 'train.pth')

    return EvaluationReport(probs, labels)


    