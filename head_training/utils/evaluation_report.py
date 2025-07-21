import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

class EvaluationReport:
    FIXED_SENSITIVITIES = (0.9, 0.95, 1.0)
    SENSITIVITY_WEIGHTS = (1.0, 1.5)
    N_SAMPLES = 10
    LINSPACE_STEPS = 100

    def __init__(self, probs, labels):
        self.probs = probs.numpy()
        self.targets = labels.numpy().astype(int)

    def _confusion_matrix(self, threshold):
        preds = (self.probs > threshold).astype(int)
        targets = np.concatenate([self.targets, [0, 1]])
        preds = np.concatenate([preds, [0, 1]])
        cm = confusion_matrix(targets, preds)
        row_sums = cm.sum(axis=1, keepdims=True)
        return cm / row_sums

    def sensitivity(self, threshold):
        cm = self._confusion_matrix(threshold)
        return cm[1, 1]

    def specificity(self, threshold):
        cm = self._confusion_matrix(threshold)
        return cm[0, 0]

    def specificity_at(self, fixed_sensitivity, tol=1e-3):
        thresholds = np.linspace(0, 1, self.LINSPACE_STEPS)
        best_spec = 0.0

        for threshold in thresholds:
            cm = self._confusion_matrix(threshold)
            sens = cm[1, 1]
            spec = cm[0, 0]

            if abs(sens - fixed_sensitivity) <= tol and spec > best_spec:
                best_spec = spec

        return best_spec

    def auc_resampled(self):
        pos_mask = self.targets == 1
        neg_mask = self.targets == 0

        pos_probs = self.probs[pos_mask]
        pos_labels = self.targets[pos_mask]

        neg_probs = self.probs[neg_mask]
        neg_labels = self.targets[neg_mask]

        aucs = []
        rng = np.random.default_rng()

        for _ in range(self.N_SAMPLES):
            idx = rng.choice(len(neg_probs), size=len(pos_probs), replace=False)
            sample_probs = np.concatenate([pos_probs, neg_probs[idx]])
            sample_labels = np.concatenate([pos_labels, neg_labels[idx]])

            try:
                auc = roc_auc_score(sample_labels, sample_probs)
                aucs.append(auc)
            except ValueError:
                continue

        return np.mean(aucs), np.var(aucs)

    def optimal_threshold(self, sensitivity_weight):
        thresholds = np.linspace(0, 1, self.LINSPACE_STEPS)
        best_score = -float('inf')
        best_threshold = None

        for threshold in thresholds:
            cm = self._confusion_matrix(threshold)
            sens = cm[1, 1]
            spec = cm[0, 0]
            score = sensitivity_weight * sens + (1 - sensitivity_weight) * spec

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def sensitivity_specificity_at_threshold(self, threshold):
        cm = self._confusion_matrix(threshold)
        return cm[1, 1], cm[0, 0]

    def optimal_threshold_resampled(self, sensitivity_weight, steps=1000):
        pos_mask = self.targets == 1
        neg_mask = self.targets == 0

        pos_probs = self.probs[pos_mask]
        pos_labels = self.targets[pos_mask]

        neg_probs = self.probs[neg_mask]
        neg_labels = self.targets[neg_mask]

        thresholds = []
        rng = np.random.default_rng()

        for _ in range(self.N_SAMPLES):
            idx = rng.choice(len(neg_probs), size=len(pos_probs), replace=False)
            sample_probs = np.concatenate([pos_probs, neg_probs[idx]])
            sample_labels = np.concatenate([pos_labels, neg_labels[idx]])

            try:
                fpr, tpr, ths = roc_curve(sample_labels, sample_probs)
                score = sensitivity_weight * tpr + (1 - sensitivity_weight) * (1 - fpr)
                best_idx = np.argmax(score)
                thresholds.append(ths[best_idx])
            except ValueError:
                continue

        thresholds = np.array(thresholds)
        mean_thresh = thresholds.mean()
        std_thresh = thresholds.std()

        sens, spec = self.sensitivity_specificity_at_threshold(mean_thresh)

        return {
            'mean_threshold': mean_thresh,
            'std_threshold': std_thresh,
            'sensitivity_at_mean_threshold': sens,
            'specificity_at_mean_threshold': spec
        }

    def summary(self):
        summary = {}

        for weight in self.SENSITIVITY_WEIGHTS:
            opt = self.optimal_threshold_resampled(sensitivity_weight=weight)
            prefix = f'weight_{weight}'

            summary[f'{prefix}_mean_threshold'] = opt['mean_threshold']
            summary[f'{prefix}_std_threshold'] = opt['std_threshold']
            summary[f'{prefix}_sensitivity_at_mean_threshold'] = opt['sensitivity_at_mean_threshold']
            summary[f'{prefix}_specificity_at_mean_threshold'] = opt['specificity_at_mean_threshold']

        for sens_level in self.FIXED_SENSITIVITIES:
            spec = self.specificity_at(fixed_sensitivity=sens_level)
            summary[f'specificity_at_sens_{int(sens_level * 100)}'] = spec

        auc_mean, auc_std = self.auc_resampled()
        summary['auc_mean'] = auc_mean
        summary['auc_std'] = auc_std

        return summary

    @staticmethod
    def summary_keys():
        keys = []

        for weight in EvaluationReport.SENSITIVITY_WEIGHTS:
            prefix = f'weight_{weight}'
            keys.extend([
                f'{prefix}_mean_threshold',
                f'{prefix}_std_threshold',
                f'{prefix}_sensitivity_at_mean_threshold',
                f'{prefix}_specificity_at_mean_threshold'
            ])

        for sens in EvaluationReport.FIXED_SENSITIVITIES:
            keys.append(f'specificity_at_sens_{int(sens * 100)}')

        keys.extend(['auc_mean', 'auc_std'])
        return keys

