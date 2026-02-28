import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

class EvaluationReport:
    FIXED_SENSITIVITIES = (0.9, 0.95, 1.0)
    SENSITIVITY_WEIGHTS = (0.5, 0.75)
    N_SAMPLES = 10
    LINSPACE_STEPS = 100

    def __init__(self, probs, labels, debug=False):
        self.probs = probs.numpy()
        self.targets = labels.numpy().astype(int)

        if debug:
            self.probs = np.concatenate([self.probs, [0, 1]])
            self.targets = np.concatenate([self.targets, [0, 1]])
        
    def _confusion_matrix(self, threshold):
        preds = (self.probs > threshold).astype(int)
        cm = confusion_matrix(self.targets, preds)
        row_sums = cm.sum(axis=1, keepdims=True)
        return cm / row_sums

    def sensitivity(self, threshold):
        cm = self._confusion_matrix(threshold)
        return cm[1, 1]

    def specificity(self, threshold):
        cm = self._confusion_matrix(threshold)
        return cm[0, 0]

    def specificity_at(self, fixed_sensitivity):
        thresholds = np.linspace(1, 0, self.LINSPACE_STEPS)

        for threshold in thresholds:
            if self.sensitivity(threshold) >= fixed_sensitivity:
                return self.specificity(threshold)

        return 0.0
    
    def balanced_accuracy(self, threshold):
        sens = self.sensitivity(threshold)
        spec = self.specificity(threshold)
        return 0.5 * (sens + spec)
    
    def auc(self):
        return roc_auc_score(self.targets, self.probs)

    def optimal_threshold(self, sensitivity_weight):
        thresholds = np.linspace(0, 1, self.LINSPACE_STEPS)
        best_score = -float('inf')
        best_threshold = None

        for threshold in thresholds:
            cm = self._confusion_matrix(threshold)
            sens = self.sensitivity(threshold)
            spec = self.specificity(threshold)
            score = sensitivity_weight * sens + (1 - sensitivity_weight) * spec

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def summary(self):
        summary = {}
        summary['auc'] = self.auc()

        for weight in self.SENSITIVITY_WEIGHTS:
            suffix = f'w{weight}'
            threshold = self.optimal_threshold(weight)
            summary[f'opt_th_{suffix}'] = threshold
            summary[f'spec_{suffix}'] = self.specificity(threshold)
            summary[f'sens_{suffix}'] = self.sensitivity(threshold)
            summary[f'bal_acc_{suffix}'] = self.balanced_accuracy(threshold)

        for sens_level in self.FIXED_SENSITIVITIES:
            summary[f'spec_{int(sens_level * 100)}'] = self.specificity_at(fixed_sensitivity=sens_level)

        return summary

    @staticmethod
    def summary_keys():
        keys = ['auc']

        for weight in EvaluationReport.SENSITIVITY_WEIGHTS:
            suffix = f'w{weight}'
            keys.extend([
                f'opt_th_{suffix}',
                f'spec_{suffix}',
                f'sens_{suffix}',
                f'bal_acc_{suffix}'
            ])

        for sens in EvaluationReport.FIXED_SENSITIVITIES:
            keys.append(f'spec_{int(sens * 100)}')

        return keys



