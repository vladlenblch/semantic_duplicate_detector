import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score


class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def roc_auc(y_true, y_proba):
        return roc_auc_score(y_true, y_proba)

    @staticmethod
    def precision_at_k(y_true, y_score, k):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length")

        if len(y_score) == 0:
            raise ValueError("y_score must not be empty")

        if 0 < k < 1:
            k = max(1, int(len(y_score) * k))

        k = int(k)

        if k <= 0:
            raise ValueError("k must be positive")

        k = min(k, len(y_score))
        top_k_indexes = np.argsort(y_score)[::-1][:k]
        top_k_true = y_true[top_k_indexes]

        return float(np.mean(top_k_true == 1))

    @staticmethod
    def report(y_true, y_pred, y_score, k):
        return {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "roc_auc": Metrics.roc_auc(y_true, y_score),
            "precision_at_k": Metrics.precision_at_k(y_true, y_score, k),
        }
