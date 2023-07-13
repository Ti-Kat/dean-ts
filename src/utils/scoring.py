from sklearn.metrics import roc_auc_score as auc
import numpy as np


def compute_auc_score(y_score: np.ndarray, y_true: np.ndarray, print_result=False) -> float:
    """Computes the AUC-ROC score"""
    auc_score = auc(y_true, y_score)
    if print_result:
        print(f'Reached AUC score of {auc_score}')
    return auc_score
