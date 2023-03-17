import numpy as np


def compute_ensemble_score(y_scores):
    return np.mean(y_scores, axis=0)
