import numpy as np


def statinf(q):
    return {"shape": q.shape, "mean": np.mean(q), "std": np.std(q), "min": np.min(q), "max": np.max(q)}
