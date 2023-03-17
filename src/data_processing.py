import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def shift(arr: np.array, n: int, replace_value=np.nan):
    """ Efficiently shifts a numpy array and replaces the pushed out values

    @param arr: The array that is to be shifted
    @param n: Shift distance (negative values means shifting "from right to left")
    @param replace_value: Value by which to replace shifted positions
    @return: The shifted array
    """
    e = np.empty((arr.shape[0]))
    if n >= 0:
        e[:n] = replace_value
        e[n:] = arr[:-n, 0]
    else:
        e[n:] = replace_value
        e[:n] = arr[-n:, 0]
    return e


def split_train_test(arr: np.array, test_ratio: float = 0.2, standardize: bool = True) -> (np.array, np.array):
    # Split according to ratio
    split_index = int(arr.shape[0] * (1-test_ratio))
    train, test = arr[0:split_index], arr[split_index:]

    # Standardize to zero mean and unit variance
    if standardize:
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

    return train, test


def add_lag_features(arr: np.array, lag_indices):
    result = arr.copy()
    for lag in lag_indices:
        feature = shift(arr, lag)
        result = np.c_[result, feature]
    return result


def normalize_pandas(df: pd.DataFrame):
    return (df - df.mean()) / df.std()


def add_time_colum_pandas(df: pd.DataFrame):
    df.insert(loc=0, column='time', value=np.arange(len(df.index)))
