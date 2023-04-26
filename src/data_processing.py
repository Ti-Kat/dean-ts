import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(test: np.ndarray, train: np.ndarray) -> (np.ndarray, np.ndarray):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return test, train


def create_lag_feature(arr: np.ndarray, channel: int, lag: int, replace_value=np.nan) -> np.ndarray:
    """ Creates a lag feature by efficiently shifting a numpy array and potentially replacing the pushed out values

    @param arr: The initial array from which to create the lag feature
    @param channel: Channel to use for the lag feature
    @param lag: Shift distance (negative values means shifting "from right to left")
    @param replace_value: Value by which to replace shifted positions
    @return: The lag feature
    """
    e = np.empty((arr.shape[0]))
    if lag >= 0:
        e[:lag] = replace_value
        e[lag:] = arr[:-lag, channel]
    else:
        e[lag:] = replace_value
        e[:lag] = arr[-lag:, channel]
    return e


def add_lag_features(arr: np.ndarray, lag_indices) -> np.ndarray:
    result = arr.copy()
    for lag in lag_indices:
        for feature_index in range(arr.shape[1]):
            lag_feature = create_lag_feature(arr, feature_index, lag)
            result = np.c_[result, lag_feature]
    return result


def split_train_test(arr: np.ndarray, test_ratio: float = 0.2, standardization: bool = True) -> (
        np.ndarray, np.ndarray):
    # Split according to ratio
    split_index = int(arr.shape[0] * (1 - test_ratio))
    train, test = arr[0:split_index], arr[split_index:]

    # Standardize to zero mean and unit variance
    if standardization:
        return standardize(test, train)
    else:
        return train, test


def normalize_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def add_time_colum_pandas(df: pd.DataFrame) -> None:
    df.insert(loc=0, column='time', value=np.arange(len(df.index)))
