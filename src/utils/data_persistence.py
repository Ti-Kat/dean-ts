import os
import sys

import pandas as pd

# Specify path information
RESULT_BASE_PATH = '../results'  # Relative base path for DEAN-TS results and model information
DATASET_BASE_PATH = '../datasets/'  # Relative base path for datasets

# Maps a dataset by name to its training and test data as well as the periodicity of the time series
# The periodicity should be specified in the configuration accordingly
DATASET_MAPPING = {
    'ecg-combined-diff-1': {
        'train': 'ecg/ecg-combined-diff-1/train_no_anomaly.csv',
        'test': 'ecg/ecg-combined-diff-1/test.csv',
        'periodicity': 100,
    },
    'ecg-diff-count-6': {
        'train': 'ecg/ecg-diff-count-6/train_no_anomaly.csv',
        'test': 'ecg/ecg-diff-count-6/test.csv',
        'periodicity': 20,
    },
    'ecg-diff-count-7': {
        'train': 'ecg/ecg-diff-count-7/train_no_anomaly.csv',
        'test': 'ecg/ecg-diff-count-7/test.csv',
        'periodicity': 20,
    },
    'ecg-diff-count-8': {
        'train': 'ecg/ecg-diff-count-8/train_no_anomaly.csv',
        'test': 'ecg/ecg-diff-count-8/test.csv',
        'periodicity': 20,
    },
    'ecg-diff-count-9': {
        'train': 'ecg/ecg-diff-count-9/train_no_anomaly.csv',
        'test': 'ecg/ecg-diff-count-9/test.csv',
        'periodicity': 20,
    },
    'ecg-length-10': {
        'train': 'ecg/ecg-length-10/train_no_anomaly.csv',
        'test': 'ecg/ecg-length-10/test.csv',
        'periodicity': 20,
    },
    'ecg-length-100': {
        'train': 'ecg/ecg-length-100/train_no_anomaly.csv',
        'test': 'ecg/ecg-length-100/test.csv',
        'periodicity': 20,
    },
    'ecg-noise-10%': {
        'train': 'ecg/ecg-noise-10%/train_no_anomaly.csv',
        'test': 'ecg/ecg-noise-10%/test.csv',
        'periodicity': 100,
    },
    'ecg-trend-sinus': {
        'train': 'ecg/ecg-trend-sinus/train_no_anomaly.csv',
        'test': 'ecg/ecg-trend-sinus/test.csv',
        'periodicity': 20,
    },
    'ecg-type-amplitude': {
        'train': 'ecg/ecg-type-amplitude/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-amplitude/test.csv',
        'periodicity': 15,
    },
    'ecg-type-extremum': {
        'train': 'ecg/ecg-type-extremum/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-extremum/test.csv',
        'periodicity': 15,
    },
    'ecg-type-frequency': {
        'train': 'ecg/ecg-type-frequency/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-frequency/test.csv',
        'periodicity': 15,
    },
    'ecg-type-mean': {
        'train': 'ecg/ecg-type-mean/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-mean/test.csv',
        'periodicity': 15,
    },
    'ecg-type-pattern': {
        'train': 'ecg/ecg-type-pattern/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-pattern/test.csv',
        'periodicity': 15,
    },
    'ecg-type-pattern-shift': {
        'train': 'ecg/ecg-type-pattern-shift/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-pattern-shift/test.csv',
        'periodicity': 15,
    },
    'ecg-type-platform': {
        'train': 'ecg/ecg-type-platform/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-platform/test.csv',
        'periodicity': 15,
    },
    'ecg-type-trend': {
        'train': 'ecg/ecg-type-trend/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-trend/test.csv',
        'periodicity': 15,
    },
    'ecg-type-variance': {
        'train': 'ecg/ecg-type-variance/train_no_anomaly.csv',
        'test': 'ecg/ecg-type-variance/test.csv',
        'periodicity': 15,
    },
}


def load_dataset(dataset_name: str = 'guten_tag_uts', return_format: str = 'numpy'):
    """ Loads the specified dataset and returns it either as numpy array or pandas dataframe

    @param dataset_name: Specifies the dataset
    @param return_format: Either 'pandas' or 'numpy'
    """
    if dataset_name in DATASET_MAPPING:
        mapping = DATASET_MAPPING
    else:
        sys.exit('Unknown dataset')

    data_sets = {}
    for key in ['test', 'train']:
        dataset_path = DATASET_BASE_PATH + mapping[dataset_name][key]

        df = pd.read_csv(
            dataset_path,
        )

        if return_format == 'numpy':
            df = df.to_numpy()

        data_sets[key] = df

    return data_sets


def get_period(dataset_name: str):
    if dataset_name not in DATASET_MAPPING:
        sys.exit('Unknown dataset')
    else:
        return DATASET_MAPPING[dataset_name]['periodicity']


def create_result_dir(dataset_name: str, model_index: int = None):
    if model_index is None:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}'
    else:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}/{model_index}'

    if os.path.isdir(result_dir):
        return result_dir
    os.makedirs(result_dir, exist_ok=False)

    return result_dir
