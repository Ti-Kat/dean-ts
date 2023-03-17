import pandas as pd
import numpy as np
import os
import shutil
import sys
import yaml

# Load parameter configuration and specify path information
config = yaml.safe_load(open('../config/configuration.yaml'))
RESULT_BASE_PATH = config['result_path']
DATASET_BASE_PATH = config['dataset_path']

DATASET_MAPPING = {
    'gecco': 'gecco_iot/gecco_iot.csv',

    'twitter': 'nab/twitter_numenta_kaggle.csv',
    'art_daily': 'nab/art_daily_flatmiddle.csv',

    'point_global': 'tods/uts/point_global_0.05.csv',
    'point_contextual': 'tods/uts/point_contextual_0.05.csv',

    'collective_global': 'tods/uts/collective_global_0.05.csv',
    'collective_trend': 'tods/uts/collective_trend_0.05.csv',
    'collective_seasonal': 'tods/uts/collective_seasonal_0.05.csv',
}


def load_dataset(dataset_name: str = 'twitter', return_format: str = 'pandas'):
    """ Loads the specified dataset and returns it either as numpy array or pandas dataframe

    @param dataset_name: Specifies the dataset, default is the twitter numenta dataset from kaggle
    @param return_format: Either 'pandas' or 'numpy'
    """
    if dataset_name in DATASET_MAPPING:
        file_path = DATASET_BASE_PATH + DATASET_MAPPING[dataset_name]
    else:
        sys.exit('Unknown dataset')

    # TODO: Unify dataset composition (time, values, anomaly)
    df = pd.read_csv(
        file_path,
        # index_col='time',
    )

    # df = pd.read_csv(
    #     file_path,
    #     index_col='timestamp',
    #     parse_dates=['timestamp'])

    if return_format == 'numpy':
        df = df.to_numpy()

    return df


def create_result_dir(dataset_name: str, model_index: int = None):
    if model_index is None:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}'
    else:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}/{model_index}'

    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=False)

    return result_dir


# TODO: Implement this
def store_general_information(result_dir, dataset_name, configuration):
    return


# TODO: Use *args instead
def store_results(result_dir, y_score, auc_score, lag_indices):
    np.savez_compressed(f'{result_dir}/result.npz',
                        y_score=y_score,
                        auc_score=auc_score,
                        lag_indices=lag_indices)
