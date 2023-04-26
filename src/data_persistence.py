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
    'guten_tag_uts': {
        'test': 'gutenTag/uts/test.csv',
        'train': 'gutenTag/uts/train_no_anomaly.csv',
        'train_anomaly': 'gutenTag/uts/train_anomaly.csv',
    },
    'guten_tag_mts': {
        'test': 'gutenTag/mts/test.csv',
        'train': 'gutenTag/mts/train_no_anomaly.csv',
        'train_anomaly': 'gutenTag/mts/train_anomaly.csv',
    },
}


def load_dataset(dataset_name: str = 'guten_tag_uts', return_format: str = 'pandas'):
    """ Loads the specified dataset and returns it either as numpy array or pandas dataframe

    @param dataset_name: Specifies the dataset, default is the twitter numenta dataset from kaggle
    @param return_format: Either 'pandas' or 'numpy'
    """
    if dataset_name not in DATASET_MAPPING:
        sys.exit('Unknown dataset')

    data_sets = {}
    for key in ['test', 'train', 'train_anomaly']:
        dataset_path = DATASET_BASE_PATH + DATASET_MAPPING[dataset_name][key]

        df = pd.read_csv(
            dataset_path,
            # index_col='timestamp',
            # parse_dates=['timestamp']
        )

        if return_format == 'numpy':
            df = df.to_numpy()

        data_sets[key] = df

    return data_sets


def create_result_dir(dataset_name: str, model_index: int = None):
    if model_index is None:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}'
    else:
        result_dir = f'{RESULT_BASE_PATH}/{dataset_name}/{model_index}'

    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=False)

    return result_dir


def store_as_yaml(result_dir, file_name, content: dict):
    with open(f'{result_dir}/{file_name}', 'w') as yaml_file:
        yaml.dump(content, yaml_file, default_flow_style=False)


def store_as_npz(result_dir, file_name, content: dict):
    np.savez_compressed(f'{result_dir}/{file_name}', **content)


def store_general_information(result_dir, dataset_name, configuration):
    store_as_yaml(
        result_dir,
        'general_infos.yaml',
        {
            'dataset': dataset_name,
            'config': configuration,
        }
    )


def store_results(result_dir, results: dict):
    store_as_npz(result_dir, 'result.npz', results)


def load_results(result_dir):
    results = np.load(f'{result_dir}/result.npz', allow_pickle=True)
    return {key: results[key] for key in results}
