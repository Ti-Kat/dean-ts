import numpy as np
import time
from multiprocessing import Process, Manager

from config import config
from data_persistence import load_dataset, create_result_dir, store_general_information
from model_combination import compute_ensemble_score
from model_evaluation import compute_auc_score
from submodel_dean_controller import DeanTsController

if __name__ == '__main__':
    st = time.time()

    # Load parameter configuration and specify dataset
    dataset_name = 'guten_tag_uts_1'

    # Load data
    complete_dataset = load_dataset(dataset_name, 'numpy')

    # Specify which indices of the test split correspond to actual anomalies
    split_index = int(complete_dataset.shape[0] * (1 - config['test_ratio']))
    test_indices_normal = np.where(complete_dataset[split_index:, -1] == 0)[0]
    test_indices_anomalous = np.where(complete_dataset[split_index:, -1] == 1)[0]

    # Drop time and anomaly columns
    dataset = np.delete(complete_dataset, obj=[0, -1], axis=1)
    channel_count = dataset.shape[1]

    # Define perfect scores
    y_true = complete_dataset[split_index:, -1].astype(int)

    # Persist general information for current run
    result_dir = create_result_dir(dataset_name)
    store_general_information(result_dir, dataset_name, config)

    # Init list to gather submodel scores
    y_scores_list = Manager().list()

    # Train submodels
    processes = [Process(target=DeanTsController.run, args=(dataset, y_true, dataset_name, i + config['ensemble_size'], y_scores_list))
                 for i in range(0, config['ensemble_size'])]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Ensemble score
    print('Ensemble score:')
    y_score_final = compute_ensemble_score(np.array(y_scores_list))
    auc_score_final = compute_auc_score(y_score_final, y_true, print_result=True)

    print(f'Total runtime: {time.time() - st}')
