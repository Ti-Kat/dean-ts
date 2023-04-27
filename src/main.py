import time

from config import config
from data_persistence import load_dataset, create_result_dir, store_general_information
from src.utils.scoring import compute_auc_score
from dean_controller import DeanTsController

if __name__ == '__main__':
    st = time.time()

    # Load parameter configuration and specify dataset
    dataset_name = 'guten_tag_mts'

    # Load data
    datasets = load_dataset(dataset_name, 'numpy')
    train_data = datasets['train']
    test_data = datasets['test']

    # Define perfect scores
    y_true = test_data[:, -1].astype(int)

    # Persist general information for current run
    result_dir = create_result_dir(dataset_name)
    store_general_information(result_dir, dataset_name, config)

    # Init controller
    controller = DeanTsController(config, train_data, test_data, dataset_name)

    # Train ensemble on train data
    controller.train()

    # Score test data
    y_score = controller.predict()

    # Print results
    print('\nEnsemble score:')
    auc_score_final = compute_auc_score(y_score=y_score,
                                        y_true=y_true,
                                        print_result=True)

    print('\nModel scores:')
    for y_score_model in controller.ensemble.submodel_scores:
        compute_auc_score(y_score=y_score_model,
                          y_true=y_true,
                          print_result=True)

    print(f'\nTotal runtime: {time.time() - st}')
