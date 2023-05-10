import time

from config import create_config_from_yaml
from data_persistence import load_dataset, create_result_dir, store_general_information
from src.utils.plotting import plot_line_plotly
from src.utils.scoring import compute_auc_score
from dean_controller import DeanTsController
import numpy as np

if __name__ == '__main__':
    st = time.time()

    # Load parameter configuration and specify dataset
    dataset_name = 'guten_tag_uts'
    config = create_config_from_yaml()

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
    controller = DeanTsController(config, verbose=True)
    # controller = DeanTsController.load(result_dir + '/model.p')

    # Train ensemble on train data
    controller.train(train_data)

    # Score test data
    y_score = controller.predict(test_data)

    # Store model
    # controller.save(result_dir + '/model.p')

    # Print results
    print('\nModel scores:')
    for y_score_model in controller.ensemble.submodel_scores:
        compute_auc_score(y_score=y_score_model,
                          y_true=y_true,
                          print_result=True)

    print('\nEnsemble score:')
    auc_score_final = compute_auc_score(y_score=y_score,
                                        y_true=y_true,
                                        print_result=True)

    # Plot result interactively
    plot_line_plotly(np.c_[test_data, y_score, controller.ensemble.submodel_scores.T], config, dataset_name)

    print(f'\nTotal runtime: {time.time() - st}')
