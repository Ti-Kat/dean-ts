import time

from config import create_config_from_yaml
from src.dean_controller import DeanTsController
from src.dean_interpreter import DeanTsInterpreter
from src.utils.data_exploration import plot_result_interactive, plot_roc_curve, plot_pr_curve
from src.utils.data_persistence import load_dataset, create_result_dir, get_period
from src.utils.scoring import compute_auc_score

# Select dataset (see src.utils.data_persistence for more options)
dataset_name = 'ecg-noise-10%'

if __name__ == '__main__':
    # Load parameter configuration
    config = create_config_from_yaml()

    # Persist general information for current run
    result_dir = create_result_dir(dataset_name)

    # Load data
    datasets = load_dataset(dataset_name)
    train_data = datasets['train']
    test_data = datasets['test']

    # Update periodicity for potential tsd usage
    config['period'] = get_period(dataset_name)

    # Define ground truth as perfect scores
    y_true = test_data[:, -1].astype(int)

    # Save starting time
    st = time.time()

    # Init or load controller
    controller = DeanTsController(config, verbose=True)
    # controller = DeanTsController.load(result_dir + '/model.pkl')

    # Train ensemble on train data
    controller.train(train_data)

    # Score test data
    y_score = controller.predict(test_data)

    # Store model
    controller.save(result_dir + '/model.pkl')

    # Print performance results and runtime
    print('\nModel scores:')
    for y_score_model in controller.ensemble.submodel_scores:
        compute_auc_score(y_score=y_score_model,
                          y_true=y_true,
                          print_result=True)

    print('\nEnsemble score:')
    auc_score_ensemble = compute_auc_score(y_score=y_score,
                                           y_true=y_true,
                                           print_result=True)

    runtime = time.time() - st
    print(f'\nTotal runtime: {runtime}')

    # Plot result
    plot_result_interactive(test_data=datasets['test'],
                            controller=controller,
                            dataset_name=dataset_name,
                            score=auc_score_ensemble,
                            runtime=runtime)

    plot_roc_curve(y_true, y_score)
    plot_pr_curve(y_true, y_score)

    # Plot TSD feature combination importance (depends on a suitably large ensemble size for robust results)
    interpreter = DeanTsInterpreter(controller)
    interpreter.compute_importance()
    interpreter.plot_importance_scores(3800, 3899)
