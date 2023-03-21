import numpy as np
import time
from pprint import pprint

from config import config
from data_persistence import load_dataset, create_result_dir, store_results, store_general_information, load_results
from data_processing import split_train_test, add_lag_features
from model_combination import compute_ensemble_score
from model_evaluation import compute_auc_score
from submodel_dean_basic import build_submodel, train_submodel, score

st = time.process_time()

# Load parameter configuration and specify dataset
dataset_name = 'point_global'

# Load data
complete_dataset = load_dataset(dataset_name, 'numpy')

# Specify which indices of the test split correspond to actual anomalies
split_index = int(complete_dataset.shape[0] * (1 - config['test_ratio']))
test_indices_normal = np.where(complete_dataset[split_index:, 2] == 0)[0]
test_indices_anomalous = np.where(complete_dataset[split_index:, 2] == 1)[0]

# Drop time and anomaly columns
dataset = np.delete(complete_dataset, obj=[0, -1], axis=1)
channel_count = dataset.shape[1]

# Define perfect scores
y_true = np.concatenate((
    np.zeros_like(test_indices_normal),
    np.ones_like(test_indices_anomalous),
))

# Persist general information for current run
result_dir = create_result_dir(dataset_name)
store_general_information(result_dir, dataset_name, config)

# Init list to gather submodel scores
y_scores = []

# Train submodels
for i in range(0, config['ensemble_size']):
    # Create result sub directory
    result_sub_dir = create_result_dir(dataset_name, model_index=i)

    # Randomly select previous time steps for each channel
    lag_indices_per_channel = {
        channel: np.random.choice(range(1, config['look_back']),
                                  size=config['bag'] - 1,
                                  replace=False)
        for channel in range(dataset.shape[1])
    }

    # Add values from chosen time steps as features
    transformed_set = add_lag_features(dataset, lag_indices_per_channel)

    # Split into train and test
    train, test = split_train_test(transformed_set, test_ratio=config['test_ratio'])

    # Truncate up until look_back s.t. features are always defined
    train = train[config['look_back']:]

    # Build base detector model
    submodel = build_submodel([config['bag'] * channel_count] * config['depth'],
                              reg=None,
                              act='relu',
                              mean=1.0)
    train_submodel(submodel, train, config, result_sub_dir)

    # Score the model
    y_score = score(submodel,
                    train,
                    test[test_indices_normal],
                    test[test_indices_anomalous])
    y_scores.append(y_score)

    auc_score = compute_auc_score(y_score, y_true, print_result=True)

    # Save results
    store_results(result_sub_dir,
                  {
                      'y_score': y_score,
                      'auc_score': auc_score,
                      'lag_indices_per_channel': lag_indices_per_channel
                  })

    pprint(f'Time for run {i}: {time.process_time() - st}')

# Ensemble score
print('Ensemble score:')
y_score_final = compute_ensemble_score(np.array(y_scores))
auc_score_final = compute_auc_score(y_score_final, y_true, print_result=True)

pprint(f'Final time {time.process_time() - st}')
