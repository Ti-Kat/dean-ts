import numpy as np
import yaml

from config import config
from data_persistence import create_result_dir, store_results, load_dataset
from data_processing import split_train_test, add_lag_features
from model_evaluation import compute_auc_score
from submodel_dean_routines import build_submodel, train_submodel, score


class DeanTsController:
    # def __init__(self, dataset_name):
    #     self.dataset_name = dataset_name
    #
    #     complete_dataset = load_dataset(dataset_name, 'numpy')
    #     self.dataset = dataset
    #
    #
    #     self.path = os.getcwd()
    #     with open('../config/configuration.yaml') as config_file:
    #         data = yaml.safe_load(config_file)
    #     config_keys = data.keys()
    #     for k in config_keys:
    #         setattr(self, k, data.get(k))

        # Set base configuration
        # np.random.seed(self['seed'])
        # keras.utils.set_random_seed(self['seed'])

    @staticmethod
    def run(dataset, y_true, dataset_name, i, y_scores_list):
        for i in range(i, i + config['ensemble_size']):
            result_sub_dir = create_result_dir(dataset_name, model_index=i)

            # Randomly select previous time steps for each channel
            channel_count = dataset.shape[1]
            lag_indices_per_channel = {
                channel: np.random.choice(range(1, config['look_back']),
                                          size=config['bag'] - 1,
                                          replace=False)
                for channel in range(channel_count)
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
            y_score_train, y_score_test = score(submodel, train, test)

            auc_score = compute_auc_score(y_score_test, y_true, print_result=True)

            # Save results
            store_results(result_sub_dir,
                          {
                              'y_score': y_score_test,
                              'auc_score': auc_score,
                              'lag_indices_per_channel': lag_indices_per_channel,
                          })

            y_scores_list.append(y_score_test)
