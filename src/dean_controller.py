import numpy as np

from data_persistence import create_result_dir, store_results
from data_processing import split_train_test, add_lag_features
from src.dean_ensemble import DeanTsEnsemble
from src.utils.scoring import compute_auc_score
from dean_submodel import DeanTsSubmodel


class DeanTsController:
    def __init__(self, config, test, train, dataset_name=None):
        self.config = config
        self.dataset_name = dataset_name
        self.ensemble = DeanTsEnsemble(config, test, train)

    def load(self, path):
        pass

    def save(self, path):
        pass

    def train(self):
        self.ensemble.train_models()

    def predict(self):
        self.ensemble.predict_with_submodels()
        self.ensemble.compute_ensemble_score()
        return self.ensemble.ensemble_score

    # def run(self, dataset, y_true, dataset_name, i, y_scores_list):
    #     for i in range(i, i + self.config['ensemble_size']):
    #         result_sub_dir = create_result_dir(dataset_name, model_index=i)
    #
    #         # Split into train and test
    #         train, test = split_train_test(transformed_set, test_ratio=config['test_ratio'])
    #
    #         DeanTsSubmodel.train_submodel(submodel, train, config, result_sub_dir)
    #
    #         # Score the model
    #         y_score_train, y_score_test = DeanTsSubmodel.score(submodel, train, self.test)
    #
    #         auc_score = compute_auc_score(y_score_test, y_true, print_result=True)
    #
    #         # Save results
    #         store_results(result_sub_dir,
    #                       {
    #                           'y_score': y_score_test,
    #                           'auc_score': auc_score,
    #                           'lag_indices': lag_indices,
    #                       })
    #
    #         y_scores_list.append(y_score_test)
