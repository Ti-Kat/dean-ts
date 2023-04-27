import numpy as np

from src.config import Config
from src.data_processing import standardize
from src.dean_submodel import DeanTsSubmodel


class DeanTsEnsemble:
    def __init__(self, config: Config, test_data: np.ndarray, train_data: np.ndarray):
        self.config = config
        self.test_data, self.train_data = standardize(test_data,
                                                      train_data)
        self.submodels: dict[int, DeanTsSubmodel] = {}
        self.submodel_scores = np.zeros(shape=(config['ensemble_size'], test_data.shape[0]))
        self.ensemble_score = np.zeros(shape=test_data.shape[0])

    def train_models(self):
        print('Start training submodels \n')

        # Drop "timestamp" and "is_anomaly" columns
        train_data = np.delete(self.train_data, obj=[0, -1], axis=1)
        channel_count = train_data.shape[1]

        for i in range(0, self.config['ensemble_size']):
            lag_indices = np.random.choice(range(1, self.config['look_back']),
                                           size=self.config['bag'] - 1,
                                           replace=False)

            submodel = DeanTsSubmodel(lag_indices=lag_indices,
                                      look_back=self.config['look_back'])

            submodel.build_submodel([self.config['bag'] * channel_count] * self.config['depth'])

            submodel.train(train_data)

            self.submodels[i] = submodel

    def predict_with_submodels(self):
        print('Start prediction for submodels \n')

        # Drop "timestamp" and "is_anomaly" columns
        test_data = np.delete(self.test_data, obj=[0, -1], axis=1)

        for i in range(0, self.config['ensemble_size']):
            submodel = self.submodels[i]
            submodel.score(test_data)
            self.submodel_scores[i, self.config['look_back']:] = submodel.scores_window

    def compute_ensemble_score(self):
        self.ensemble_score = np.mean(self.submodel_scores, axis=0)
