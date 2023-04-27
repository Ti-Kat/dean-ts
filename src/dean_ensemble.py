import numpy as np
from sklearn.preprocessing import StandardScaler

from src.dean_submodel import DeanTsLagModel


class DeanTsEnsemble:
    def __init__(self, config: dict, train_data: np.ndarray):
        self.ensemble_score = None
        self.config = config
        self.scaler = StandardScaler()
        self.train_data = self.scaler.fit_transform(train_data)

        self.submodels: dict[int, DeanTsLagModel] = {}
        self.submodel_scores = None

    def train_models(self):
        # Drop "timestamp" and "is_anomaly" columns
        train_data = np.delete(self.train_data, obj=[0, -1], axis=1)
        channel_count = train_data.shape[1]

        for i in range(0, self.config['ensemble_size']):
            lag_indices = np.random.choice(range(1, self.config['look_back']),
                                           size=self.config['bag'] - 1,
                                           replace=False)

            submodel = DeanTsLagModel(lag_indices=lag_indices,
                                      look_back=self.config['look_back'])

            submodel.build_submodel([self.config['bag'] * channel_count] * self.config['depth'])

            submodel.train(train_data)

            self.submodels[i] = submodel

    def predict_with_submodels(self, test_data: np.ndarray):
        # Drop "timestamp" and "is_anomaly" columns
        test_data = np.delete(test_data, obj=[0, -1], axis=1)

        # Standardize
        test_data = self.scaler.transform(test_data)

        for i in range(0, self.config['ensemble_size']):
            submodel = self.submodels[i]
            submodel.score(test_data)
            self.submodel_scores[i, self.config['look_back']:] = submodel.scores_window

    def compute_ensemble_score(self):
        self.ensemble_score = np.mean(self.submodel_scores, axis=0)
