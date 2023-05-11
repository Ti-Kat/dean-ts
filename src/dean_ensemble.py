import numpy as np
from sklearn.preprocessing import StandardScaler

from src.dean_submodel import DeanTsLagModel


class DeanTsEnsemble:
    def __init__(self, config: dict):
        self.config: dict = config
        self.scaler: StandardScaler = StandardScaler()
        self.submodels: dict[int, DeanTsLagModel] = {}

        self.submodel_scores: np.ndarray | None = None
        self.ensemble_score: np.ndarray | None = None

    def train_models(self, train_data: np.ndarray, subsampling=None, feature_bagging=False):
        # Drop "timestamp" and "is_anomaly" columns
        train_data = np.delete(train_data, obj=[0, -1], axis=1)

        # Standardize training data and prepare scaler
        train_data = self.scaler.fit_transform(train_data)

        channel_count = train_data.shape[1]
        for i in range(0, self.config['ensemble_size']):
            lag_indices = np.random.choice(range(1, self.config['look_back']),
                                           size=self.config['bag'] - 1,
                                           replace=False)
            train_range = None
            if subsampling == 'vs':
                train_size = np.random.randint(self.config['vs_lower'],
                                               self.config['vs_upper'] + 1)
                train_start_index = np.random.randint(0, train_data.shape[0] - train_size)
                train_range = (train_start_index, train_start_index + train_size)
            elif subsampling == 'fixed':
                pass

            features = None
            feature_count = channel_count
            if feature_bagging and channel_count > 1:
                lower = min(self.config['fb_lower'], channel_count)
                upper = min(self.config['fb_upper'], channel_count)
                feature_count = np.random.randint(lower, upper+1)
                features = np.random.choice(range(0, channel_count),
                                            size=feature_count,
                                            replace=False)

            submodel = DeanTsLagModel(lag_indices=lag_indices,
                                      look_back=self.config['look_back'],
                                      train_range=train_range,
                                      features=features)

            submodel.build_submodel([self.config['bag'] * feature_count] * self.config['depth'],
                                    lr=self.config['lr'],
                                    bias=self.config['bias'])

            submodel.train(train_data, batch_size=self.config['batch'])

            self.submodels[i] = submodel

    def predict_with_submodels(self, test_data: np.ndarray, reverse_window=True):
        # Drop "timestamp" and "is_anomaly" columns
        test_data = np.delete(test_data, obj=[0, -1], axis=1)

        # Standardize test data by scaler fitted to training data
        test_data = self.scaler.transform(test_data)

        self.submodel_scores = np.zeros(shape=(self.config['ensemble_size'], test_data.shape[0]))
        for i in range(0, self.config['ensemble_size']):
            submodel = self.submodels[i]
            submodel.score(test_data)
            if reverse_window:
                self.submodel_scores[i] = submodel.scores
            else:
                self.submodel_scores[i, self.config['look_back']:] = submodel.scores_window

    def compute_ensemble_score(self, method='thresh', weights=None, threshold=0):
        if method == 'average':
            self.ensemble_score = np.average(self.submodel_scores, weights=weights, axis=0)
        elif method == 'max':
            self.ensemble_score = np.max(self.submodel_scores, axis=0)
        elif method == 'median':
            self.ensemble_score = np.median(self.submodel_scores, axis=0)
        elif method == 'thresh':
            from scipy.stats import zscore
            z_scores = zscore(self.submodel_scores, axis=1)
            z_scores[z_scores < threshold] = self.config['thresh_value']
            z_score_sum = np.sum(z_scores, axis=0)
            self.ensemble_score = z_score_sum / np.max(z_score_sum)
