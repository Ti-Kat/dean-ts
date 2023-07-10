import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from src.dean_submodel import DeanTsSubmodel


class DeanTsEnsemble:
    def __init__(self, config: dict):
        self.config: dict = config
        self.scaler: StandardScaler = StandardScaler()
        self.submodels: dict[int, DeanTsSubmodel] = {}

        self.submodel_scores: np.ndarray | None = None
        self.ensemble_score: np.ndarray | None = None

    def train_models(self, train_data: np.ndarray, subsampling=None, feature_bagging=False):
        if self.config['tsd']:
            decomposition = STL(train_data[:, 1], period=self.config['period']).fit()
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            resid = decomposition.resid
            train_data = np.c_[train_data[:, 0], trend, seasonal, resid, train_data[:, -1]]

        # Drop "timestamp" and "is_anomaly" columns
        train_data = np.delete(train_data, obj=[0, -1], axis=1)

        # Standardize training data and prepare scaler
        train_data = self.scaler.fit_transform(train_data)

        train_ranges = []
        if subsampling == 'structured':
            # Prepare structured subsampling by determining training ranges
            ss_r = np.array(self.config['ss_r'])
            ss_m = np.array(self.config['ss_m'])
            sample_sizes = (train_data.shape[0] * ss_r).astype(int)

            for i, r_i in enumerate(ss_r):
                start = 0
                for _ in range(0, int(1 / r_i)):
                    train_ranges += [(start, start + sample_sizes[i] - 1) for _ in range(ss_m[i])]
                    start += sample_sizes[i]

        channel_count = train_data.shape[1]
        for i in range(0, self.config['ensemble_size']):
            # Randomly choose look_back range and lag_indices for submodel
            look_back = np.random.randint(self.config['look_back_range'][0],
                                          self.config['look_back_range'][1] + 1)

            lag_indices = np.random.choice(range(1, look_back),
                                           size=self.config['lag_indices_count'],
                                           replace=False)

            # Apply subsampling
            train_range = None
            if subsampling == 'random':
                train_size = np.random.randint(self.config['rs_range'][0],
                                               self.config['rs_range'][1] + 1)
                train_start_index = np.random.randint(0, train_data.shape[0] - train_size)
                train_range = (train_start_index, train_start_index + train_size)
            if subsampling == 'structured':
                train_range = train_ranges[i]

            # Feature bagging
            features = None
            feature_count = channel_count
            if feature_bagging and channel_count > 1:
                lower = min(self.config['fb_range'][0], channel_count)
                upper = min(self.config['fb_range'][1], channel_count)
                feature_count = np.random.randint(lower, upper + 1)
                features = np.random.choice(range(0, channel_count),
                                            size=feature_count,
                                            replace=False)

            # Build and train model
            submodel = DeanTsSubmodel(lag_indices=lag_indices,
                                      look_back=look_back,
                                      train_range=train_range,
                                      features=features)

            submodel.build_submodel([(self.config['lag_indices_count'] + 1) * feature_count] * self.config['depth'],
                                    act=self.config['activation'],
                                    lr=self.config['lr'],
                                    bias=self.config['bias'])

            submodel.train(train_data, batch_size=self.config['batch'])

            self.submodels[i] = submodel

    def predict_with_submodels(self, test_data: np.ndarray, reverse_window=True):
        if self.config['tsd']:
            decomposition = STL(test_data[:, 1], period=self.config['period']).fit()
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            resid = decomposition.resid
            test_data = np.c_[test_data[:, 0], trend, seasonal, resid, test_data[:, -1]]

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
                self.submodel_scores[i, submodel.look_back:] = submodel.scores_window

    def compute_ensemble_score(self, method='thresh', weights=None, threshold=0):
        if method == 'average':
            ensemble_score = np.average(self.submodel_scores, weights=weights, axis=0)
        elif method == 'dean':
            ensemble_score = np.sqrt(np.sum(self.submodel_scores ** 2, axis=0)) / self.config['ensemble_size']
        elif method == 'max':
            ensemble_score = np.nanmax(self.submodel_scores, axis=0)
        elif method == 'median':
            ensemble_score = np.median(self.submodel_scores, axis=0)
        elif method == 'thresh':
            from scipy.stats import zscore
            z_scores = np.nan_to_num(zscore(self.submodel_scores, nan_policy='omit', axis=1))
            z_scores[z_scores < threshold] = 0
            ensemble_score = np.sum(z_scores, axis=0)
        else:
            raise Exception('No valid ensemble combination method selected!')

        # Normalize to [0,1]
        self.ensemble_score = ensemble_score / np.max(ensemble_score)
