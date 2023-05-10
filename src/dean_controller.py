import numpy as np
import tensorflow.keras as keras

from src.dean_ensemble import DeanTsEnsemble
import pickle


class DeanTsController:
    def __init__(self, config: dict, dataset_name=None, verbose=False):
        self.config = config
        np.random.seed(config['seed'])
        keras.utils.set_random_seed(config['seed'])

        self.dataset_name = dataset_name
        self.verbose = verbose
        self.ensemble = DeanTsEnsemble(config)

    @staticmethod
    def load(path, verbose=True) -> 'DeanTsController':
        if verbose:
            print('\n Load model:')
        return pickle.load(open(path, 'rb'))

    def save(self, path):
        if self.verbose:
            print('\n Save model:')
        pickle.dump(self, open(path, 'wb'))

    def train(self, train_data):
        if self.verbose:
            print('Start training submodels \n')
        self.ensemble.train_models(train_data,
                                   subsampling=self.config['subsampling'],
                                   feature_bagging=self.config['feature_bagging'])

    def predict(self, test_data):
        if self.verbose:
            print('Start prediction for submodels \n')
        self.ensemble.predict_with_submodels(test_data, reverse_window=self.config['reverse_window'])

        if self.verbose:
            print('Compute ensemble score\n')
        self.ensemble.compute_ensemble_score(method=self.config['combination_method'])
        return self.ensemble.ensemble_score
