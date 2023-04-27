from src.dean_ensemble import DeanTsEnsemble
import pickle


class DeanTsController:
    def __init__(self, config, test, train, dataset_name=None, verbose=False):
        self.config = config
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.ensemble = DeanTsEnsemble(config, test, train)

    @staticmethod
    def load(path, verbose=True) -> 'DeanTsController':
        if verbose:
            print('\n Load model:')
        return pickle.load(open(path + '/model.p', 'rb'))

    def save(self, path):
        if self.verbose:
            print('\n Save model:')
        pickle.dump(self, open(path + '/model.p', 'wb'))

    def train(self):
        if self.verbose:
            print('Start training submodels \n')
        self.ensemble.train_models()

    def predict(self):
        if self.verbose:
            print('Start prediction for submodels \n')
        self.ensemble.predict_with_submodels()

        if self.verbose:
            print('Compute ensemble score\n')
        self.ensemble.compute_ensemble_score()
        return self.ensemble.ensemble_score
