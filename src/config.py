import os
import yaml

import numpy as np
import tensorflow.keras as keras


class Config:

    def __init__(self):
        self.path = os.getcwd()
        with open('../config/configuration.yaml') as config_file:
            data = yaml.safe_load(config_file)
        config_keys = data.keys()
        for k in config_keys:
            setattr(self, k, data.get(k))

        # Set base configuration
        np.random.seed(self['seed'])
        keras.utils.set_random_seed(self['seed'])

    def __getitem__(self, key):
        return getattr(self, key)


config = Config()
