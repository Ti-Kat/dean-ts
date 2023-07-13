import numpy as np
import tensorflow.keras as keras
from keras.callbacks import History
from keras.layers import Input, Dense, LeakyReLU
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam


class DeanTsSubmodel:
    def __init__(self, lag_indices, look_back, train_range, features):
        self.lag_indices = lag_indices
        self.look_back = look_back
        self.train_range = train_range
        self.features = features
        self.q: float = 1.0

        self.model: Model | None = None
        self.history: History | None = None
        self.scores: np.ndarray | None = None
        self.scores_window: np.ndarray | None = None

    def preprocess_data(self, data):
        """Prepares input data by prepending previous observations according to the specified lag indices"""
        def create_lag_feature(arr: np.ndarray, channel: int, lag: int, replace_value=np.nan) -> np.ndarray:
            e = np.empty((arr.shape[0]))
            if lag >= 0:
                e[:lag] = replace_value
                e[lag:] = arr[:-lag, channel]
            else:
                e[lag:] = replace_value
                e[:lag] = arr[-lag:, channel]
            return e

        def add_lag_features(arr: np.ndarray, lag_indices) -> np.ndarray:
            result = arr.copy()
            for lag in lag_indices:
                for feature_index in range(arr.shape[1]):
                    lag_feature = create_lag_feature(arr, feature_index, lag)
                    result = np.c_[result, lag_feature]
            return result

        # Select feature subset if feature bagging is used
        if self.features is not None:
            data = data[:, self.features]

        # Add lag features
        data = add_lag_features(data, self.lag_indices)
        return data[self.look_back:]

    def build_submodel(self, unit_sizes, reg=None, act='elu', mean=1.0, lr=0.01, bias=False):
        """Builds basic dean submodel"""
        # Set MLP architecture
        inputs = Input(shape=(unit_sizes[0],))

        outputs = inputs
        for units in unit_sizes[1:-1]:
            if act == 'leaky':
                activation = None
            else:
                activation = act

            outputs = Dense(units,
                            activation=activation,
                            use_bias=bias,
                            kernel_initializer=keras.initializers.TruncatedNormal(),
                            kernel_regularizer=reg)(outputs)

            if act == 'leaky':
                outputs = LeakyReLU(alpha=0.1)(outputs)

        outputs = Dense(unit_sizes[-1],
                        activation='linear',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(),
                        kernel_regularizer=reg)(outputs)

        # Set loss function
        target = keras.backend.ones_like(outputs) * mean
        loss = keras.backend.mean(
            mse(target, outputs)
        )

        # Build submodel
        model = Model(inputs, outputs)
        model.add_loss(loss)
        model.compile(Adam(learning_rate=lr))

        self.model = model

    def train(self, train_data: np.ndarray, epochs=500, batch_size=32, validation_split=0.25, verbosity=0):
        """Trains submodel and sets history object"""
        # Preprocess train data
        train_data = self.preprocess_data(train_data)

        if self.train_range:
            train_data = train_data[self.train_range[0]:self.train_range[1]]

        # Define callbacks
        cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               keras.callbacks.TerminateOnNaN()]

        # Print model summary
        # self.model.summary()

        # Train the model
        self.history = self.model.fit(train_data, None,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      shuffle=False,
                                      verbose=verbosity,
                                      callbacks=cbs)
        # Set q
        predict_train = self.model.predict(train_data)
        predict_train_singular = np.mean(predict_train, axis=1)
        self.q = np.mean(predict_train_singular)

    def score(self, test: np.ndarray):
        """Scores test data and sets scores_window and scores accordingly"""
        # Preprocess test data
        test = self.preprocess_data(test)

        # Predict the output of our datasets
        predict_test = self.model.predict(test)

        # Average out the last dimension, to get one value for each samples
        predict_test_singular = np.mean(predict_test, axis=-1)

        # Calculate deviation of predictions to q
        test_deviations = np.nan_to_num(np.abs(predict_test_singular - self.q))

        # Scale to [0,1]
        test_deviations /= np.max(test_deviations)

        self.scores_window = np.nan_to_num(test_deviations)
        self.reverse_window(test.shape)

    def reverse_window(self, test_shape):
        """Reverse the input windows so that each initial data point is assigned the
        average score over each window it was part of
        """
        scores = np.zeros(test_shape[0] + self.look_back)
        denominators = np.zeros(test_shape[0] + self.look_back)
        indices = np.full(shape=self.lag_indices.shape[0], fill_value=self.look_back) - self.lag_indices
        indices = np.append(indices, [self.look_back])

        for i, _ in enumerate(self.scores_window):
            scores[indices + i] += self.scores_window[i]
            denominators[indices + i] += 1

        np.maximum(denominators, 1, out=denominators)
        scores = scores / denominators
        score_max = np.nanmax(scores)
        if score_max > 0:
            self.scores = scores / score_max
        else:
            self.scores = np.zeros(scores.shape)
