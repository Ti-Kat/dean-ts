import tensorflow.keras as keras
from keras.callbacks import History
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mse
from keras.optimizers import Adam
import numpy as np

from src.data_processing import add_lag_features


class DeanTsLagModel:
    def __init__(self, lag_indices, look_back):
        self.lag_indices = lag_indices
        self.look_back = look_back
        self.q: float = 1.0

        self.model: Model | None = None
        self.history: History | None = None
        self.scores: np.ndarray | None = None
        self.scores_window: np.ndarray | None = None

    def preprocess_data(self, data):
        data = add_lag_features(data, self.lag_indices)
        return data[self.look_back:]

    def build_submodel(self, unit_sizes, reg=None, act='relu', mean=1.0, lr=0.01):
        """ Builds basic dean submodel
        """
        # Sett DNN architecture
        inputs = Input(shape=(unit_sizes[0],))

        outputs = inputs
        for units in unit_sizes[1:-1]:
            outputs = Dense(units,
                            activation=act,
                            use_bias=False,
                            kernel_initializer=keras.initializers.TruncatedNormal(),
                            kernel_regularizer=reg)(outputs)

        outputs = Dense(unit_sizes[-1],
                        activation='linear',
                        use_bias=False,
                        kernel_initializer=keras.initializers.TruncatedNormal(),
                        kernel_regularizer=reg)(outputs)

        # Set loss function
        target = keras.backend.ones_like(outputs) * mean
        loss = keras.backend.mean(
            mse(outputs, target)
        )

        # Build submodel
        model = Model(inputs, outputs)
        model.add_loss(loss)
        model.compile(Adam(learning_rate=lr))

        self.model = model

    def train(self, train_data: np.ndarray, epochs=500, batch_size=32, validation_split=0.25, verbosity=0):
        """ Trains submodel and sets history object
        """
        # Preprocess train data
        train_data = self.preprocess_data(train_data)

        # Define callbacks
        cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
               keras.callbacks.TerminateOnNaN()]

        # Print model summary
        # model.summary()

        # Train the model
        self.history = self.model.fit(train_data, None,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      verbose=verbosity,
                                      callbacks=cbs)
        # Set q
        predict_train = self.model.predict(train_data)
        predict_train_singular = np.mean(predict_train, axis=-1)
        self.q = np.mean(predict_train_singular)

    def score(self, test: np.ndarray):
        """ Scores test set and sets scores_window and scores accordingly
        """
        # Preprocess test data
        test = self.preprocess_data(test)

        # Predict the output of our datasets
        predict_test = self.model.predict(test)

        # Average out the last dimension, to get one value for each samples
        predict_test_singular = np.mean(predict_test, axis=-1)

        # Calculate deviation of predictions to q
        test_deviations = np.abs(predict_test_singular - self.q)

        # Scale to [0,1]
        test_deviations /= np.max(test_deviations)

        self.scores_window = test_deviations
        self.reverse_window(test.shape)

    def reverse_window(self, test_shape):
        # TODO: Implement this
        scores = np.zeros(test_shape[0])
        self.scores = scores
