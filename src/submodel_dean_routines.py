import tensorflow.keras as keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mse
from keras.optimizers import Adam
import numpy as np


def build_submodel(unit_sizes, reg=None, act='relu', mean=1.0, lr=0.01) -> Model:
    """ Builds basic dean submodel
    """
    # Define dnn layer architecture
    inputs = Input(shape=(unit_sizes[0],))

    outputs = inputs
    for units in unit_sizes[1:-1]:
        outputs = Dense(units,
                        activation=act,
                        use_bias=True,
                        kernel_initializer=keras.initializers.TruncatedNormal(),
                        kernel_regularizer=reg)(outputs)

    outputs = Dense(unit_sizes[-1],
                    activation='linear',
                    use_bias=False,
                    kernel_initializer=keras.initializers.TruncatedNormal(),
                    kernel_regularizer=reg)(outputs)

    # Define loss function
    target = keras.backend.ones_like(outputs) * mean
    loss = keras.backend.mean(
        mse(outputs, target)
    )

    # Build submodel
    model = Model(inputs, outputs)
    model.add_loss(loss)
    model.compile(Adam(learning_rate=lr))

    return model


def train_submodel(model: Model, train: np.ndarray, hyper, result_dir: str, verbosity=0) -> keras.callbacks.History:
    """ Trains submodel and returns history object
    """
    # Define callbacks
    cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
           keras.callbacks.TerminateOnNaN(),
           keras.callbacks.ModelCheckpoint(f'{result_dir}/model.tf',
                                           monitor='val_loss',
                                           verbose=verbosity,
                                           save_best_only=True,
                                           save_weights_only=True)]

    # Print model summary
    # model.summary()

    # Train the model
    history = model.fit(train, None,
                        epochs=500,
                        batch_size=hyper['batch'],
                        validation_split=0.25,
                        verbose=verbosity,
                        callbacks=cbs)

    return history


def score(model: Model, train: np.ndarray, test: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Return y_score (order: first normal scores than anomalous ones)
    """
    # Predict the output of our datasets
    predict_train = model.predict(train)
    predict_test = model.predict(test)

    # Average out the last dimension, to get one value for each samples
    predict_train_singular = np.mean(predict_train, axis=-1)
    predict_test_singular = np.mean(predict_test, axis=-1)

    # Calculate the mean training prediction (q in the paper)
    q = np.mean(predict_train_singular)

    # Calculate deviation of predictions to the mean
    train_deviations = np.abs(predict_train_singular - q)
    test_deviations = np.abs(predict_test_singular - q)

    return train_deviations, test_deviations
