import sys
import os
import shutil
import time
import yaml

import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.losses import mse
from keras.optimizers import Adam, SGD, RMSprop

import numpy as np

# Load hyperparameter configuration
hyper = yaml.safe_load(open('../../config/hyper.yaml'))

# Load data, and change the shape into (samples, features)
from loaddata import load_data

(x_train_init, y_train), (x_test_init, y_test) = load_data()
if len(x_train_init.shape) > 2:
    x_train_init = np.reshape(x_train_init, (x_train_init.shape[0], np.prod(x_train_init.shape[1:])))
    x_test_init = np.reshape(x_test_init, (x_test_init.shape[0], np.prod(x_test_init.shape[1:])))


# Train one model and save it at result_path
def train(result_path):
    x_train, x_test = x_train_init.copy(), x_test_init.copy()

    # Choose feature subset for current model (without replacement)
    number_of_dims = int(x_train.shape[1])
    chosen_dims = np.random.choice([i for i in range(number_of_dims)], hyper['bag'], replace=False)

    x_train = np.concatenate([np.expand_dims(x_train[:, use], axis=1) for use in chosen_dims], axis=1)
    x_test = np.concatenate([np.expand_dims(x_test[:, use], axis=1) for use in chosen_dims], axis=1)

    # Normalize the data, so that the mean is zero, and the standard deviation is one
    def normalize(q):
        # TODO: Probably use this instead?
        # return (q - np.mean(q)) / np.std(q)
        return (q - np.mean(x_train)) / np.std(x_train)

    #
    def get_data(x, y, norm=True, norm_dex=7, n=-1):
        if norm:
            ids = np.where(y == norm_dex)
        else:
            ids = np.where(y != norm_dex)
        qx = x[ids]
        if n > 0:
            qx = qx[:n]
        qy = np.reshape(qx, (int(qx.shape[0]), hyper['bag']))
        return normalize(qy)

    # Split data into normal and abnormal samples. Train only on normal ones.
    train_data = get_data(x_train, y_train, norm=True, norm_dex=hyper['dex'])
    at = get_data(x_test, y_test, norm=False, norm_dex=hyper['dex'])
    t = get_data(x_test, y_test, norm=True, norm_dex=hyper['dex'])

    # Build one base detector neural network model via tensorflow
    def build_model(q, reg=None, act='relu', mean=1.0):
        inn = Input(shape=(hyper['bag'],))
        w = inn
        for aq in q[1:-1]:
            # Change this line to use constant shifts
            # w=Dense(aq,activation=act,use_bias=True,kernel_initializer=keras.initializers.TruncatedNormal(),kernel_regularizer=reg)(w)
            w = Dense(aq,
                      activation=act,
                      use_bias=False,
                      kernel_initializer=keras.initializers.TruncatedNormal(),
                      kernel_regularizer=reg)(w)
        w = Dense(q[-1],
                  activation='linear',
                  use_bias=False,
                  kernel_initializer=keras.initializers.TruncatedNormal(),
                  kernel_regularizer=reg)(w)
        m = Model(inn, w, name='oneoff')
        zero = K.ones_like(w) * mean
        loss = mse(w, zero)
        loss = K.mean(loss)
        m.add_loss(loss)
        m.compile(Adam(learning_rate=hyper['lr']))
        return m

    # Build the model
    m = build_model([hyper['bag'] for _ in range(hyper['depth'])],
                    reg=None,
                    act='relu',
                    mean=1.0)

    # Define callbacks
    cbs = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
           keras.callbacks.TerminateOnNaN(),
           keras.callbacks.ModelCheckpoint(f'{result_path}/model.tf',
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True)]

    # Print model summary
    m.summary()

    # Train the model
    history = m.fit(train_data, None,
                    epochs=500,
                    batch_size=hyper['batch'],
                    validation_split=0.25,
                    verbose=1,
                    callbacks=cbs)

    # Predict the output of our datasets
    pain = m.predict(train_data)
    p = m.predict(t)
    w = m.predict(at)

    # Average out the last dimension, to get one value for each samples
    ppain = np.mean(pain, axis=-1)
    pp = np.mean(p, axis=-1)
    ww = np.mean(w, axis=-1)

    from sklearn.metrics import roc_auc_score as auc

    # Calculate the mean prediction (q in the paper)
    m = np.mean(ppain)

    # and the deviation of each to the mean
    pd = np.abs(pp - m)  # if this worked, the values in the array pd should be much smaller
    wd = np.abs(ww - m)  # than in the array wd
    y_score = np.concatenate((pd, wd))
    y_true = np.concatenate((np.zeros_like(pp), np.ones_like(ww)))

    # Calculate auc score of a single model
    auc_score = auc(y_true, y_score)
    print(f'reached auc of {auc_score}')

    # and save the necessary results for merge.py to combine the submodel predictions into an ensemble
    np.savez_compressed(f'{result_path}/result.npz', y_true=y_true, y_score=y_score, chosen_dims=chosen_dims)


# Initiate training multiple models
if __name__ == '__main__':
    # This allows calling train.py with up to two arguments, specifying the indices where the results will be stored.
    # Using a 2nd argument quickly overrides the specified number of rounds.
    result_index_lower = 0
    result_index_upper = hyper['rounds']

    if len(sys.argv) > 1:
        result_index_lower = int(sys.argv[1])
        result_index_upper = result_index_lower + hyper['rounds']
    if len(sys.argv) > 2:
        result_index_upper = result_index_lower + int(sys.argv[2])

    for training_round in range(result_index_lower, result_index_upper):
        path = f'../results/{training_round}/'

        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=False)

        train(path)
