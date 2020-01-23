r_seed = 122  # 119, 121 and 122 are good examples

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(r_seed)
rn.seed(r_seed)
tf.set_random_seed(r_seed)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
import optuna
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.utils import np_utils
from keras import initializers

# Local Import
import pyMAGIQ


# ************************************************************************/
def neuralnet(num_layer, activation, mid_units, reg_val, reg_ker):
    """
    construct neural network here
    """
    nrate = 5
    nfreq = 30
    ndata = 4
    ncomp = 2

    initializers.Initializer()
    initializers.RandomNormal(seed=r_seed)

    # initialize model
    model = Sequential()

    # add layers
    model.add(Dense(activation=activation, units=mid_units,
                    kernel_regularizer=regularizers.l1(10**(reg_ker)),
                    input_dim=ndata*ncomp*nfreq*2))

    for i in range(num_layer):
        model.add(Dense(activation=activation,
                        kernel_regularizer=regularizers.l1(10**(reg_val)),
                        units=mid_units))

    model.add(Dense(activation='softmax', units=nrate,
                    kernel_regularizer=regularizers.l1(10**(reg_ker))))

    return model


# ************************************************************************/
def objective(trial):
    # clear session
    K.clear_session()

    nrate = 5
    epochs = 300
    batch_size = 32

    datadir = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'

    # X_trainpath = datadir + '/preprocessed/X_train.csv'
    # y_trainpath = datadir + '/preprocessed/y_train.csv'
    # SiteIDpath  = datadir + '/preprocessed/siteID_train.csv'

    # when using data augumentation
    X_trainpath = datadir + '/preprocessed/X_train_aug.csv'
    y_trainpath = datadir + '/preprocessed/y_train_aug.csv'
    SiteIDpath = datadir + '/preprocessed/siteID_train_aug.csv'

    # ************************************************************************/
    # Loading X and y data in training datasets
    df = pd.read_csv(X_trainpath, sep=',', header=None)
    X_train = df.values

    df = pd.read_csv(y_trainpath, sep=',', header=None)
    y_train = df.values[:, 0]

    df = pd.read_csv(SiteIDpath, sep=',', header=None)

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nrate)

    # ************************************************************************/
    # sort X and y in random
    # This is necessary for the validation_split as validation_split splits only tail part
    X_random, y_random, randindex = pyMAGIQ.utils.MAGIQlib.MAGIQlib.randomsortXY(X_train, y_train)

    # number of parameters to optimize
    # number of layers
    num_layer = trial.suggest_int("num_layer", 3, 9)

    # number of units in hidden layers
    mid_units = int(trial.suggest_discrete_uniform("mid_units", 10, 80, 10))

    # activation function
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

    # optimizer
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

    # reg_val
    reg_val = int(trial.suggest_discrete_uniform("reg_val", -8, -4, -1))

    # reg_ker
    reg_ker = int(trial.suggest_discrete_uniform("reg_ker", -7, -2, -1))

    model = neuralnet(num_layer, activation, mid_units, reg_val, reg_ker)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_random, y_random, verbose=0, epochs=epochs,
                        batch_size=batch_size, validation_split=0.1)

    # finds hyper parameters to maximize val_acc
    return 1 - history.history["val_acc"][-1]


# ************************************************************************/
def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=10)
    print('best_params')
    print(study.best_params)
    print('-1 x best_value')
    print(-study.best_value)

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x: x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))


# ************************************************************************/
if __name__ == '__main__':
    main()
