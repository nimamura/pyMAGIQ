# r_seed = 119 # 119 is a good example
r_seed = 122 # 121 and 122 are good examples

import numpy as np
import tensorflow as tf
import random as rn

import os
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
import csv
import json

import optuna
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import model_from_json
from keras.utils import plot_model
from keras.utils import np_utils
from keras import initializers

# Local Import
import pyMAGIQ

#************************************************************************/
def neuralnet(num_layer, activation, mid_units, reg_val, reg_ker):

    nrate = 5
    nfreq = 30
    ndata = 4
    ncomp = 2
    epochs = 300
    batch_size = 32

    # regularization parameters
    # reg_val = 0.000001
    # reg_ker = 0.00003

    initializers.Initializer()
    initializers.RandomNormal(seed=r_seed)

    # initialize model
    model = Sequential()

    # add layers
    model.add(Dense( activation=activation, units=mid_units,
                     kernel_regularizer=regularizers.l1(10**(reg_ker)),
                     input_dim=ndata*ncomp*nfreq*2))

    for i in range(num_layer):
        model.add(Dense( activation=activation,
                        kernel_regularizer=regularizers.l1(10**(reg_val)),
                        units=mid_units ))

    model.add(Dense( activation='softmax', units=nrate,
                     kernel_regularizer=regularizers.l1(10**(reg_ker))))

    return model

#************************************************************************/
def objective(trial):
    # clear session
    K.clear_session()

    nrate = 5
    nfreq = 30
    ndata = 4
    ncomp = 2
    epochs = 300
    batch_size = 32

    # datadir = '/Users/nimamura/Library/Mobile Documents/com~apple~CloudDocs/pyMAGIQ/pyMAGIQ/survey/USArray2019March'
    datadir = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'

    traindir    = datadir + '/train'
    # X_trainpath = datadir + '/preprocessed/X_train.csv'
    # y_trainpath = datadir + '/preprocessed/y_train.csv'
    # rate_trainpath  = datadir + '/preprocessed/rate_train.csv'
    # SiteIDpath  = datadir + '/preprocessed/siteID_train.csv'
    # outputpath = datadir +'/outputs'

    # when using data augumentation
    X_trainpath = datadir + '/preprocessed/X_train_aug.csv'
    y_trainpath = datadir + '/preprocessed/y_train_aug.csv'
    rate_trainpath  = datadir + '/preprocessed/rate_train_aug.csv'
    SiteIDpath  = datadir + '/preprocessed/siteID_train_aug.csv'
    outputpath = datadir +'/outputs'

    #************************************************************************/
    # Loading X and y data in training datasets
    df=pd.read_csv(X_trainpath, sep=',',header=None)
    X_train = df.values

    df=pd.read_csv(y_trainpath, sep=',',header=None)
    y_train = df.values[:,0]

    df=pd.read_csv(SiteIDpath, sep=',',header=None)
    SiteID_train = df.values[:,0]

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nrate)

    #************************************************************************/
    # sort X and y in random
    # This is necessary for the validation_split as validation_split splits only tail part
    X_random, y_random, randindex = pyMAGIQ.utils.MAGIQlib.MAGIQlib.randomsortXY(X_train, y_train)

    #最適化するパラメータの設定
    #畳込み層の数
    num_layer = trial.suggest_int("num_layer", 3, 9)

    #Hidden層のユニット数
    mid_units = int(trial.suggest_discrete_uniform("mid_units", 10, 80, 10))

    #活性化関数
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

    #optimizer
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

    #reg_val
    reg_val = int(trial.suggest_discrete_uniform("reg_val", -8, -4, -1))

    #reg_ker
    reg_ker = int(trial.suggest_discrete_uniform("reg_ker", -7, -2, -1))

    model = neuralnet(num_layer, activation, mid_units, reg_val, reg_ker)
    model.compile(optimizer=optimizer,
          loss="categorical_crossentropy",
          metrics=["accuracy"])

    history = model.fit(X_random, y_random, verbose=0, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    #検証用データに対する正答率が最大となるハイパーパラメータを求める
    return 1 - history.history["val_acc"][-1]

def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=10)
    print('best_params')
    print(study.best_params)
    print('-1 x best_value')
    print(-study.best_value)

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))


if __name__ == '__main__':
    main()
#************************************************************************/
# parameter in neural network
# nunits = 40

# nrate = 5
# nfreq = 30
# ndata = 4
# ncomp = 2
# epochs = 300
# batch_size = 32

# number of bagging
# rounds = 1
# rounds = 5

#************************************************************************/
# setting directories
# frequency sets for USArray (Currently, this code works only with these frequencies)
# freqsets = [  1.367187E-01, 1.093750E-01, 8.593753E-02, 6.640627E-02, 5.078124E-02, 3.906250E-02,
#         3.027344E-02, 2.343750E-02, 1.855469E-02, 1.464844E-02, 1.171875E-02, 9.765625E-03,
#         7.568361E-03, 5.859374E-03, 4.638671E-03, 3.662109E-03, 2.929688E-03, 2.441406E-03,
#         1.892090E-03, 1.464844E-03, 1.159668E-03, 9.155271E-04, 7.324221E-04, 6.103516E-04,
#         4.425049E-04, 3.204346E-04, 2.136230E-04, 1.373291E-04, 8.392331E-05, 5.340577E-05]

# # number of frequency used in data
# nfreq = len(freqsets)

# datadir = '/Users/nimamura/Library/Mobile Documents/com~apple~CloudDocs/pyMAGIQ/pyMAGIQ/survey/USArray2019March'
# datadir = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'

# traindir    = datadir + '/train'
# X_trainpath = datadir + '/preprocessed/X_train.csv'
# y_trainpath = datadir + '/preprocessed/y_train.csv'
# rate_trainpath  = datadir + '/preprocessed/rate_train.csv'
# SiteIDpath  = datadir + '/preprocessed/siteID_train.csv'
# outputpath = datadir +'/outputs'

# when using data augumentation
# X_trainpath = datadir + '/preprocessed/X_train_aug.csv'
# y_trainpath = datadir + '/preprocessed/y_train_aug.csv'
# rate_trainpath  = datadir + '/preprocessed/rate_train_aug.csv'
# SiteIDpath  = datadir + '/preprocessed/siteID_train_aug.csv'
# outputpath = datadir +'/outputs'

# #************************************************************************/
# # Loading X and y data in training datasets
# df=pd.read_csv(X_trainpath, sep=',',header=None)
# X_train = df.values
#
# df=pd.read_csv(y_trainpath, sep=',',header=None)
# y_train = df.values[:,0]
#
# df=pd.read_csv(SiteIDpath, sep=',',header=None)
# SiteID_train = df.values[:,0]
#
# # convert class vectors to binary class matrices
# y_train = np_utils.to_categorical(y_train, nrate)
#
# #************************************************************************/
# # sort X and y in random
# # This is necessary for the validation_split as validation_split splits only tail part
# X_random, y_random, randindex = pyMAGIQ.utils.MAGIQlib.MAGIQlib.randomsortXY(X_train, y_train)

# read model
# model = neu=ralnet()
#
# # setting optimization
# model.compile(loss="categorical_crossentropy",
#               optimizer=Adamax(),
#               metrics=["accuracy"])
#
# model.summary()
# # learning parameters
# history = model.fit(X_random, y_random, epochs=epochs,
#                     validation_split=0.1, batch_size=batch_size)
#
# val_acc = max(history.history['val_acc'])
#
# pyMAGIQ.utils.MAGIQlib.MAGIQlib.plot_history(history,outputpath+'/acc_loss.eps')
# print('max val accuracy :',max(history.history['val_acc']))
#
# # plot_model returns error when using python 3, so commented out.
# # save model in png file
# # plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
#
# # save model
# model_json = model.to_json()
# with open(outputpath+"/model.json",mode='w') as f:
#     f.write(model_json)
#
# # save weight
# model.save_weights(outputpath+'/weights.hdf5')
#
# #************************************************************************/
# # Human vs AI
# # take validation model and rate
# ndata_val = int(len(X_train)*0.1)
# Xval = X_random[-ndata_val:-1,:]
# yval = y_random[-ndata_val:-1]
# index_val = randindex[-ndata_val:-1]
#
# # read model
# model = model_from_json(open(outputpath+'/model.json').read())
#
# # read weights
# model.load_weights(outputpath+'/weights.hdf5')
#
# model.summary()
#
# # # setting optimization
# # model.compile(loss="categorical_crossentropy",
# #               optimizer=Adamax(),
# #               metrics=["accuracy"])
# #evaluate unrated
# yval_est  = model.predict(Xval)
#
# pyMAGIQ.utils.MAGIQlib.MAGIQlib.ratehistogram2d(yval_est,yval,5,fpath=outputpath+'/hist2d.eps')
