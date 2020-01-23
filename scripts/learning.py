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

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adamax
from keras import regularizers
from keras.models import model_from_json
from keras.utils import np_utils
from keras import initializers

# Local Import
import pyMAGIQ


# ************************************************************************/
def neuralnet():
    """
    construct neural network here
    """
    # regularization parameters
    reg_val = 0.000001
    reg_ker = 0.00003
    activation = 'relu'

    initializers.Initializer()
    initializers.RandomNormal(seed=r_seed)

    # initialize model
    model = Sequential()

    # add layers
    reg_val = -6
    reg_ker = -5.5
    mid_units = 50
    num_layer = 7
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
# parameter in neural network
nunits = 40

nrate = 5
nfreq = 30
ndata = 4
ncomp = 2
epochs = 500
batch_size = 32

# ************************************************************************/
# setting directories

datadir = '/Users/nimamura/work/pyMAGIQ/survey/USArray2020Jan_test'

traindir = datadir + '/train'
# X_trainpath = datadir + '/preprocessed/X_train.csv'
# y_trainpath = datadir + '/preprocessed/y_train.csv'
# rate_trainpath  = datadir + '/preprocessed/rate_train.csv'
# SiteIDpath  = datadir + '/preprocessed/siteID_train.csv'
# outputpath = datadir +'/outputs'

# when using data augumentation
X_trainpath = datadir + '/preprocessed/X_train_aug.csv'
y_trainpath = datadir + '/preprocessed/y_train_aug.csv'
rate_trainpath = datadir + '/preprocessed/rate_train_aug.csv'
SiteIDpath = datadir + '/preprocessed/siteID_train_aug.csv'
outputpath = datadir + '/outputs'

# ************************************************************************/
# Loading X and y data in training datasets
df = pd.read_csv(X_trainpath, sep=',', header=None)
X_train = df.values

df = pd.read_csv(y_trainpath, sep=',', header=None)
y_train = df.values[:, 0]

df = pd.read_csv(SiteIDpath, sep=',', header=None)
SiteID_train = df.values[:, 0]

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nrate)

# ************************************************************************/
# sort X and y in random
# This is necessary for the validation_split as validation_split splits only tail part
X_random, y_random, SiteID_random, randindex = \
    pyMAGIQ.utils.MAGIQlib.MAGIQlib.randomsortXY(X_train, y_train, SiteID_train)

# read model
model = neuralnet()

# setting optimization
model.compile(loss="categorical_crossentropy",
              optimizer=Adamax(),
              metrics=["accuracy"])

model.summary()
# learning parameters
history = model.fit(X_random, y_random, epochs=epochs,
                    validation_split=0.1, batch_size=batch_size)

val_acc = max(history.history['val_acc'])

pyMAGIQ.utils.MAGIQlib.MAGIQlib.plot_history(history, outputpath+'/acc_loss.eps')
print('max val accuracy :', max(history.history['val_acc']))

# plot_model returns error when using python 3, so commented out.
# save model in png file
# plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

# save model
model_json = model.to_json()
with open(outputpath+"/model.json", mode='w') as f:
    f.write(model_json)

# save weight
model.save_weights(outputpath+'/weights.hdf5')

# ************************************************************************/
# Human vs AI
# take validation model and rate
ndata_val = int(len(X_train)*0.1)
Xval = X_random[-ndata_val:-1, :]
yval = y_random[-ndata_val:-1]
index_val = randindex[-ndata_val:-1]
SiteIDval = SiteID_random[-ndata_val:-1]

# read model
model = model_from_json(open(outputpath+'/model.json').read())

# read weights
model.load_weights(outputpath+'/weights.hdf5')

model.summary()

# evaluate unrated
yval_est = model.predict(Xval)

pyMAGIQ.utils.MAGIQlib.MAGIQlib.ratehistogram2d(yval_est, yval, 5, fpath=outputpath+'/hist2d.eps')
