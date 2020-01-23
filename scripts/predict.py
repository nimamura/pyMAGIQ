#!/usr/bin/env python
"""
predict rate by using model and weights
"""
R_SEED = 122  # 121 and 122 are good examples as random seeds

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(R_SEED)
rn.seed(R_SEED)
tf.set_random_seed(R_SEED)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
from keras.models import model_from_json

# load xml_unrated dict with pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


# ************************************************************************/
# DATADIR = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'
DATADIR = '/Users/nimamura/work/pyMAGIQ/survey/USArray2020Jan_test'
OUTPATH = DATADIR + '/outputs'

# read model and weights
model = model_from_json(open(OUTPATH + '/model.json').read())
model.load_weights(OUTPATH + '/weights.hdf5')

model.summary()

# ************************************************************************/
# Set unrated data info
unrateddir = DATADIR + '/unrated'
X_unratedpath = DATADIR + '/preprocessed/X_unrated.csv'
y_unratedpath = DATADIR + '/preprocessed/y_unrated.csv'
rate_unratedpath = DATADIR + '/preprocessed/rate_unrated.csv'
SiteIDpath_unrated = DATADIR + '/preprocessed/siteID_unrated.csv'

unratedlists = sorted([f for f in os.listdir(unrateddir) if not f.startswith('.')])

nMTunrated = len(unratedlists)

# ************************************************************************/
# loading unrated data and initialize y for unrated
# read X_unrated
df = pd.read_csv(X_unratedpath, sep=',', header=None)
X_unrated = df.values

with open(OUTPATH + '/xml_unrated.p', 'rb') as fp:
    xml_unrated = pickle.load(fp)

# evaluate unrated
y_unrated = model.predict(X_unrated)
y_unrated = np.argmax(y_unrated, axis=1) + 1
np.savetxt(y_unratedpath, y_unrated, fmt='%d', delimiter=',')
