r_seed = 122 # 121 and 122 are good examples

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# Local Import
import pyMAGIQ

# load xml_unrated dict with pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

#************************************************************************/
# parameter in neural network
nunits = 40
nrate = 5
nfreq = 30
ndata = 4
ncomp = 2

#************************************************************************/
# setting directories
# frequency sets for USArray (Currently, this code works only with these frequencies)
freqsets = [  1.367187E-01, 1.093750E-01, 8.593753E-02, 6.640627E-02, 5.078124E-02, 3.906250E-02,
        3.027344E-02, 2.343750E-02, 1.855469E-02, 1.464844E-02, 1.171875E-02, 9.765625E-03,
        7.568361E-03, 5.859374E-03, 4.638671E-03, 3.662109E-03, 2.929688E-03, 2.441406E-03,
        1.892090E-03, 1.464844E-03, 1.159668E-03, 9.155271E-04, 7.324221E-04, 6.103516E-04,
        4.425049E-04, 3.204346E-04, 2.136230E-04, 1.373291E-04, 8.392331E-05, 5.340577E-05]

# # number of frequency used in data
nfreq = len(freqsets)

# datadir = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'
datadir = '/Users/nimamura/work/pyMAGIQ/survey/USArray2020Jan_test'
outputpath = datadir +'/outputs'

# read model and weights
model = model_from_json(open(outputpath+'/model.json').read())
model.load_weights(outputpath+'/weights.hdf5')

model.summary()

#************************************************************************/
# Set unrated data info
unrateddir = datadir + '/unrated'
X_unratedpath = datadir + '/preprocessed/X_unrated.csv'
y_unratedpath = datadir + '/preprocessed/y_unrated.csv'
rate_unratedpath = datadir + '/preprocessed/rate_unrated.csv'
SiteIDpath_unrated = datadir + '/preprocessed/siteID_unrated.csv'

unratedlists = sorted([f for f in os.listdir(unrateddir) if not f.startswith('.')])

nMTunrated = len(unratedlists)

#************************************************************************/
# loading unrated data and initialize y for unrated
# read X_unrated
df=pd.read_csv(X_unratedpath, sep=',',header=None)
X_unrated = df.values

with open(outputpath+'/xml_unrated.p', 'rb') as fp:
    xml_unrated = pickle.load(fp)

#evaluate unrated
y_unrated  = model.predict(X_unrated)
y_unrated = np.argmax(y_unrated,axis=1)+1
np.savetxt( y_unratedpath, y_unrated, fmt='%d', delimiter=',')

# y_unrated = y_unrated.reshape(len(y_unrated))

# target = 'unrated'
#
# xmin = -140
# xmax = -65.0
# ymin = 45.0
# ymax = 65.0
#
# latlist = []
# lonlist = []
#
# #************************************************************************/
# # read rate
# ratelist = pyMAGIQ.utils.iofiles.io.read_rate_ytrain(y_unratedpath)
#
# # read site ID
# siteIDlist = pyMAGIQ.utils.iofiles.io.read_siteID(SiteIDpath_unrated)
#
# for i in range(len(xml_unrated)):
#     latlist.append(xml_unrated[i]['lat'])
#     lonlist.append(xml_unrated[i]['lon'])
#
# #************************************************************************/
# # ID list to plot
# # Change this list as you like
# targetSiteID_unrated = ['MT_TF_Lithoprobe.SNO323.1999',
#                 'MT_TF_UofAlberta.sa175.2010',
#                 'MT_TF_Lithoprobe.WST139.1998',
#                 'MT_TF_Lithoprobe.ABT310.1993-1996',
#                 'MT_TF_Lithoprobe.ABT301.1993-1996',
#                 'MT_TF_Lithoprobe.ABT006.1993-1996',
#                 'MT_TF_UofAlberta.bc310.2009',
#                 'MT_TF_Lithoprobe.WST004.1998',
#                 'MT_TF_CAFE-MT.CAF11.2010' ]
#
# target = 'unrated'
#
# alldata = {}
#
# # read rate
# ratelist = pyMAGIQ.utils.iofiles.io.read_rate_ytrain(y_unratedpath)
#
# # change variable name
# targetSiteID = targetSiteID_unrated
#
# siteIDlist = pyMAGIQ.utils.iofiles.io.read_siteID(SiteIDpath_unrated)
#
# for i in range(len(targetSiteID)):
#     print( targetSiteID[i] )
#     # initialize dictionary for current MT site
#     alldata[targetSiteID[i]] = {}
#     siteIndex = siteIDlist.index( targetSiteID[i] )
#
#     MTdir = datadir + '/' + target + '/' + targetSiteID[i]
#
#     # filename of xml
#     XMLlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('xml')] )
#
#     # path to xml and edi file
#     XMLpath = MTdir + '/' + XMLlist[0]
#
#     period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err, _ = pyMAGIQ.utils.iofiles.io.readXML(XMLpath,True)
#
#     prate = ratelist[siteIndex]
#
#     Zapp2 = Zapp[:,1:2].flatten().tolist()
#     Zapp_err2 = Zapp_err[:,1:2].flatten().tolist()
#     Zphs2 = Zphs[:,1:2].flatten().tolist()
#     Zphs_err2 = Zphs_err[:,1:2].flatten().tolist()
#
#     Zapp3 = Zapp[:,2:3].flatten().tolist()
#     Zapp_err3 = Zapp_err[:,2:3].flatten().tolist()
#     Zphs3 = Zphs[:,2:3].flatten().tolist()
#     Zphs_err3 = Zphs_err[:,2:3].flatten().tolist()
#
#     alldata[targetSiteID[i]]['Zapp_xy']     = Zapp2
#     alldata[targetSiteID[i]]['Zapp_err_xy'] = Zapp_err2
#     alldata[targetSiteID[i]]['Zphs_xy']     = Zphs2
#     alldata[targetSiteID[i]]['Zphs_err_xy'] = Zphs_err2
#
#     alldata[targetSiteID[i]]['Zapp_yx']     = Zapp3
#     alldata[targetSiteID[i]]['Zapp_err_yx'] = Zapp_err3
#     alldata[targetSiteID[i]]['Zphs_yx']     = Zphs3
#     alldata[targetSiteID[i]]['Zphs_err_yx'] = Zphs_err3
#
#     alldata[targetSiteID[i]]['period']      = period
#     alldata[targetSiteID[i]]['prate']       = prate
#
# pyMAGIQ.vis.plotZ.plotZ(alldata,targetSiteID,outputpath+'/Appres.eps')
