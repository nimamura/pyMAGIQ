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

#************************************************************************/
def neuralnet():
    # regularization parameters
    reg_val = 0.000001
    reg_ker = 0.00003
    activation = 'relu'

    initializers.Initializer()
    initializers.RandomNormal(seed=r_seed)

    # initialize model
    model = Sequential()

    # add layers
    # model.add(Dense( activation=activation, units=nunits,
    #                  kernel_regularizer=regularizers.l1(reg_ker),
    #                  input_dim=ndata*ncomp*nfreq*2))
    #
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits*3 ))
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits*2 ))
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.8) ))
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.5) ))
    #
    # model.add(Dense( activation=activation,
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.25) ))
    #
    #
    # model.add(Dense( activation='softmax', units=nrate,
    #                  kernel_regularizer=regularizers.l1(reg_ker)))

    # add layers
    reg_val = -6
    reg_ker = -5.5
    mid_units = 50
    num_layer = 7

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

    # initialize model
    # model = Sequential()

    # add layers
    # model.add(Dense( activation='relu', units=nunits,
    #                  kernel_regularizer=regularizers.l1(reg_ker),
    #                  input_dim=ndata*ncomp*nfreq*2))
    #
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits*3 ))
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits*2 ))
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.8) ))
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.5) ))
    #
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=int(nunits*0.25) ))
    #
    #
    # model.add(Dense( activation='softmax', units=nrate,
    #                  kernel_regularizer=regularizers.l1(reg_ker)))
    #
    # return model

#************************************************************************/
# parameter in neural network
nunits = 40

nrate = 5
nfreq = 30
ndata = 4
ncomp = 2
epochs = 1
batch_size = 32

# number of bagging
# rounds = 1
# rounds = 5

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

# datadir = '/Users/nimamura/Library/Mobile Documents/com~apple~CloudDocs/pyMAGIQ/pyMAGIQ/survey/USArray2019March'
datadir = '/home/server/pi/homes/nimamura/pyMAGIQ/survey/USArray2019March'

traindir    = datadir + '/train'
# X_trainpath = datadir + '/preprocessed/X_train.csv'
# y_trainpath = datadir + '/preprocessed/y_train.csv'
# rate_trainpath  = datadir + '/preprocessed/rate_train.csv'
# SiteIDpath  = datadir + '/preprocessed/siteID_train.csv'
# outputpath = datadir +'/outputs'

# # when using data augumentation
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

print('=================')
print(X_train.shape, y_train.shape, len(SiteID_train))
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nrate)

#************************************************************************/
# sort X and y in random
# This is necessary for the validation_split as validation_split splits only tail part
X_random, y_random, randindex = pyMAGIQ.utils.MAGIQlib.MAGIQlib.randomsortXY(X_train, y_train)

# read model
# model = neuralnet()
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
# pyMAGIQ.utils.MAGIQlib.MAGIQlib.plot_history(history,outputpath+'/acc_loss.eps',disp=False)
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

#************************************************************************/
# Human vs AI
# take validation model and rate
ndata_val = int(len(X_train)*0.1)
Xval = X_random[-ndata_val:-1,:]
yval = y_random[-ndata_val:-1]
index_val = randindex[-ndata_val:-1]

# read model
model = model_from_json(open(outputpath+'/model.json').read())

# read weights
model.load_weights(outputpath+'/weights.hdf5')

model.summary()

# # setting optimization
# model.compile(loss="categorical_crossentropy",
#               optimizer=Adamax(),
#               metrics=["accuracy"])
#evaluate unrated
yval_est  = model.predict(Xval)

# pyMAGIQ.utils.MAGIQlib.MAGIQlib.ratehistogram2d(yval_est,yval,5,fpath=outputpath+'/hist2d.eps')

#************************************************************************/
# Set unrated data info
unrateddir       = datadir + '/unrated'
X_unratedpath    = datadir + '/preprocessed/X_unrated.csv'
y_unratedpath    = datadir + '/preprocessed/y_unrated.csv'
rate_unratedpath = datadir + '/preprocessed/rate_unrated.csv'
SiteIDpath_unrated = datadir + '/preprocessed/siteID_unrated.csv'

unratedlists = sorted( [f for f in os.listdir(unrateddir) if not f.startswith('.')] )

nMTunrated  = len(unratedlists)

#************************************************************************/
# loading unrated data and initialize y for unrated
# read X_unrated
df=pd.read_csv(X_unratedpath, sep=',',header=None)
X_unrated = df.values

# load xml_unrated dict with pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open(outputpath+'/xml_unrated.p', 'rb') as fp:
    xml_unrated = pickle.load(fp)

#evaluate unrated
y_unrated  = model.predict(X_unrated)
y_unrated = np.argmax(y_unrated,axis=1)
np.savetxt( y_unratedpath, y_unrated, fmt='%d', delimiter=',')

y_unrated = y_unrated.reshape(len(y_unrated))

target = 'unrated'

xmin = -140
xmax = -65.0
ymin = 45.0
ymax = 65.0

latlist = []
lonlist = []

#************************************************************************/
# read rate
ratelist = pyMAGIQ.utils.iofiles.io.read_rate_ytrain(y_unratedpath)

# read site ID
siteIDlist = pyMAGIQ.utils.iofiles.io.read_siteID(SiteIDpath_unrated)

for i in range(len(xml_unrated)):
    latlist.append(xml_unrated[i]['lat'])
    lonlist.append(xml_unrated[i]['lon'])

fname = outputpath+'/imageCanada.eps'
pyMAGIQ.vis.plotUSmap.plotMap(datadir,xmin,xmax,ymin,ymax,latlist,lonlist,ratelist,fname)

#************************************************************************/
for i in range(1,6):
    print(i, ' number of index ',ratelist.count(i))
# ratelist.index(3)


#************************************************************************/
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#************************************************************************/
def plot_Z(alldata,targetSiteID,fpath):

    fig = plt.figure(figsize=(14,12))
    outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)

    for i in range(9):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        # target Site ID name
        target = targetSiteID[i]

        # read variable
        Zapp_xy     = alldata[target]['Zapp_xy']
        Zapp_err_xy = alldata[target]['Zapp_err_xy']
        Zphs_xy     = alldata[target]['Zphs_xy']
        Zphs_err_xy = alldata[target]['Zphs_err_xy']

        Zapp_yx     = alldata[target]['Zapp_yx']
        Zapp_err_yx = alldata[target]['Zapp_err_yx']
        Zphs_yx     = alldata[target]['Zphs_yx']
        Zphs_err_yx = alldata[target]['Zphs_err_yx']

        period      = alldata[target]['period']
        prate       = alldata[target]['prate']
        if 'grate' in alldata[target].keys():
            grate   = alldata[target]['grate']
            title_str = 'Pred Rate: ' + str(prate) + '  Given Rate: ' + str(grate)
        else:
            title_str = 'Pred Rate: ' + str(prate)


        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.grid()
            if j == 0:
                ax.set_title(title_str)
                ax.errorbar(period, Zapp_xy, yerr=Zapp_err_xy, fmt='o', mfc='none', capsize=4, label='xy')
                ax.errorbar(period, Zapp_yx, yerr=Zapp_err_yx, fmt='o', mfc='none', capsize=4, label='yx', c='r')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim([1.E-0,1.E+5])
                plt.setp(ax.get_xticklabels(), visible=False)
                if i ==0 or i==3 or i== 6:
                    ax.set_ylabel("App. Res.")

            else:
                ax.errorbar(period,Zphs_xy,yerr=Zphs_err_xy,fmt='o',mfc='none',capsize=4)
                ax.errorbar(period,Zphs_yx,yerr=Zphs_err_yx,fmt='o',mfc='none',capsize=4,c='r')
                ax.set_xscale('log')
                ax.set_xlim([1.E-0,1.E+5])
                ax.set_ylim([0,90])
                ax.set_yticks([0,30,60,90])
                if i ==0 or i==3 or i== 6:
                    ax.set_ylabel("Phase")

                if i >5:
                    ax.set_xlabel("Period (sec)")
                # ax.grid()

            fig.add_subplot(ax)

    plt.savefig(fpath,format='eps')
    # plt.show()



#************************************************************************/
# ID list to plot
# Change this list as you like
targetSiteID_unrated = ['MT_TF_Lithoprobe.SNO323.1999',
                'MT_TF_UofAlberta.sa175.2010',
                'MT_TF_Lithoprobe.WST139.1998',
                'MT_TF_Lithoprobe.ABT310.1993-1996',
                'MT_TF_Lithoprobe.ABT301.1993-1996',
                'MT_TF_Lithoprobe.ABT006.1993-1996',
                'MT_TF_UofAlberta.bc310.2009',
                'MT_TF_Lithoprobe.WST004.1998',
                'MT_TF_CAFE-MT.CAF11.2010' ]

target = 'unrated'

alldata = {}

# read rate
ratelist = pyMAGIQ.utils.iofiles.io.read_rate_ytrain(y_unratedpath)

# change variable name
targetSiteID = targetSiteID_unrated

siteIDlist = pyMAGIQ.utils.iofiles.io.read_siteID(SiteIDpath_unrated)

for i in range(len(targetSiteID)):
    print( targetSiteID[i] )
    # initialize dictionary for current MT site
    alldata[targetSiteID[i]] = {}
    siteIndex = siteIDlist.index( targetSiteID[i] )

    MTdir = datadir + '/' + target + '/' + targetSiteID[i]

    # filename of xml
    XMLlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('xml')] )

    # path to xml and edi file
    XMLpath = MTdir + '/' + XMLlist[0]

    period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err, _ = pyMAGIQ.utils.iofiles.io.readXML(XMLpath,True)

    prate = ratelist[siteIndex]

    Zapp2 = Zapp[:,1:2].flatten().tolist()
    Zapp_err2 = Zapp_err[:,1:2].flatten().tolist()
    Zphs2 = Zphs[:,1:2].flatten().tolist()
    Zphs_err2 = Zphs_err[:,1:2].flatten().tolist()

    Zapp3 = Zapp[:,2:3].flatten().tolist()
    Zapp_err3 = Zapp_err[:,2:3].flatten().tolist()
    Zphs3 = Zphs[:,2:3].flatten().tolist()
    Zphs_err3 = Zphs_err[:,2:3].flatten().tolist()

    alldata[targetSiteID[i]]['Zapp_xy']     = Zapp2
    alldata[targetSiteID[i]]['Zapp_err_xy'] = Zapp_err2
    alldata[targetSiteID[i]]['Zphs_xy']     = Zphs2
    alldata[targetSiteID[i]]['Zphs_err_xy'] = Zphs_err2

    alldata[targetSiteID[i]]['Zapp_yx']     = Zapp3
    alldata[targetSiteID[i]]['Zapp_err_yx'] = Zapp_err3
    alldata[targetSiteID[i]]['Zphs_yx']     = Zphs3
    alldata[targetSiteID[i]]['Zphs_err_yx'] = Zphs_err3

    alldata[targetSiteID[i]]['period']      = period
    alldata[targetSiteID[i]]['prate']       = prate

plot_Z(alldata,targetSiteID,outputpath+'/Appres.eps')


XMLvaldir = datadir + '/validation'
# list in training directory and test directory
vallists = sorted( [f for f in os.listdir(XMLvaldir) if not f.startswith('.')] )

# Make XML list from xml files
xml_val, SiteID_val = pyMAGIQ.utils.iofiles.io.get_XMLlists(nfreq,XMLvaldir,vallists,freqsets)

# read X (input data) and y (output data: rate) from XML files
y_val, rate_val, Zapplist, Zapp_errlist, Zphslist, Zphs_errlist = pyMAGIQ.utils.MAGIQlib.MAGIQlib.xml2Z(xml_val,SiteID_val,nfreq,Rotate=False,freq_interp=freqsets)

X_val = pyMAGIQ.utils.MAGIQlib.MAGIQlib.getXfromZ(Zapplist, Zapp_errlist, Zphslist, Zphs_errlist, nfreq, SiteID_val)

# reshape X_train to 2-D array
X_val = X_val.reshape(len(X_val),nfreq*ndata*ncomp*2).astype('float32')

# predict X_val
y_val_est = model.predict(X_val)

y_val_est = np.argmax( y_val_est,axis=1 )

targetSiteID = []
alldata = {}
for i in range(len(vallists)):
    SiteID = vallists[i]
    targetSiteID.append( SiteID )
    prate = y_val_est[i]+1
    grate = y_val[i]+1

    Zapp2 = Zapplist[i,:,1:2].flatten().tolist()
    Zapp_err2 = Zapp_errlist[i,:,1:2].flatten().tolist()
    Zphs2 = Zphslist[i,:,1:2].flatten().tolist()
    Zphs_err2 = Zphs_errlist[i,:,1:2].flatten().tolist()

    Zapp3 = Zapplist[i,:,2:3].flatten().tolist()
    Zapp_err3 = Zapp_errlist[i,:,2:3].flatten().tolist()
    Zphs3 = Zphslist[i,:,2:3].flatten().tolist()
    Zphs_err3 = Zphs_errlist[i,:,2:3].flatten().tolist()

    alldata[SiteID] = {}
    alldata[SiteID]['Zapp_xy']     = Zapp2
    alldata[SiteID]['Zapp_err_xy'] = Zapp_err2
    alldata[SiteID]['Zphs_xy']     = Zphs2
    alldata[SiteID]['Zphs_err_xy'] = Zphs_err2

    alldata[SiteID]['Zapp_yx']     = Zapp3
    alldata[SiteID]['Zapp_err_yx'] = Zapp_err3
    alldata[SiteID]['Zphs_yx']     = Zphs3
    alldata[SiteID]['Zphs_err_yx'] = Zphs_err3

    alldata[SiteID]['period']      = xml_val[i]['period']
    alldata[SiteID]['prate']       = prate
    alldata[SiteID]['grate']       = grate

    print(prate,grate)

plot_Z(alldata,targetSiteID,outputpath+'/AppRes_val.eps')


for i in range(len(index_val)):
    index = index_val[i]
#     print(i, np.argmax(yval[i])+1)
    grate = np.argmax( yval[i] ) + 1
    prate = np.argmax( yval_est[i] ) + 1
    print(i, 'grate', grate, 'prate',prate)
#************************************************************************/
XMLsensdir = datadir + '/sensitivity'
# list in training directory and test directory
senslists = sorted( [f for f in os.listdir(XMLsensdir) if not f.startswith('.')] )

# Make XML list from xml files
xml_sens, SiteID_sens = pyMAGIQ.utils.iofiles.io.get_XMLlists(nfreq,XMLsensdir,senslists,freqsets)

# read X (input data) and y (output data: rate) from XML files
y_sens, rate_sens, Zapplist, Zapp_errlist, Zphslist, Zphs_errlist = pyMAGIQ.utils.MAGIQlib.MAGIQlib.xml2Z(xml_sens,SiteID_sens,nfreq,Rotate=True,freq_interp=freqsets)

X_sens = pyMAGIQ.utils.MAGIQlib.MAGIQlib.getXfromZ(Zapplist, Zapp_errlist, Zphslist, Zphs_errlist, nfreq, SiteID_sens)

# reshape X_train to 2-D array
X_sens = X_sens.reshape(len(X_sens),nfreq*ndata*ncomp*2).astype('float32')

# predict X_val
y_sens_est = model.predict(X_sens)
y_sens_est = np.argmax( y_sens_est,axis=1 )

X_ori = X_sens
# model changing rate (5%)
delta = 0.02

# initialize dfdx
dfdx = np.zeros(np.shape(X_sens),dtype=np.float32)
rel_score = np.zeros(np.shape(X_sens),dtype=np.float32)

# read model
model = model_from_json(open(outputpath+'/model.json').read())
# read weights
model.load_weights(outputpath+'/weights.hdf5')

for n in range(len(senslists)):
#     print('rate[n]',rate[n])
    y_current = y_sens[n]
    print('Rating ', y_current+1)
    for i in range(np.shape(X_sens)[1]):

        #################################
        # forward difference
        X_diff = X_sens[n,].reshape(1,nfreq*ndata*ncomp*2)
        X_diff[0,i] = X_diff[0,i]*(1.0+delta)

        # Apply X_sens to model
        #evaluate unrated
        yval_fwd  = model.predict(X_diff)
#         print('yval_fwd',yval_fwd)

        #################################
        # backward difference
        X_diff = X_sens[n,].reshape(1,nfreq*ndata*ncomp*2)
        X_diff[0,i] = X_diff[0,i]*(1.0-delta)

        # Apply X_sens to model
        #evaluate unrated
        yval_bwd  = model.predict(X_diff)
#         print('yval_bwd',yval_bwd)

#         print('fwd-bwd',yval_fwd[0,Rating] ,yval_bwd[0,Rating] )
        dfdx[n,i] = (yval_fwd[0,y_current] - yval_bwd[0,y_current]) / (X_diff[0,i]*delta*2.0)
        rel_score[n,i] = dfdx[n,i] **2.0

rel_score_reshape = rel_score.reshape(len(senslists),nfreq,ndata*ncomp*2).astype('float32')

#
plt.figure(figsize=(18,18))

for n in range(len(senslists)):
    period = xml_sens[n]['period']

# for n in range(1):
    score = rel_score_reshape[n,:,:]
    for i in range(ndata*2):
        if i%2 == 0:
            comp = '(xy)'
        else:
            comp = '(yx)'

        if i == 0:
            title = 'Apparent resistivity'
        elif i == 1:
            title = 'Phase'
        elif i == 2:
            title = 'Error bar of apparent resistivity'
        elif i == 3:
            title = 'Error bar of phase'
        elif i == 4:
            title = 'Continuity of apparent resistivity'
        elif i == 5:
            title = 'Continuity of phase'
        elif i == 6:
            title = 'Continuity of error bar of apparent resitivity'
        elif i == 7:
            title = 'Continuity of error bar of phase'


        plt.subplot(len(senslists)*2,4,i+1+n*8)

        plt.semilogx(period,score[:,0+2*i],label='xy')
        plt.semilogx(period,score[:,1+2*i],label='yx',c='r')
        plt.tight_layout()
        plt.title(title)
        plt.xlim([1.E-0,1.E+5])

#         plt.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
#         plt.semilogx(period,rel_score_reshape[0,:,0+2*i],label='xy')
#         plt.semilogx(period,rel_score_reshape[0,:,1+2*i],label='yx')

        if n == 0 and i<4:
            plt.legend()


        if n == len(senslists)-1 and i>3:
            plt.xlabel('Period (sec)')

plt.savefig(outputpath+'/sens_analy.eps',format='eps')
plt.show()
