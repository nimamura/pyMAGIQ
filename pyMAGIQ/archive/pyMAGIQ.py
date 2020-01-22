#!/usr/bin/python
# -*- coding: utf-8 -*-

#*************************************************************************
 #  File Name: TSpred.py
 #
 #  Created By: Naoto Imamura (nimamura)
 #
 #  Purpose: This script finds file including NaN value in it
 #*************************************************************************

#************************************************************************/
 #imports
#************************************************************************/
import csv
import sys
import os.path
import json
import numpy as np
import pandas as pd
from scipy import signal

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten, Conv1D
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam, Nadam,Adamax
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import model_from_json
from keras.utils import plot_model
from keras.utils import np_utils

#************************************************************************/
 #global variables
#************************************************************************/
# period for basic USArray
freq_interp = [  1.367187E-01, 1.093750E-01, 8.593753E-02, 6.640627E-02, 5.078124E-02, 3.906250E-02,
        3.027344E-02, 2.343750E-02, 1.855469E-02, 1.464844E-02, 1.171875E-02, 9.765625E-03,
        7.568361E-03, 5.859374E-03, 4.638671E-03, 3.662109E-03, 2.929688E-03, 2.441406E-03,
        1.892090E-03, 1.464844E-03, 1.159668E-03, 9.155271E-04, 7.324221E-04, 6.103516E-04,
        4.425049E-04, 3.204346E-04, 2.136230E-04, 1.373291E-04, 8.392331E-05, 5.340577E-05]

#************************************************************************/
 #                            main()
 #Main function in python program
 #Inputs:
 #Outputs:
#************************************************************************/
def main(argv):

    # nunits = 60
    nunits = 10
    # nunits = 10
    # nunits = 100
    nrate = 5
    nfreq = 30
    ndata = 8
    epochs = 500
    batch_size = 32

    ############################
    # read xml data
    # datadir = '/Users/nimamura/GIC/survey/ZML/data'
    datadir = '/home/server/pi/homes/nimamura/GIC/survey/ZML/data'
    # traindir    = datadir + '/unreadable'
    traindir    = datadir + '/train'
    testdir     = datadir + '/test'
    X_trainpath = datadir + '/X_train.csv'
    X_testpath  = datadir + '/X_test.csv'
    y_trainpath = datadir + '/y_train.csv'
    y_testpath  = datadir + '/y_test.csv'
    rate_trainpath  = datadir + '/rate_train.csv'
    rate_testpath  = datadir + '/rate_test.csv'
    SiteIDpath  = datadir + '/siteID_train.csv'

    trainlists = sorted( [f for f in os.listdir(traindir) if not f.startswith('.')] )
    testlists  = sorted( [f for f in os.listdir(testdir) if not f.startswith('.')] )

    nMTtrain = len(trainlists)
    nMTtest  = len(testlists)

    # read saved X and y data if it exists
    if not os.path.isfile(X_trainpath):
        # read X_train and y_train
        X_train = np.zeros((nMTtrain,nfreq,ndata),dtype=np.float32)    # Note that real value
        y_train = np.zeros((nMTtrain,),dtype=np.int)

        X_train, y_train, rate_train, SiteID = pyMAGIQ.gicpy.MAGIQlib.MAGIQlib.getXYdataset(X_train,y_train,nfreq,traindir,trainlists,freq_interp)

        # read X_test and y_test
        X_test = np.zeros((nMTtest,nfreq,ndata),dtype=np.float32)    # Note that real value
        y_test = np.zeros((nMTtest,),dtype=np.int)

        X_test, y_test, rate_test, SiteID_test = pyMAGIQ.gicpy.MAGIQlib.MAGIQlib.getXYdataset(X_test,y_test,nfreq,testdir,testlists,freq_interp)

        # convert shape
        X_train = X_train.reshape(len(X_train),nfreq*ndata).astype('float32')
        X_test  = X_test.reshape(len(X_test),nfreq*ndata).astype('float32')

        # save X and y data in csv format. You don't have to read EDI files again
        # np.savetxt( X_trainpath, X_train, fmt="%.5e", delimiter=',')
        # np.savetxt( X_testpath,  X_test, fmt="%.5e", delimiter=',')
        np.savetxt( X_trainpath, X_train, delimiter=',')
        np.savetxt( X_testpath,  X_test, delimiter=',')
        np.savetxt( y_trainpath, y_train, fmt='%d', delimiter=',')
        np.savetxt( y_testpath,  y_test, fmt='%d', delimiter=',')
        np.savetxt( rate_trainpath, rate_train, fmt='%d', delimiter=',')
        np.savetxt( rate_testpath,  rate_test, fmt='%d', delimiter=',')

        with open(SiteIDpath,'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in SiteID:
                writer.writerow([val])
    else:
        # read saved X and y data if it exists
        X_train = np.loadtxt( X_trainpath , dtype=np.float32, delimiter=",")
        X_test  = np.loadtxt( X_testpath  , dtype=np.float32, delimiter=",")
        y_train = np.loadtxt( y_trainpath , dtype=np.int, delimiter=",")
        y_test  = np.loadtxt( y_testpath  , dtype=np.int, delimiter=",")
        rate_train = np.loadtxt( rate_trainpath , dtype=np.int, delimiter=",")
        rate_test  = np.loadtxt( rate_testpath  , dtype=np.int, delimiter=",")

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nrate)
    y_test  = np_utils.to_categorical(y_test, nrate)

    ############################
    # create model

    # model = model_from_json(open('model.json').read())

    # read weights
    # model.load_weights('weights.hdf5')

    # initialize model
    model = Sequential()
    #
    # add layers
    reg_val = 0.0001
    reg_ker = 0.001
    model.add(Dense( activation='relu', units=nunits,
                     kernel_regularizer=regularizers.l1(reg_ker),
                     # kernel_regularizer=regularizers.l1(0.006),
                     # activity_regularizer=regularizers.l1(0.001),
                     input_dim=ndata*nfreq))

    model.add(Dense( activation='relu',
                     kernel_regularizer=regularizers.l1(reg_val),
                     units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))
    # model.add(Dense( activation='relu',
    #                  kernel_regularizer=regularizers.l1(reg_val),
    #                  units=nunits ))

    model.add(Dense( activation='softmax', units=nrate,
                     kernel_regularizer=regularizers.l1(reg_ker),
                     # kernel_regularizer=regularizers.l1(0.005),
                     # activity_regularizer=regularizers.l1(0.001),
                     name='preds' ))

    # add layers
    # model.add(Dense( activation='sigmoid', units=nunits,
    #                  kernel_regularizer=regularizers.l1(0.003),
    #                  activity_regularizer=regularizers.l1(0.003), input_dim=ndata*nfreq))
    # model.add(Dense( activation='tanh', units=nunits,
    #                  activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dense( activation='relu', units=nunits,
    #                  activity_regularizer=regularizers.l1(0.005)))
    # model.add(Dense( activation='relu', units=nunits,
    #                  activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dense( activation='relu', units=nunits))
    # # model.add(Dense( activation='relu', units=nunits))
    # # model.add(Dense( activation='relu', units=nunits ))
    # # model.add(Dense( activation='relu', units=nunits ))
    # model.add(Dense( activation="softmax", units=nrate))

    # Used in presentaion, but ignored nfreq is not 30
    # model.add(Dense( activation='sigmoid', units=nunits,
    #                  kernel_regularizer=regularizers.l1(0.002),
    #                  activity_regularizer=regularizers.l1(0.002), input_dim=ndata*nfreq))
    #
    # model.add(Dense( activation='tanh', units=nunits,
    #                  activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dense( activation='relu', units=nunits,
    #                  activity_regularizer=regularizers.l1(0.001)))
    # # model.add(Dense( activation='relu', units=nunits,
    # #                  activity_regularizer=regularizers.l1(0.001)))
    # model.add(Dense( activation='relu', units=nunits))
    # # model.add(Dense( activation='relu', units=nunits))
    # # model.add(Dense( activation='relu', units=nunits ))
    # # model.add(Dense( activation='relu', units=nunits ))
    # model.add(Dense( activation="softmax", units=nrate))

    # setting optimization
    model.compile(loss="categorical_crossentropy",
                  # optimizer=Nadam(),
                  optimizer=Adamax(),
                  # optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                  # optimizer=Adam(),
                  metrics=["accuracy"])

    model.summary()
    # learning parameters
    history = model.fit(X_train, y_train, epochs=epochs,
                         validation_split=0.0, batch_size=batch_size)

    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    #evaluate test
    # y_est  = model.predict(X_test)

    # print y_test
    # print y_est

    # #######################
    # # save model
    model_json = model.to_json()
    with open("model.json",mode='w') as f:
        f.write(model_json)

    # save weight
    model.save_weights('weights.hdf5')

    # plot history of convergence
    magiq.plot_history(history)

    # rate histogram
    # magiq.ratehistogram2d(y_est,y_test,nrate,fname='hist2d.png')
    # magiq.ratehistogram1d(rate_train,fname='hist.png')

#************************************************************************/
 #                            __name__
 #Needed to define main function in python program
#************************************************************************/
if __name__ == "__main__":
    main(sys.argv[1:])
