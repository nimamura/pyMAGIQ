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
import sys
import os.path
import csv
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from utils.utils.iofiles import io as utils
from utils.utils.MAGIQlib import MAGIQlib as magiq

import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam, Adamax
from keras import regularizers
from keras.models import model_from_json

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

    nfreq = 30
    ndata = 8

    ############################
    # read xml data
    # datadir = '/Users/nimamura/GIC/survey/ZML/data'
    datadir          = '/home/server/pi/homes/nimamura/GIC/survey/ZML/data'
    traindir         = datadir + '/train'
    unrateddir       = datadir + '/unrated'
    X_trainpath      = datadir + '/X_train.csv'
    X_unratedpath    = datadir + '/X_unrated.csv'
    y_trainpath      = datadir + '/y_train.csv'
    y_unratedpath    = datadir + '/y_unrated.csv'
    rate_trainpath   = datadir + '/rate_train.csv'
    rate_unratedpath = datadir + '/rate_unrated.csv'
    SiteIDpath       = datadir + '/siteID_unrated.csv'

    trainlists   = sorted( [f for f in os.listdir(traindir) if not f.startswith('.')] )
    unratedlists = sorted( [f for f in os.listdir(unrateddir) if not f.startswith('.')] )

    nMTtrain = len(trainlists)
    nMTunrated  = len(unratedlists)

    # read X_unrated
    X_unrated = np.zeros((nMTunrated,nfreq,ndata),dtype=np.float32)    # Note that real value
    X_unrated,SiteID = magiq.getX_unrated(X_unrated,nfreq,unrateddir,unratedlists,freq_interp)

    # convert shape
    X_unrated = X_unrated.reshape(len(X_unrated),nfreq*ndata).astype('float32')

    # save X and y data in csv format. You don't have to read EDI files again
    np.savetxt( X_unratedpath,  X_unrated, delimiter=',')
    with open(SiteIDpath,'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in SiteID:
            writer.writerow([val])

    # ############################
    # read model
    model = model_from_json(open('model.json').read())

    # read weights
    model.load_weights('weights.hdf5')

    model.summary()

    # setting optimization
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adamax(),
                  metrics=["accuracy"])

    #evaluate unrated
    y_est  = model.predict(X_unrated)

    np.savetxt( y_unratedpath, y_est, delimiter=',' )

#************************************************************************/
 #                            __name__
 #Needed to define main function in python program
#************************************************************************/
if __name__ == "__main__":
    main(sys.argv[1:])
