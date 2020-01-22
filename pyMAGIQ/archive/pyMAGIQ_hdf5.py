#!/usr/bin/python
# -*- coding: utf-8 -*-

#*************************************************************************
 #  File Name: pyMAGIQ_hdf5.py
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
import h5py

#************************************************************************/
 #global variables
#************************************************************************/
# period for basic USArray
freq_interp = [  1.367187E-01, 1.093750E-01, 8.593753E-02, 6.640627E-02, 5.078124E-02, 3.906250E-02,
        3.027344E-02, 2.343750E-02, 1.855469E-02, 1.464844E-02, 1.171875E-02, 9.765625E-03,
        7.568361E-03, 5.859374E-03, 4.638671E-03, 3.662109E-03, 2.929688E-03, 2.441406E-03,
        1.892090E-03, 1.464844E-03, 1.159668E-03, 9.155271E-04, 7.324221E-04, 6.103516E-04,
        4.425049E-04, 3.204346E-04, 2.136230E-04, 1.373291E-04, 8.392331E-05, 5.340577E-05]

period = [ 1.0/x for x in freq_interp ]

nfreq = 30
# xy and yx components for Zapp, Zphs and those error bars
ndata = 4
ncomp = 2

#************************************************************************/
 #                            main()
 #Main function in python program
 #Inputs:
 #Outputs:
#************************************************************************/
def main(argv):

    filename = 'weights.hdf5'
    f = h5py.File(filename, 'r')
    # List all groups
    dense1 = f['dense_1']['dense_1']
    bias   = dense1['bias:0']
    kernel = dense1['kernel:0']
    # convert h5py dataset to ndarray
    dset = kernel[()]

    nnode = len(dset[0])
    print(nnode)

    Wmatrix = np.zeros( (ncomp*ndata,nfreq,nnode) ,dtype=np.float32)
    for inode in range(nnode):
        tmp = dset[:,inode]
        Wmatrix[:,:,inode] = tmp.reshape(ncomp*ndata,nfreq,order='F')

    # plot_mesh_dset(Wmatrix)
    plot_line_dset(Wmatrix)


#************************************************************************/
def plot_mesh_dset(Wmatrix):

    nnode = Wmatrix.shape[2]

    yaxis = range(ndata*ncomp+1)

    plt.figure(figsize=(20,14))

    for inode in range(nnode):
        W = np.log10(np.abs(Wmatrix[:,:,inode]))
        plt.subplot(10,6,inode+1)

        plt.pcolormesh(period,yaxis,W)
        plt.xscale('log')
        plt.xlabel('Period (sec)')
        plt.clim(-5,-2)
        # plt.clim(0.0,5.E-3)

    plt.show()

#************************************************************************/
def plot_line_dset(Wmatrix):

    nnode = Wmatrix.shape[2]

    plt.figure(figsize=(18,12))

    for i in range(8):
        if i%2 == 0:
            comp = '(xy)'
        else:
            comp = '(yx)'

        if i//2 == 0:
            data = 'Apparent resistivity'
        elif i//2 == 1:
            data = 'Phase'
        elif i//2 == 2:
            data = 'Error bar of apparent resistivity'
        elif i//2 == 3:
            data = 'Error bar of phase'

        plt.subplot(2,4,i+1)
        string = data + ' ' + comp
        plt.title(string)
        for inode in range(nnode):
            W = np.log10( np.abs( Wmatrix[i,:,inode] ) )
            plt.semilogx(period,W,'k-o')
            plt.xlabel('Period (sec)')

    plt.savefig('./cnn_group.eps',format='eps')
    plt.show()


#************************************************************************/
 #                            __name__
 #Needed to define main function in python program
#************************************************************************/
if __name__ == "__main__":
    main(sys.argv[1:])
