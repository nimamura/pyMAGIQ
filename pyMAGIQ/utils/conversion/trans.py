#!/usr/bin/env python

"""
=============
trans module
=============

Functions
----------
    - bandflist
    - interpolateZ
    - getAppRes

NI, 2018

"""

#=================================================================
 #imports
#=================================================================
import sys
import numpy as np
from scipy import signal
import cmath

#=================================================================
# def bandflist(f,mband):
#
#     # length of tband is mband-1
#     if mband == 30:
#         tband = [8.22858,10.38961,13.34759,17.37557,22.64616, \
#             29.31613,37.84947,48.28071,61.08071,76.80001,93.86667, \
#             117.26452,151.39785,193.12281,244.32282,307.20001, \
#             375.46668,469.05806,605.59140,772.49125,977.29126, \
#             1228.80005,1501.86670,1949.13104,2690.31201, \
#             3900.95252,5981.46045,9598.70728,15320.10450]
#     elif mband == 18:
#         tband = [ 151.39785,193.12281,244.32282,307.20001, \
#             375.46668,469.05806,605.59140,772.49125,977.29126, \
#             1228.80005,1501.86670,1949.13104,2690.31201, \
#             3900.95252,5981.46045,9598.70728,15320.10450]
#     else:
#         print('Given mband is not right number')
#         print('Please check a function named bandflist in trans.py')
#         sys.exit()
#
#     bandf = np.zeros((len(f),1),dtype=int)
#
#     # index 0 is DC component, so use highest tband
#     # smaller index number is for lower frequency
#     bandf[0,0] = mband-1
#     for i in range(1,len(f)):
#         period = 1.0/f[i]
#         icount = 0
#         for j in tband:
#             # count the number of tband smaller than given period.
#             # That will give the band index
#             if period > j:
#                 icount += 1
#
#         bandf[i,0] = icount
#
#     return bandf

#=================================================================
#
def interpolateZ(ZEDI, periodZEDI):

    inter_period = np.array( [7.31429,  9.14286, 11.63636, 15.05882, 19.69231, \
                         25.60000, 33.03226, 42.66667, 53.89474, 68.26667, \
                         85.33334, 102.40000, 132.12903, 170.66667, \
                         215.57895, 273.06668, 341.33334, 409.60001, \
                         528.51611, 682.66669, 862.31580, 1092.26672, \
                         1365.33337, 1638.40002, 2259.86206, 3120.76196, \
                         4681.14307, 7281.77783, 11915.63672, 18724.57227] )

    nperiod = len(inter_period)
    Zinterp = np.zeros((nperiod,4),dtype=np.complex)

    # make interpolation to calculate impedance value for the specific
    # frequency value
    Zinterp[...,0] = np.interp(inter_period,periodZEDI,ZEDI[...,0])
    Zinterp[...,1] = np.interp(inter_period,periodZEDI,ZEDI[...,1])
    Zinterp[...,2] = np.interp(inter_period,periodZEDI,ZEDI[...,2])
    Zinterp[...,3] = np.interp(inter_period,periodZEDI,ZEDI[...,3])

    return Zinterp

#=================================================================
def getAppRes(xmldata):

    Z      = xmldata['Z']
    ZVAR   = xmldata['ZVAR']
    period = xmldata['period']
    nfreq  = xmldata['nfreq']

    Zapp     = np.zeros(Z.shape, dtype=float)
    Zapp_err = np.zeros(Z.shape, dtype=float)
    Zphs     = np.zeros(Z.shape, dtype=float)
    Zphs_err = np.zeros(Z.shape, dtype=float)

    for i in range(nfreq):
        period1 = period[i]

        for j in range(4):
            Zapp[i,j]     = np.abs(Z[i,j])**2.0 *period1 / 5.0
            Zapp_err[i,j] = np.sqrt( 2*period1*Zapp[i,j]*ZVAR[i,j]/5.0  )
            Zphs[i,j]     = np.angle(Z[i,j], deg=True)
            if Zphs[i,j] < 0:
                Zphs[i,j] = 180.0 + Zphs[i,j]
            Zphs_err[i,j] = 180.0 /(np.pi * np.abs(Z[i,j])) *np.sqrt( ZVAR[i,j]/2.0 )
            # print( i,j,2*period1*Zapp[i,j]*ZVAR[i,j]/5.0,Zapp[i,j],ZVAR[i,j])


    return Zapp, Zapp_err, Zphs, Zphs_err
