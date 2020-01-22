#!/usr/bin/env python

"""
=============
io module
=============

Functions
----------
    - getXYdataset
    - checklists
    - getRating
    - plot_history
    - ratehistogram1d
    - ratehistogram2d
    - randomsortXY
    - rotateXML
    - xml2Z
    - getXfromZ
    - dataaug

NI, 2018

"""
#=================================================================
 #imports
#=================================================================
import sys
import os.path
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pyMAGIQ

#************************************************************************/
def getXYdataset(X,y,nfreq,fdir,lists,freq_interp):

    icount = 0
    rate=np.zeros((len(lists,)),dtype=int)
    train_list = []
    xml_dict = {}
    # loop for MT training list
    for MTdirname in lists:
        print( MTdirname )

        MTdir = fdir + '/' + MTdirname
        # read xml and edi file in the folder
        XMLlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('xml')] )
        # EDIlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('edi')] )
        # ZRRlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('zrr')] )

        # check there is only one xml and edi file in the directory
        checklists(XMLlist=XMLlist)

        # path to xml and edi file
        XMLpath = MTdir + '/' + XMLlist[0]
        # EDIpath = MTdir + '/' + EDIlist[0]
        # ZRRpath = MTdir + '/' + ZRRlist[0]

        # get rate in xml file
        Rating = int( getRating(XMLpath) )

        # get information in EDI file
        #
        # xmldata, Zapp, Zapp_err, Zphs, Zphs_err = utils.readXML(XMLpath,True)
        period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err, period_ori = pyMAGIQ.utils.iofiles.io.readXML(XMLpath,True,freq_interp)
        # period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err = utils.readXML(XMLpath,True,freq_interp)

        # Z = utils.readXML(XMLpath)
        # periodEDI, Z, ZERR, Zapp, Zapp_err, Zphs, Zphs_err = gicio_edi.readEDIall(EDIpath)
        # periodEDI, Z, ZERR, Zapp, Zapp_err, Zphs, Zphs_err = gicio_edi.readEDIall(EDIpath, freq_interp)
        nperiod = len(Zapp)

        # if maximum value periodEDI is smaller than 1000 sec, skip it because interpolation won't work well
        if max(period_ori) < 1000.0:
            print( 'skip it! Maximum period is smaller than 1000 sec', MTdirname )
            continue

        # if number of frequency is differnet from nfreq, then skip to read it
        # if nperiod is not nfreq:
        #     print 'skip this', MTdirname
        #     continue

        # make X_train and y_train
        # if Rating > 3:
        #     y[icount,1] = 1
        # else:
        #     y[icount,0] = 1
             #y[icount,Rating-1] = 1
        y[icount] = Rating-1
        # y[icount,Rating-1] = 1
        rate[icount] = Rating

        for ifreq in range(nfreq):
            # if Z[ifreq,0].real > 1.E+06:
            #     print MTdirname, 'over 1.E+07'

            # Normalize dataset and input it
            X[icount, ifreq,  0] = (np.log10( Zapp[ifreq,1] )+4)*0.01
            X[icount, ifreq,  1] = (np.log10( Zapp[ifreq,2] )+4)*0.01
            # X[icount, ifreq,  2] = (np.log10( Zapp_err[ifreq,1] )+4)*0.01
            # X[icount, ifreq,  3] = (np.log10( Zapp_err[ifreq,2] )+4)*0.01
            X[icount, ifreq,  2] = (np.log10( Zapp_err[ifreq,1] )+6)*0.01
            X[icount, ifreq,  3] = (np.log10( Zapp_err[ifreq,2] )+6)*0.01
            # Added 180.0 because phase could be goes under zero.
            # Also, we divided phase by 540 because it can go over 180.
            X[icount, ifreq,  4] = (Zphs[ifreq,1]+180.0) / 540.0
            X[icount, ifreq,  5] = (Zphs[ifreq,2]+180.0) / 540.0
            X[icount, ifreq,  6] = Zphs_err[ifreq,1] / 540.0
            X[icount, ifreq,  7] = Zphs_err[ifreq,2] / 540.0

            if ifreq < nfreq-1:
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq,1]) - np.log10(Zapp[ifreq+1,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq,2]) - np.log10(Zapp[ifreq+1,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq,1])-np.log10(Zapp_err[ifreq+1,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq,2])-np.log10(Zapp_err[ifreq+1,2])) ) * 0.01
                # X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq,1]) - np.log10(Zapp[ifreq+1,1]))  ) * 0.1
                # X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq,2]) - np.log10(Zapp[ifreq+1,2]))  ) * 0.1
                # X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq,1])-np.log10(Zapp_err[ifreq+1,1])) ) * 0.1
                # X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq,2])-np.log10(Zapp_err[ifreq+1,2])) ) * 0.1

                X[icount, ifreq, 12] = np.abs((Zphs[ifreq,1]-Zphs[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq,2]-Zphs[ifreq+1,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq,1]-Zphs_err[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq,2]-Zphs_err[ifreq+1,2]) / 540.0 )

            else:
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq-1,1]) - np.log10(Zapp[ifreq,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq-1,2]) - np.log10(Zapp[ifreq,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq-1,1])-np.log10(Zapp_err[ifreq,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq-1,2])-np.log10(Zapp_err[ifreq,2])) ) * 0.01
                # X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq-1,1]) - np.log10(Zapp[ifreq,1]))  ) * 0.1
                # X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq-1,2]) - np.log10(Zapp[ifreq,2]))  ) * 0.1
                # X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq-1,1])-np.log10(Zapp_err[ifreq,1])) ) * 0.1
                # X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq-1,2])-np.log10(Zapp_err[ifreq,2])) ) * 0.1
                X[icount, ifreq, 12] = np.abs((Zphs[ifreq-1,1]-Zphs[ifreq,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq-1,2]-Zphs[ifreq,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq-1,1]-Zphs_err[ifreq,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq-1,2]-Zphs_err[ifreq,2]) / 540.0 )

            # print( 'Zapp   [%12.5e %12.5e %12.5e %12.5e] '%(X[icount, ifreq, 0],X[icount, ifreq, 1],X[icount, ifreq, 2],X[icount, ifreq, 3]) )
            # print( 'Zphs   [%12.5e %12.5e %12.5e %12.5e] '%(X[icount, ifreq, 8],X[icount, ifreq, 9],X[icount, ifreq, 10],X[icount, ifreq, 11]) )
            # print( 'Zapp_err [%12.5e %12.5e %12.5e %12.5e] '%(X[icount, ifreq, 4],X[icount, ifreq, 5],X[icount, ifreq, 6],X[icount, ifreq, 7] ) )
            # print( 'Zphs_err [%12.5e %12.5e %12.5e %12.5e] '%(X[icount, ifreq, 12],X[icount, ifreq, 13],X[icount, ifreq, 14],X[icount, ifreq, 15]) )

        train_list.append(MTdirname)
        xml_dict[icount] = xmldata
        if np.isnan(X).any():
            print( 'here is nan',MTdirname )

        icount += 1

    X = X[0:icount]
    y = y[0:icount]
    rate = rate[0:icount]

    return X,y,rate,train_list,xml_dict

#************************************************************************/
def getX_unrated(X,nfreq,fdir,lists,freq_interp):

    icount = 0
    unrated_list = []
    xml_dict = {}

    # loop for MT training list
    for MTdirname in lists:
        print( MTdirname )

        MTdir = fdir + '/' + MTdirname
        XMLlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('xml')] )
        # EDIlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('edi')] )

        # check there is only one xml and edi file in the directory
        checklists(XMLlist=XMLlist)

        XMLpath = MTdir + '/' + XMLlist[0]
        # EDIpath = MTdir + '/' + EDIlist[0]

        # if len(EDIlist) > 1:
        #     EDIlistlongest = max(EDIlist, key=len)
        #     EDIpath = MTdir + '/' + EDIlistlongest

        try:
            period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err, period_ori = pyMAGIQ.utils.iofiles.io.readXML(XMLpath,True,freq_interp)
            # period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err = utils.readXML(XMLpath,True,freq_interp)
            # period, Z, ZERR, Zapp, Zapp_err, Zphs, Zphs_err = utils.readEDIall(EDIpath, freq_interp)
        except ValueError:
            print( 'skip it! ValueError when reading XML file ', MTdirname )
            continue
        except IndexError:
            print( 'skip it! IndexError when reading XML file ', MTdirname )
            continue

        nperiod = len(Zapp)
        print(max(period_ori))

        # if maximum value period is smaller than 100 sec, skip it because interpolation won't work well
        if max(period_ori) < 10000.0:
            print( 'skip it! Maximum period is smaller than 10000 sec, so its danger to interpolate ', MTdirname )
            continue

        # # if number of frequency is differnet from nfreq, then skip to read it
        # if nperiod is not nfreq:
        #     print( 'skip this', MTdirname )
        #     continue

        # make X_train
        for ifreq in range(nfreq):
            # if Z[ifreq,0].real > 1.E+06:
            #     print( MTdirname, 'over 1.E+07')

            X[icount, ifreq,  0] = (np.log10( Zapp[ifreq,1] )+4)*0.01
            X[icount, ifreq,  1] = (np.log10( Zapp[ifreq,2] )+4)*0.01
            X[icount, ifreq,  2] = (np.log10( Zapp_err[ifreq,1] )+6)*0.01
            X[icount, ifreq,  3] = (np.log10( Zapp_err[ifreq,2] )+6)*0.01
            # X[icount, ifreq,  2] = (np.log10( Zapp_err[ifreq,1] )+4)*0.01
            # X[icount, ifreq,  3] = (np.log10( Zapp_err[ifreq,2] )+4)*0.01
            # Added 180.0 because phase could be goes under zero.
            # Also, we divided phase by 540 because it can go over 180.
            X[icount, ifreq,  4] = (Zphs[ifreq,1]+180.0) / 540.0
            X[icount, ifreq,  5] = (Zphs[ifreq,2]+180.0) / 540.0
            X[icount, ifreq,  6] = Zphs_err[ifreq,1] / 540.0
            X[icount, ifreq,  7] = Zphs_err[ifreq,2] / 540.0

            if ifreq < nfreq-1:
                # X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq,1]) - np.log10(Zapp[ifreq+1,1]))  ) * 0.1
                # X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq,2]) - np.log10(Zapp[ifreq+1,2]))  ) * 0.1
                # X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq,1])-np.log10(Zapp_err[ifreq+1,1])) ) * 0.1
                # X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq,2])-np.log10(Zapp_err[ifreq+1,2])) ) * 0.1
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq,1]) - np.log10(Zapp[ifreq+1,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq,2]) - np.log10(Zapp[ifreq+1,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq,1])-np.log10(Zapp_err[ifreq+1,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq,2])-np.log10(Zapp_err[ifreq+1,2])) ) * 0.01

                X[icount, ifreq, 12] = np.abs((Zphs[ifreq,1]-Zphs[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq,2]-Zphs[ifreq+1,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq,1]-Zphs_err[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq,2]-Zphs_err[ifreq+1,2]) / 540.0 )

            else:
                # X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq-1,1]) - np.log10(Zapp[ifreq,1]))  ) * 0.1
                # X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq-1,2]) - np.log10(Zapp[ifreq,2]))  ) * 0.1
                # X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq-1,1])-np.log10(Zapp_err[ifreq,1])) ) * 0.1
                # X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq-1,2])-np.log10(Zapp_err[ifreq,2])) ) * 0.1
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq-1,1]) - np.log10(Zapp[ifreq,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq-1,2]) - np.log10(Zapp[ifreq,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq-1,1])-np.log10(Zapp_err[ifreq,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq-1,2])-np.log10(Zapp_err[ifreq,2])) ) * 0.01

                X[icount, ifreq, 12] = np.abs((Zphs[ifreq-1,1]-Zphs[ifreq,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq-1,2]-Zphs[ifreq,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq-1,1]-Zphs_err[ifreq,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq-1,2]-Zphs_err[ifreq,2]) / 540.0 )

        unrated_list.append(MTdirname)
        xml_dict[icount] = xmldata
        if np.isnan(X).any():
            print( 'here is nan',MTdirname )
        icount += 1

    X = X[0:icount]

    return X,unrated_list,xml_dict

#************************************************************************/
def checklists(XMLlist=None,EDIlist=None,Zlist=None):
    # confirm number of xml is one
    # if XMLlist exists
    if XMLlist:
        if len(XMLlist) is not 1:
            print( 'number of xmlfile is not one' )
            sys.exit()

    # if EDIlist exists
    if EDIlist:
        if len(EDIlist) is not 1:
            print( 'number of edifile is not one' )
            sys.exit()

    # if Zlist exists
    if Zlist:
        if len(Zlist) is not 1:
            print( 'number of z-file is not one' )
            sys.exit()

#************************************************************************/
def getRating(XMLpath):
    tree = ET.parse(XMLpath)
    root = tree.getroot()
    # find Rating
    for children in root.iter('Rating'):
        Rating = children.text

    return Rating

#************************************************************************/
def plot_history(history,savepath,disp=True):

    plt.figure(figsize=(6,9))
    plt.subplot(2,1,1)
    # plot history of accuracy
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    # plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")

    plt.subplot(2,1,2)
    # plot history of model loss
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')

    plt.savefig(savepath,bbox_inches='tight',format='eps')
    if disp:
        plt.show()

#************************************************************************/
def ratehistogram1d(rate,fname=None):
    unique, counts = np.unique(rate, return_counts=True)

    height = [counts[0], counts[1], counts[2], counts[3], counts[4]]
    bars = ('1', '2', '3', '4', '5')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos,height)
    plt.xticks(y_pos,bars)
    plt.xlabel('Rate')
    plt.ylabel('Number of MT sites')
    if not fname:
        plt.show()

    else:
        plt.savefig(fname, bbox_inches='tight')

#************************************************************************/
def ratehistogram2d(est,ans,nrate,fpath=None):

    histogram = np.zeros( (nrate,nrate), dtype=np.int )
    nMTtest = len(est)
    vmin = 0.5
    vmax = 5.5

    # cast estimation to logical
    index = np.argmax(est,axis=1)
    for i in range(nMTtest):
        est_logic = np.zeros( (nrate,), dtype=np.int )
        # cast estimation to logical
        est_logic[index[i]] = 1

        # make 2d distribution
        histogram += np.outer( est_logic[0:nrate], ans[i,0:nrate].astype(int) )

    print( histogram )

    plt.figure()
    # visualize
    plt.imshow(histogram, origin='lower', extent=[vmin, vmax, vmin, vmax] )
    plt.colorbar()
    # plt.title('confusion matrix')
    plt.xlabel('Given rate')
    plt.ylabel('Predicted rate')
    if not fpath:
        plt.show()

    else:
        plt.savefig(fpath, bbox_inches='tight', format='eps')

#************************************************************************/
def randomsortXY(X, y, SiteID):

    X_r = np.zeros(X.shape)
    y_r = np.zeros(y.shape)
    SiteID_r = np.zeros(SiteID.shape)

    randindex = np.random.randint(len(X),size=len(X))

    X_r = X[randindex,:]
    y_r = y[randindex,:]
    SiteID_r = SiteID[randindex]

    return X_r, y_r, SiteID_r, randindex

#************************************************************************/
def rotateXML(xmldata,angle,nfreq):

    # convert decl in radius
    theta = np.radians(angle)
    # create rotation matrix
    c,s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    RT = R.T

    for k, v in xmldata.items():
        if k=='Z' or k=='InvCoh' or k=='ResCov' or k=='ZVAR':
            for i in range(nfreq):
                MatGeogra = np.zeros((2,2),dtype=v.dtype)
                MatGeomag = np.zeros((2,2),dtype=v.dtype)

                MatGeogra = pyMAGIQ.utils.iofiles.io.array1x4_to_2x2(v[i,0:4])

                MatGeogra = np.dot(MatGeogra,R)
                MatGeomag = np.dot(RT,MatGeogra)

                xmldata[k][i,0:4] = pyMAGIQ.utils.iofiles.io.array2x2_to_1x4(MatGeomag)


    return xmldata

#************************************************************************/
def xml2Z(xml_dict, sitelist, nfreq, Rotate=False, freq_interp=None, unrated=False):

    rate = np.zeros((len(xml_dict)), dtype=int)
    Zapplist     = np.zeros((len(xml_dict),nfreq,4), dtype=float)
    Zapp_errlist = np.zeros((len(xml_dict),nfreq,4), dtype=float)
    Zphslist     = np.zeros((len(xml_dict),nfreq,4), dtype=float)
    Zphs_errlist = np.zeros((len(xml_dict),nfreq,4), dtype=float)
    if not unrated:
        y = np.zeros((len(xml_dict),),dtype=np.int)

    for icount in range(len(xml_dict)):
        xmldata = xml_dict[icount]
        siteID = sitelist[icount]

        # read xml parameters
        nfreq = xmldata['nfreq']
        Rating = xmldata['Rating']
        Declination = xmldata['Declination']
        freq = xmldata['freq']
        period = xmldata['period']
        Z = xmldata['Z']
        ZVAR = xmldata['ZVAR']
        InvCoh = xmldata['InvCoh']
        ResCov = xmldata['ResCov']
        lon = xmldata['lon']
        lat = xmldata['lat']
        readingZVAR = xmldata['ZVARExistence']

        if not unrated:
            y[icount] = Rating-1
            rate[icount] = Rating

        if Rotate:
            xmldata = rotateXML(xmldata,Declination,nfreq)

        # update ZVAR from InvCoh and ResCov
        if not readingZVAR:
            for i in range(nfreq):
                xmldata['ZVAR'][i,0] = np.abs( xmldata['ResCov'][i,0] * xmldata['InvCoh'][i,0] )
                xmldata['ZVAR'][i,1] = np.abs( xmldata['ResCov'][i,0] * xmldata['InvCoh'][i,3] )
                xmldata['ZVAR'][i,2] = np.abs( xmldata['ResCov'][i,3] * xmldata['InvCoh'][i,0] )
                xmldata['ZVAR'][i,3] = np.abs( xmldata['ResCov'][i,3] * xmldata['InvCoh'][i,3] )

        # update ZVAR from ZVAR
        else:
            for i in range(nfreq):
                xmldata['ZVAR'][i,0] = np.abs( xmldata['ZVAR'][i,0] )
                xmldata['ZVAR'][i,1] = np.abs( xmldata['ZVAR'][i,1] )
                xmldata['ZVAR'][i,2] = np.abs( xmldata['ZVAR'][i,2] )
                xmldata['ZVAR'][i,3] = np.abs( xmldata['ZVAR'][i,3] )

        Zapp, Zapp_err, Zphs, Zphs_err = pyMAGIQ.utils.conversion.trans.getAppRes(xmldata)

        # if freq_interp exists, then interpolate
        if freq_interp:
            nfreq_interp = len(freq_interp)

            period_interp = np.zeros( (nfreq_interp,),  dtype=float )
            for i in range(nfreq_interp):
                period_interp[i] = 1.0/freq_interp[i]

            Zapp_interp     = np.zeros((nfreq_interp,4),dtype=np.float32)
            Zapp_err_interp = np.zeros((nfreq_interp,4),dtype=np.float32)
            Zphs_interp     = np.zeros((nfreq_interp,4),dtype=np.float32)
            Zphs_err_interp = np.zeros((nfreq_interp,4),dtype=np.float32)

            for i in range(4):
                # xp in interp must be increasing, so flip it
                Zapp_interp[...,i]     = np.interp( freq_interp, freq, np.flip(Zapp[...,i],0)     )
                Zapp_err_interp[...,i] = np.interp( freq_interp, freq, np.flip(Zapp_err[...,i],0) )
                Zphs_interp[...,i]     = np.interp( freq_interp, freq, np.flip(Zphs[...,i],0)     )
                Zphs_err_interp[...,i] = np.interp( freq_interp, freq, np.flip(Zphs_err[...,i],0) )

            Zapplist[icount]     = Zapp_interp
            Zapp_errlist[icount] = Zapp_err_interp
            Zphslist[icount]     = Zphs_interp
            Zphs_errlist[icount] = Zphs_err_interp

        else:
            Zapplist[icount]     = Zapp
            Zapp_errlist[icount] = Zapp_err
            Zphslist[icount]     = Zphs
            Zphs_errlist[icount] = Zphs_err

    if not unrated:
        return y, rate, Zapplist, Zapp_errlist, Zphslist, Zphs_errlist
    else:
        return Zapplist, Zapp_errlist, Zphslist, Zphs_errlist

#************************************************************************/
def getXfromZ(Zapplist, Zapp_errlist, Zphslist, Zphs_errlist, nfreq, sitelist):

    X = np.zeros((np.shape(Zapplist)[0],nfreq, 16),dtype=float)

    for icount in range(np.shape(Zapplist)[0]):
        SiteID = sitelist[icount]

        Zapp     = Zapplist[icount,:,:]
        Zapp_err = Zapp_errlist[icount,:,:]
        Zphs     = Zphslist[icount,:,:]
        Zphs_err = Zphs_errlist[icount,:,:]

        for ifreq in range(nfreq):

            X[icount, ifreq,  0] = (np.log10( Zapp[ifreq,1] )+4)*0.01
            X[icount, ifreq,  1] = (np.log10( Zapp[ifreq,2] )+4)*0.01
            X[icount, ifreq,  2] = (np.log10( Zapp_err[ifreq,1] )+6)*0.01
            X[icount, ifreq,  3] = (np.log10( Zapp_err[ifreq,2] )+6)*0.01
            # Added 180.0 because phase could be goes under zero.
            # Also, we divided phase by 540 because it can go over 180.
            X[icount, ifreq,  4] = (Zphs[ifreq,1]+180.0) / 540.0
            X[icount, ifreq,  5] = (Zphs[ifreq,2]+180.0) / 540.0
            X[icount, ifreq,  6] = Zphs_err[ifreq,1] / 540.0
            X[icount, ifreq,  7] = Zphs_err[ifreq,2] / 540.0

            if ifreq < nfreq-1:
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq,1]) - np.log10(Zapp[ifreq+1,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq,2]) - np.log10(Zapp[ifreq+1,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq,1])-np.log10(Zapp_err[ifreq+1,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq,2])-np.log10(Zapp_err[ifreq+1,2])) ) * 0.01

                X[icount, ifreq, 12] = np.abs((Zphs[ifreq,1]-Zphs[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq,2]-Zphs[ifreq+1,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq,1]-Zphs_err[ifreq+1,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq,2]-Zphs_err[ifreq+1,2]) / 540.0 )

            else:
                X[icount, ifreq,  8] = np.abs((np.log10(Zapp[ifreq-1,1]) - np.log10(Zapp[ifreq,1]))  ) * 0.01
                X[icount, ifreq,  9] = np.abs((np.log10(Zapp[ifreq-1,2]) - np.log10(Zapp[ifreq,2]))  ) * 0.01
                X[icount, ifreq, 10] = np.abs((np.log10(Zapp_err[ifreq-1,1])-np.log10(Zapp_err[ifreq,1])) ) * 0.01
                X[icount, ifreq, 11] = np.abs((np.log10(Zapp_err[ifreq-1,2])-np.log10(Zapp_err[ifreq,2])) ) * 0.01

                X[icount, ifreq, 12] = np.abs((Zphs[ifreq-1,1]-Zphs[ifreq,1]) / 540.0 )
                X[icount, ifreq, 13] = np.abs((Zphs[ifreq-1,2]-Zphs[ifreq,2]) / 540.0 )
                X[icount, ifreq, 14] = np.abs((Zphs_err[ifreq-1,1]-Zphs_err[ifreq,1]) / 540.0 )
                X[icount, ifreq, 15] = np.abs((Zphs_err[ifreq-1,2]-Zphs_err[ifreq,2]) / 540.0 )

        if np.isnan(X).any():
            print( 'here is nan',SiteID )

    return X

#************************************************************************/
def dataaug(xml_dict, sitelist):

    ratelist1 = [1,2,3]
    ratelist2 = [4,5]
    rotationlist1 = [a for a in range(-15,16) if a % 1 == 0]
    rotationlist2 = [a for a in range(-10,11) if a % 10 == 0]
    # rotationlist1 = [a for a in range(10,360) if a % 10 == 0]
    # rotationlist2 = [a for a in range(10,360) if a % 90 == 0]
    dictcount = len(xml_dict)

    xml_dict_copy = xml_dict.copy()
    sitelist_copy = sitelist.copy()

    for icount in range(len(xml_dict)):
        xml_data = xml_dict[icount].copy()
        siteid   = sitelist[icount]

        Rating = xml_data['Rating']
        original_dec = xml_data['Declination']
        if Rating in ratelist1:

            adding_dict={}
            for rotate_angle in rotationlist1:
                # modify rotation angle in dict
                xml_data['Declination'] = rotate_angle + original_dec
                # add data to dict
                xml_dict_copy[dictcount] = xml_data.copy()

                # append siteid to sitelist
                sitelist_copy.append(siteid)

                # print(xml_dict[dictcount]['Declination'])
                dictcount += 1
        elif Rating in ratelist2:

            adding_dict={}
            for rotate_angle in rotationlist2:
                # modify rotation angle in dict
                xml_data['Declination'] = rotate_angle + original_dec
                # add data to dict
                xml_dict_copy[dictcount] = xml_data.copy()

                # append siteid to sitelist
                sitelist_copy.append(siteid)

                # print(xml_dict[dictcount]['Declination'])
                dictcount += 1

    return xml_dict_copy, sitelist_copy
