#!/usr/bin/env python

"""
=============
io module
=============

Functions
----------
    - readTSHmt
    - readTSEmt
    - readRTHobs
    - readTF
    - readZRR
    - readZRRformatted
    - readPointInterp
    - read_rate
    - read_rate_ytrain
    - read_siteID


NI, 2018

"""

#=================================================================
 #imports
#=================================================================
import sys
import csv
import os.path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pyMAGIQ

####################
# Helper functions
####################
def convert_float(s):
    try:
        return float(s)
    except:
        return None

def convert_int(s):
    try:
        return int(s)
    except:
        return None
def convert_datetime(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except:
        return None

def get_text(base, name):
    try:
        return base.find(name).text
    except:
        return None

#=================================================================
def readTSHmt(fpath):

    fid = open(fpath, 'r')
    lines = fid.readlines()
    nline = len(lines)
    fid.close()

    TSHmt = pd.read_csv(fpath,sep=',',nrows=nline,header=None,names=['Hx','Hy','Hz'])

    return TSHmt

#=================================================================
def readTSEmt(fpath):

    fid = open(fpath, 'r')
    lines = fid.readlines()
    nline = len(lines)
    fid.close()

    TSEmt = pd.read_csv(fpath,sep=',',nrows=nline,header=None,names=['Ex','Ey'])

    return TSEmt

#=================================================================
# fpath: path to outputn.asc
# Nobs: number of geomagnetic observatory
def readRTHobs(fpath):

    fid = open(fpath, 'r')
    lines = fid.readlines()
    nline = int(lines[0])
    fid.close()

    RTHobs = pd.read_csv(fpath,sep=',',skiprows=1,nrows=nline,header=None)
    # RTHobs = pd.read_csv(fpath,sep=' ',skiprows=1,nrows=nline,header=None)
    # replace 99999 with NaN
    RTHobs = RTHobs.replace(99999.0, np.nan)
    # fill NaN with before value
    RTHobs = RTHobs.fillna(method='bfill')
    RTHobs = RTHobs.fillna(method='ffill')

    return RTHobs

#=================================================================
# TF is defined as (in case of Nobs=4)
# 1st row : Txx of 1st geomagentic observatory
# 2nd row : Txy of 1st geomagentic observatory
# 3rd row : Txx of 2nd geomagentic observatory
# ...
# 8th row : Txy of 4th geomagentic observatory
# 9th row : Tyx of 1st geomagentic observatory
# 10throw : Tyy of 1st geomagentic observatory
# ...
#
def readTF(TFpath,Nobs,mband):

    TF = np.zeros((mband,Nobs*2*2),dtype=np.complex)
    iline = 0
    for line in open(TFpath, 'r'):
        data = line.split()
        for icount in range(Nobs*2*2):
            i, j = np.safe_eval(data[icount])
            TF[iline, icount] = np.complex(i, j)

        iline += 1

    return TF

#=================================================================
def readZRRformatted(ZRRpath,mband_edi,mband_tf):

    Z = np.zeros((mband_tf,4),dtype=np.complex)
    fid = open(ZRRpath,'r')
    lines = fid.readlines()
    fid.close()

    # difference between mband_edi and mband_tf
    mdiff = mband_edi - mband_tf

    for i in range(mband_tf):
        data = lines[i+mdiff].split()
        Z[i,0] = np.complex( float(data[0]),float(data[1]) )
        Z[i,1] = np.complex( float(data[2]),float(data[3]) )
        Z[i,2] = np.complex( float(data[4]),float(data[5]) )
        Z[i,3] = np.complex( float(data[6]),float(data[7]) )

    decl = float(lines[-1])
    rot = np.radians(decl)

    #  Rotates Z from geomagnetic North to map North to generate E
    #  predictions in a coordinate system where they can be properly
    #  combined (counter clockwise rotation)
    for i in range(mband_tf):
        Z = rotMat(rot, Z)
        InvCoh = rotMat(rot, InvCoh)
        ResCov = rotMat(rot, ResCov)

    return Z


#=================================================================
def readZRR(ZRRpath,Rotate=False):

    fid = open(ZRRpath,'r')
    lines = fid.readlines()
    fid.close()

    ## read metadata
    # frequnecy info
    line = lines[5].replace('\n','').split()
    nfreq = int(line[7])
    # rotation angle in radius
    data = lines[7].split()
    rot = float(data[1])/180.0 *np.pi

    # initalize
    Z      = np.zeros((nfreq,4),dtype=np.complex)
    InvCoh = np.zeros((nfreq,4),dtype=np.complex)
    ResCov = np.zeros((nfreq,4),dtype=np.complex)
    periods = np.zeros((nfreq,),dtype=np.float)

    # starting line to read Z and ZERR
    initline = 13
    # number of lines for each frequency
    chnks = 13

    for i in range(nfreq):
        periods[i] = lines[initline+i*chnks+0].split()[2]
        data = lines[initline+i*chnks+4].split()
        Z[i,0] = np.complex( float(data[0]),float(data[1]) )
        Z[i,1] = np.complex( float(data[2]),float(data[3]) )
        data = lines[initline+i*chnks+5].split()
        Z[i,2] = np.complex( float(data[0]),float(data[1]) )
        Z[i,3] = np.complex( float(data[2]),float(data[3]) )

        # Inverse Coherent Signal Power Matrix
        data = lines[initline+i*chnks+7].split()
        InvCoh[i,0] = np.complex( float(data[0]),float(data[1]) )
        data = lines[initline+i*chnks+8].split()
        InvCoh[i,1] = np.complex( float(data[0]),-float(data[1]) )
        InvCoh[i,2] = np.complex( float(data[0]),float(data[1]) )
        InvCoh[i,3] = np.complex( float(data[2]),float(data[3]) )

        # Residual Covariance
        data = lines[initline+i*chnks+11].split()
        ResCov[i,0] = np.complex( float(data[2]),float(data[3]) )
        data = lines[initline+i*chnks+12].split()
        ResCov[i,1] = np.complex( float(data[2]),-float(data[3]) )
        ResCov[i,2] = np.complex( float(data[2]),float(data[3]) )
        ResCov[i,3] = np.complex( float(data[4]),float(data[5]) )

    #  Rotates Z from geomagnetic North to map North to generate E
    #  predictions in a coordinate system where they can be properly
    #  combined (counter clockwise rotation)
    if Rotate:
        Z = rotMat(rot, Z)
        InvCoh = rotMat(rot, InvCoh)
        ResCov = rotMat(rot, ResCov)

    return Z, InvCoh, ResCov, periods


#=================================================================
def rotMat(theta, Mat):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    for i in range(np.shape(Mat)[0]):
        Mattmp = Mat[i, :]
        Mattmp = array1x4_to_2x2(Mattmp)
        Mattmp = np.matmul(np.matmul(R, Mattmp), R.transpose())
        Mattmp = array2x2_to_1x4(Mattmp)
        Mat[i, :] = Mattmp

    return Mat


#=================================================================
# default xml file is in geographic coordinate.
# If rotate option is True, returned data is in geomagnetic coordinate
def readXML(XMLpath,Rotate=False,freq_interp=None):
    #
    root = ET.parse(XMLpath).getroot()

    xml_site = root.find("Site")

    # get Data
    loc = xml_site.find("Location")
    lat = convert_float(get_text(loc, "Latitude"))
    lon = convert_float(get_text(loc, "Longitude"))
    elev = convert_float(get_text(loc, "Elevation"))
    Declination = convert_float(get_text(loc, "Declination"))

    quality = xml_site.find("DataQualityNotes")
    Rating = convert_int(get_text(quality, "Rating"))
    min_period = convert_float(get_text(quality, "GoodFromPeriod"))
    max_period = convert_float(get_text(quality, "GoodToPeriod"))

    # get number of frequency from Data's attribute
    eleData = root.findall("Data")
    nfreq = int( eleData[0].attrib['count'] )

    # find Rating
    # for children in root.iter('Rating'):
    #     Rating = int(children.text)
    #
    # # find decl
    # for children in root.iter('Declination'):
    #     Declination = float(children.text)
    #
    # # find latitude
    # for children in root.iter('Latitude'):
    #     lat = float(children.text)
    #
    # # find longitude
    # for children in root.iter('Longitude'):
    #     lon = float(children.text)

    # Canadian SPUD datasets is not set declination correctly,
    # so use orientaiton of Hx instead.
    if Declination == 0.0:
        print( 'Use Hx orientation as a declination angle, instead of original declination' )
        for children in root.iter('Magnetic'):
            if children.attrib['name'] == 'Hx':
                Declination = float(children.attrib['orientation'])

    period = np.zeros( (nfreq,),  dtype=np.float32 )
    freq   = np.zeros( (nfreq,),  dtype=np.float32 )
    Z      = np.zeros( (nfreq,4), dtype=np.complex )
    ZVAR   = np.zeros( (nfreq,4), dtype=np.float32 )
    InvCoh = np.zeros( (nfreq,4), dtype=np.complex )
    ResCov = np.zeros( (nfreq,4), dtype=np.complex )

    # find all element including Period
    elePeriod = eleData[0].findall("Period")
    # first, read period and sort its frequency
    for i in range(len(elePeriod)):
        period[i] = float( elePeriod[i].attrib['value'] )
        freq[i] = 1.0/period[i]
    # index of sorted freq
    argindex = np.argsort(period)
    period = sorted(period)
    sortedfreq = sorted(freq)
    period_ori = period

    # flag to read ZVAR or to read INVSIGCOV and RESIDCOV
    # True:read ZVAR, False: read INVSIGCOV and RESIDCONV
    elementExistence = elePeriod[0].findall("Z.INVSIGCOV")
    if elementExistence:
        readingZVAR = False
        print( 'reading Z.INVSIGCOV' )
    else:
        readingZVAR = True
        print( 'reading Z.VAR' )

    # read XML data
    for i in range(len(elePeriod)):
        ii = argindex[i]
        eleZ = elePeriod[ii].findall("Z")

        if not readingZVAR:    # reading INVSIGCOV and RESIDCOV
            eleInvCoh = elePeriod[ii].findall("Z.INVSIGCOV")
            eleResCov = elePeriod[ii].findall("Z.RESIDCOV")
        else:  # or reading ZVAR
            eleZVAR   = elePeriod[ii].findall("Z.VAR")

        for j in range(4):
            real, imag = eleZ[0].findall('value')[j].text.split()
            Z[i,j] = np.complex( float(real), float(imag) )

            if not readingZVAR: # reading INVSIGCOV and RESIDCOV
                real, imag = eleInvCoh[0].findall('value')[j].text.split()
                InvCoh[i,j] = np.complex( float(real), float(imag) )
                real, imag = eleResCov[0].findall('value')[j].text.split()
                ResCov[i,j] = np.complex( float(real), float(imag) )
            else:   # or reading ZVAR
                real = eleZVAR[0].findall('value')[j].text
                ZVAR[i,j] = float(real)

    # save everything in dict
    xmldata={}
    xmldata['nfreq'] = nfreq
    xmldata['Rating'] = Rating
    xmldata['Declination'] = Declination
    xmldata['period'] = period
    xmldata['Z'] = Z
    xmldata['ZVAR'] = ZVAR
    xmldata['InvCoh'] = InvCoh
    xmldata['ResCov'] = ResCov
    xmldata['lon'] = lon
    xmldata['lat'] = lat

    # if rotate option is on, rotate it from geographic coordinate to geomagnetic
    # coordinate
    if Rotate:
        # convert decl in radius
        theta = np.radians(Declination)
        # create rotation matrix
        c,s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        RT = R.T

        for k, v in xmldata.items():
            if k=='Z' or k=='InvCoh' or k=='ResCov' or k=='ZVAR':
                for i in range(nfreq):
                    MatGeogra = np.zeros((2,2),dtype=v.dtype)
                    MatGeomag = np.zeros((2,2),dtype=v.dtype)

                    MatGeogra = array1x4_to_2x2(v[i,0:4])

                    MatGeogra = np.dot(MatGeogra,R)
                    MatGeomag = np.dot(RT,MatGeogra)

                    v[i,0:4] = array2x2_to_1x4(MatGeomag)

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
            # xp in interp must be increasing, so flip xp and yp
            # Zapp_interp[...,i]     = np.interp( freq_interp, sortedfreq, Zapp[...,i]     )
            # Zapp_err_interp[...,i] = np.interp( freq_interp, sortedfreq, Zapp_err[...,i] )
            # Zphs_interp[...,i]     = np.interp( freq_interp, sortedfreq, Zphs[...,i]     )
            # Zphs_err_interp[...,i] = np.interp( freq_interp, sortedfreq, Zphs_err[...,i] )

            Zapp_interp[...,i]     = np.interp( freq_interp, sortedfreq, np.flip(Zapp[...,i],0)     )
            Zapp_err_interp[...,i] = np.interp( freq_interp, sortedfreq, np.flip(Zapp_err[...,i],0) )
            Zphs_interp[...,i]     = np.interp( freq_interp, sortedfreq, np.flip(Zphs[...,i],0)     )
            Zphs_err_interp[...,i] = np.interp( freq_interp, sortedfreq, np.flip(Zphs_err[...,i],0) )

            # Zapp_interp[...,i]     = np.interp( freq_interp, np.flip(freq,0), np.flip(Zapp[...,i],0)     )
            # Zapp_err_interp[...,i] = np.interp( freq_interp, np.flip(freq,0), np.flip(Zapp_err[...,i],0) )
            # Zphs_interp[...,i]     = np.interp( freq_interp, np.flip(freq,0), np.flip(Zphs[...,i],0)     )
            # Zphs_err_interp[...,i] = np.interp( freq_interp, np.flip(freq,0), np.flip(Zphs_err[...,i],0) )

        return period_interp, xmldata, Zapp_interp, Zapp_err_interp, Zphs_interp, Zphs_err_interp, period_ori

    return period, xmldata, Zapp, Zapp_err, Zphs, Zphs_err, period_ori

#=================================================================
def array1x4_to_2x2(mat1):

    mat2 = np.zeros((2, 2), dtype=mat1.dtype)

    mat2[0, 0] = mat1[0]
    mat2[0, 1] = mat1[1]
    mat2[1, 0] = mat1[2]
    mat2[1, 1] = mat1[3]

    return mat2

#=================================================================
def array2x2_to_1x4(mat1):

    mat2 = np.zeros((1, 4), dtype=mat1.dtype)

    mat2[0, 0] = mat1[0, 0]
    mat2[0, 1] = mat1[0, 1]
    mat2[0, 2] = mat1[1, 0]
    mat2[0, 3] = mat1[1, 1]

    return mat2


#=================================================================
def readPointInterp(InterpPath):

    # read Pointinterp file
    fid=open(InterpPath,'r')
    lines = fid.readlines()
    fid.close()

    # number of interpolation points
    npoint = int(lines[0])

    # initialize returning variables
    st   = np.zeros((npoint,3), dtype=np.int)
    norm = np.zeros((npoint,3), dtype=np.float32)
    lenx = np.zeros((npoint,), dtype=np.float32)
    leny = np.zeros((npoint,), dtype=np.float32)

    # loop for interpolation point lines
    for i in range(npoint):
        # read st
        lists = lines[1+i*3].split()
        for j in range(3):
            st[i,j] = int(lists[j])

        # read norm
        lists = lines[2+i*3].split()
        for j in range(3):
            norm[i,j] = float(lists[j])

        # read lenx and leny
        lists = lines[3+i*3].split()
        lenx[i] = float(lists[0])
        lenx[i] = float(lists[1])

    return npoint,st,norm,lenx,leny

#************************************************************************/
def read_rate(path):

    ratelist = []

    with open(path,'r') as f:
        reader = csv.reader(f)

        for row in reader:
            maxindex = np.argmax(row) + 1
            ratelist.append( maxindex )

    return ratelist

#************************************************************************/
def read_rate_ytrain(path):

    ratelist = []

    with open(path,'r') as f:
        reader = csv.reader(f)

        for row in reader:
            ratelist.append( int(row[0])+1 )

    return ratelist


#************************************************************************/
def read_siteID(path):

    siteIDlist = []

    with open(path,'r') as f:
        reader = csv.reader(f)

        for row in reader:
            siteIDlist.append(row[0])

    return siteIDlist


#=================================================================
# default xml file is in geographic coordinate.
# If rotate option is True, returned data is in geomagnetic coordinate
def readXMLonly(XMLpath):
    #
    root = ET.parse(XMLpath).getroot()

    xml_site = root.find("Site")

    # get Data
    loc = xml_site.find("Location")
    lat = convert_float(get_text(loc, "Latitude"))
    lon = convert_float(get_text(loc, "Longitude"))
    elev = convert_float(get_text(loc, "Elevation"))
    Declination = convert_float(get_text(loc, "Declination"))

    quality = xml_site.find("DataQualityNotes")
    Rating = convert_int(get_text(quality, "Rating"))
    min_period = convert_float(get_text(quality, "GoodFromPeriod"))
    max_period = convert_float(get_text(quality, "GoodToPeriod"))

    # get number of frequency from Data's attribute
    eleData = root.findall("Data")
    nfreq = int( eleData[0].attrib['count'] )

    # Canadian SPUD datasets is not set declination correctly,
    # so use orientaiton of Hx instead.
    if Declination == 0.0:
        print( 'Use Hx orientation as a declination angle, instead of original declination' )
        for children in root.iter('Magnetic'):
            if children.attrib['name'] == 'Hx':
                Declination = float(children.attrib['orientation'])

    period = np.zeros( (nfreq,),  dtype=np.float32 )
    freq   = np.zeros( (nfreq,),  dtype=np.float32 )
    Z      = np.zeros( (nfreq,4), dtype=np.complex )
    ZVAR   = np.zeros( (nfreq,4), dtype=np.float32 )
    InvCoh = np.zeros( (nfreq,4), dtype=np.complex )
    ResCov = np.zeros( (nfreq,4), dtype=np.complex )

    # find all element including Period
    elePeriod = eleData[0].findall("Period")
    # first, read period and sort its frequency
    for i in range(len(elePeriod)):
        period[i] = float( elePeriod[i].attrib['value'] )
        freq[i] = 1.0/period[i]
    # index of sorted freq
    argindex = np.argsort(period)
    period = sorted(period)
    sortedfreq = sorted(freq)
    period_ori = period

    # flag to read ZVAR or to read INVSIGCOV and RESIDCOV
    # True:read ZVAR, False: read INVSIGCOV and RESIDCONV
    elementExistence = elePeriod[0].findall("Z.INVSIGCOV")
    if elementExistence:
        readingZVAR = False
        print( 'reading Z.INVSIGCOV' )
    else:
        readingZVAR = True
        print( 'reading Z.VAR' )

    # read XML data
    for i in range(len(elePeriod)):
        ii = argindex[i]
        eleZ = elePeriod[ii].findall("Z")

        if not readingZVAR:    # reading INVSIGCOV and RESIDCOV
            eleInvCoh = elePeriod[ii].findall("Z.INVSIGCOV")
            eleResCov = elePeriod[ii].findall("Z.RESIDCOV")
        else:  # or reading ZVAR
            eleZVAR   = elePeriod[ii].findall("Z.VAR")

        for j in range(4):
            real, imag = eleZ[0].findall('value')[j].text.split()
            Z[i,j] = np.complex( float(real), float(imag) )

            if not readingZVAR: # reading INVSIGCOV and RESIDCOV
                real, imag = eleInvCoh[0].findall('value')[j].text.split()
                InvCoh[i,j] = np.complex( float(real), float(imag) )
                real, imag = eleResCov[0].findall('value')[j].text.split()
                ResCov[i,j] = np.complex( float(real), float(imag) )
            else:   # or reading ZVAR
                real = eleZVAR[0].findall('value')[j].text
                ZVAR[i,j] = float(real)

    # save everything in dict
    xmldata={}
    xmldata['nfreq'] = nfreq
    xmldata['Rating'] = Rating
    xmldata['Declination'] = Declination
    xmldata['freq'] = sorted(freq)
    xmldata['period'] = period
    xmldata['Z'] = Z
    xmldata['ZVAR'] = ZVAR
    xmldata['InvCoh'] = InvCoh
    xmldata['ResCov'] = ResCov
    xmldata['lon'] = lon
    xmldata['lat'] = lat
    xmldata['ZVARExistence'] = readingZVAR

    return period, xmldata


#************************************************************************/
def get_XMLlists(nfreq,fdir,lists,freq_interp,unrated=False):

    icount = 0
    train_list = []
    xml_dict = {}
    # loop for MT training list
    for MTdirname in lists:
        print( MTdirname )

        MTdir = fdir + '/' + MTdirname
        # read xml and edi file in the folder
        XMLlist = sorted( [f for f in os.listdir(MTdir) if f.endswith('xml')] )

        # check there is only one xml and edi file in the directory
        pyMAGIQ.utils.MAGIQlib.MAGIQlib.checklists(XMLlist=XMLlist)

        # path to xml and edi file
        XMLpath = MTdir + '/' + XMLlist[0]

        # get information in XML file
        try:
            period, xmldata = pyMAGIQ.utils.iofiles.io.readXMLonly(XMLpath)
        except ValueError:
            print( 'skip it! ValueError when reading XML file ', MTdirname )
            continue
        except IndexError:
            print( 'skip it! IndexError when reading XML file ', MTdirname )
            continue

        # if maximum value periodEDI is smaller than 100 sec, skip it because interpolation won't work well
        if max(period) < 10000.0:
            print( 'skip it! Maximum period is smaller than 1000 sec', MTdirname )
            continue

        if not unrated:
            if xmldata['Rating'] == 0:
                print( 'skip it! rating is 0', MTdirname )
                continue

        train_list.append(MTdirname)
        xml_dict[icount] = xmldata

        icount += 1

    return xml_dict,train_list
