#!/usr/bin/python
# -*- coding: utf-8 -*-

#*************************************************************************
 #  File Name: plotMap_Evec.py
 #
 #  Created By: Naoto Imamura (nimamura)
 #
 #  Purpose: This script creates continuous images of E vector calculated by
 #   1D and 3D impedance matrix. A movie is created by these images using following
 #   commands.
 #
 #      ffmpeg -i image%04d.jpg -an output.mp4
 #      ffmpeg -i output.mp4 -vf setpts=PTS/0.125 -an output2.mp4 (if you need slower movie)
 #
 #
 #  Examples:
 #     Run: python plotMap_Evec.py
 #
 #*************************************************************************

#************************************************************************/
 #imports
#************************************************************************/
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime
import csv
import xml.etree.ElementTree as ET

#************************************************************************/
 #                            main()
 #Main function in python program
 #Inputs:
 #Outputs:
#************************************************************************/
def main(argv):

    target = 'unrated'
    # target = 'train'

    datadir = '/Users/nimamura/GIC/survey/ZML/data'
    # datadir = '/Users/nimamura/GIC/survey/EastCoast1989'

    if target =='unrated':
        xmin = -140;
        xmax = -65.0;
        ymin = 45.0;
        ymax = 65.0;
        ratepath   = datadir + '/y_unrated.csv'
        siteIDpath = datadir + '/siteID_unrated.csv'

        # read
        ratelist = read_rate(ratepath)

    else:
        xmin = -140;
        xmax = -65.0;
        ymin = 28.0;
        ymax = 50.0;
        ratepath   = datadir + '/y_train.csv'
        siteIDpath = datadir + '/siteID_train.csv'

        # read rate
        ratelist = read_rate_ytrain(ratepath)

    siteIDlist = read_siteID(siteIDpath)

    latlist,lonlist = read_lonlat(datadir,siteIDlist,target)

    for i in range(len(siteIDlist)):
        print( siteIDlist[i], ratelist[i] )

    plotMap(datadir,xmin,xmax,ymin,ymax,latlist,lonlist,ratelist)

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

#************************************************************************/
def read_lonlat(datadir,siteIDlist,target):

    latlist = []
    lonlist = []
    ratinglist = []

    for siteID in siteIDlist:
        fname = siteID.replace('MT_TF_','') + '.xml'
        XMLpath = datadir + '/' + target + '/' + siteID + '/' + fname

        tree = ET.parse(XMLpath)
        root = tree.getroot()

        # find latitude
        for children in root.iter('Latitude'):
            lat = float(children.text)
        latlist.append(lat)

        # find longitude
        for children in root.iter('Longitude'):
            lon = float(children.text)
        lonlist.append(lon)

    return latlist,lonlist

#************************************************************************/
def plotMap(datadir,xmin,xmax,ymin,ymax,latlist,lonlist,ratelist):

    sc = 12
    c1 = np.array([213,162,132])
    c2 = np.array([213,200,135])
    c3 = np.array([152,213,135])
    c4 = np.array([134,214,175])
    c5 = np.array([134,213,213])

    fig = plt.figure(figsize=(16,9), dpi=150, facecolor='w', edgecolor='k')

    # Map
    m = Basemap(projection='merc',
                     resolution='l',
                     llcrnrlon=xmin,
                     llcrnrlat=ymin,
                     urcrnrlon=xmax,
                     urcrnrlat=ymax)

    m.drawcoastlines(color='lightgray')
    m.drawcountries(color='lightgray')
    m.fillcontinents(color='white', lake_color='#eeeeee', zorder = 0);
    m.drawmapboundary(fill_color='#eeeeee')

    cm = plt.cm.get_cmap('cool',5)
    # cm = plt.cm.get_cmap('summer_r',5)
    # cm = plt.cm.get_cmap('RdYlBu')
    x,y = m(lonlist, latlist)
    sca= m.scatter(x,y,c=ratelist, s=sc,cmap=cm)

    # for j in range(len(x)):
    #     if ratelist[j]==1:
    #         m.scatter(x[j],y[j],color=c1/255.0, s=sc)
    #     if ratelist[j]==2:
    #         m.scatter(x[j],y[j],color=c2/255.0, s=sc)
    #     if ratelist[j]==3:
    #         m.scatter(x[j],y[j],color=c3/255.0, s=sc)
    #     if ratelist[j]==4:
    #         m.scatter(x[j],y[j],color=c4/255.0, s=sc)
    #     if ratelist[j]==5:
    #         m.scatter(x[j],y[j],color=c5/255.0, s=sc)

    plt.colorbar(sca, shrink=0.6)
    plt.clim(0.5,5.5)
    fig.savefig(datadir+'/image.eps',format='eps')
    plt.show()


#************************************************************************/
 #                            __name__
 #Needed to define main function in python program
#************************************************************************/
if __name__ == "__main__":
    main(sys.argv[1:])
