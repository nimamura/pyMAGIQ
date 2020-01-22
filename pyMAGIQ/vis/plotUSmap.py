#!/usr/bin/env python

"""
=============
plotUSmap module
=============

Functions
----------
    - plotMap

NI, 2018

"""
#=================================================================
 #imports
#=================================================================
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy
import cartopy.crs as ccrs
import xml.etree.ElementTree as ET

#************************************************************************/
def plotMap(datadir,xmin,xmax,ymin,ymax,latlist,lonlist,ratelist,fpath):

    sc = 8
    extent = [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(16,9), dpi=150, facecolor='w', edgecolor='k')

    # Map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    ax.coastlines(resolution='110m')
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='gray', linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black',facecolor='white')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black',facecolor='white')

    cm = plt.cm.get_cmap('rainbow',5)
    sca = plt.scatter(lonlist, latlist, c=ratelist, s=sc, cmap=cm)
    plt.colorbar(sca, shrink=0.6)
    plt.clim(0.5,5.5)
    fig.savefig(fpath,format='eps')
    plt.show()
