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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xml.etree.ElementTree as ET

#************************************************************************/
def plotMap(datadir,xmin,xmax,ymin,ymax,latlist,lonlist,ratelist,fpath):

    sc = 12

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

    cm = plt.cm.get_cmap('rainbow',5)
    x,y = m(lonlist, latlist)
    sca= m.scatter(x,y,c=ratelist, s=sc,cmap=cm)

    plt.colorbar(sca, shrink=0.6)
    plt.clim(0.5,5.5)
    fig.savefig(fpath,format='eps')
    plt.show()
