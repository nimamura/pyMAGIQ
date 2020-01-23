#!/usr/bin/env python

"""
=============
plotUSmap module
=============

Functions
----------
    - plotmap

NI, 2018

"""
# =================================================================
# imports
# =================================================================
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


# ************************************************************************/
def plotmap(extent, latlist, lonlist, ratelist, fpath=None):
    """
    plot rate information on map
    """

    size = 8
    # extent = [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor='w', edgecolor='k')

    # Map
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    ax.coastlines(resolution='110m')
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='gray', linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor='white')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black', facecolor='white')

    cmap = plt.cm.get_cmap('rainbow', 5)
    sca = plt.scatter(lonlist, latlist, c=ratelist, s=size, cmap=cmap)
    plt.colorbar(sca, shrink=0.6)
    plt.clim(0.5, 5.5)

    if fpath:
        fig.savefig(fpath, format='eps')
    else:
        plt.show()
