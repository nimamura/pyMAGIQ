#!/usr/bin/env python

"""
=============
plotUSmap module
=============

Functions
----------
    - plotZ

NI, 2020

"""
#=================================================================
 #imports
#=================================================================
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


#************************************************************************/
def plotZ(alldata,targetSiteID,fpath):

    fig = plt.figure(figsize=(14,12))
    outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)

    for i in range(9):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        # target Site ID name
        target = targetSiteID[i]

        # read variable
        Zapp_xy     = alldata[target]['Zapp_xy']
        Zapp_err_xy = alldata[target]['Zapp_err_xy']
        Zphs_xy     = alldata[target]['Zphs_xy']
        Zphs_err_xy = alldata[target]['Zphs_err_xy']

        Zapp_yx     = alldata[target]['Zapp_yx']
        Zapp_err_yx = alldata[target]['Zapp_err_yx']
        Zphs_yx     = alldata[target]['Zphs_yx']
        Zphs_err_yx = alldata[target]['Zphs_err_yx']

        period      = alldata[target]['period']
        prate       = alldata[target]['prate']
        if 'grate' in alldata[target].keys():
            grate   = alldata[target]['grate']
            title_str = 'Pred Rate: ' + str(prate) + '  Given Rate: ' + str(grate)
        else:
            title_str = 'Pred Rate: ' + str(prate)


        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.grid()
            if j == 0:
                ax.set_title(title_str)
                ax.errorbar(period, Zapp_xy, yerr=Zapp_err_xy, fmt='o', mfc='none', capsize=4, label='xy')
                ax.errorbar(period, Zapp_yx, yerr=Zapp_err_yx, fmt='o', mfc='none', capsize=4, label='yx', c='r')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim([1.E-0,1.E+5])
                plt.setp(ax.get_xticklabels(), visible=False)
                if i ==0 or i==3 or i== 6:
                    ax.set_ylabel("App. Res.")

            else:
                ax.errorbar(period,Zphs_xy,yerr=Zphs_err_xy,fmt='o',mfc='none',capsize=4)
                ax.errorbar(period,Zphs_yx,yerr=Zphs_err_yx,fmt='o',mfc='none',capsize=4,c='r')
                ax.set_xscale('log')
                ax.set_xlim([1.E-0,1.E+5])
                ax.set_ylim([0,90])
                ax.set_yticks([0,30,60,90])
                if i ==0 or i==3 or i== 6:
                    ax.set_ylabel("Phase")

                if i >5:
                    ax.set_xlabel("Period (sec)")
                # ax.grid()

            fig.add_subplot(ax)

    plt.savefig(fpath,format='eps')
    # plt.show()
