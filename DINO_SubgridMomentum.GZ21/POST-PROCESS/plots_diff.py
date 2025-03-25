import os
import argparse
import numpy as np
import xarray as xr
import cmocean

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors


def make_plot(data,lon,lat,infos,output):
    # args
    title, cmap, norm, tfs = infos
    data = tfs(data)
    # figure
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    # color map
    pcm = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm)
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
    plt.title(title)
    # write fig
    plt.savefig(output, bbox_inches='tight')
    plt.close()


def main(filepath_ref, filepath, var_name, infos, freq):

    # read files
    try:
        ds = xr.open_dataset(filepath)
        ds_ref = xr.open_dataset(filepath_ref)
    except:
        return

    # coordinates
    try:
        lon = ds.nav_lon.values
    except:
        lon = ds.nav_lon_grid_T.values
    try:
        lat = ds.nav_lat.values
    except:
        lat = ds.nav_lat_grid_T.values

    # get fields
    ke_ref = getattr(ds_ref,var_name).values[-1]
    if len(ke_ref.shape) == 3:
        ke_ref = ke_ref[0,:,:]
    ke = getattr(ds,var_name).values[-1]
    if len(ke.shape) == 3:
        ke = ke[0,:,:]
    diff_ke = ke - ke_ref

    # plot
    plotpath = 'NEVERWORLD.GZ21_' + freq + '_diffke.png'
    make_plot(diff_ke,lon,lat,infos,plotpath)



if __name__=="__main__":

    # mean kinetic energy anomaly
    infos = [ 'KE-DINO.GZ21 - KE-DINO (m2/s2)' , cmocean.cm.balance , colors.Normalize(vmin=-0.02, vmax=0.02), lambda x: x ]
    main( filepath_ref='NEVERWORLD_STD.1y_gridT.nc' , filepath='NEVERWORLD.1y_gridT.nc' , var_name='ke' , infos=infos , freq='1y' )

