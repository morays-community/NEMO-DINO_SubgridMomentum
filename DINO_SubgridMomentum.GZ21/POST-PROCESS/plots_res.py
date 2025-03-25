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


def main(filepath, var_name, infos, freq):

    # read files
    try:
        ds = xr.open_dataset(filepath)
    except:
        return

    print(f'Plotting {var_name}')

    try:
        lon = ds.nav_lon.values
    except:
        lon = ds.nav_lon_grid_T.values
    try:
        lat = ds.nav_lat.values
    except:
        lat = ds.nav_lat_grid_T.values

    # get fields
    var_val = getattr(ds,var_name).values[-1]
    if len(var_val.shape) == 3:
        var_val = var_val[0,:,:]

    # plot
    plotpath = 'NEVERWORLD.GZ21_' + freq + '_' + var_name + '.png'
    make_plot(var_val,lon,lat,infos,plotpath)



if __name__=="__main__":

    # snapshots
    # ---------
    # kinetic energy
    infos = [ 'KE (m2/s2)' , cmocean.cm.haline , colors.Normalize(vmin=0.0, vmax=0.03), lambda x: x ]
    main( filepath='NEVERWORLD.1d_gridTsurf.nc' , var_name='sKE' , infos=infos , freq='1d' )

    # usurf
    infos = [ 'Sea Surface U (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.3, vmax=0.3), lambda x: x ]
    main( filepath='NEVERWORLD.1d_gridUsurf.nc' , var_name='sozocrtx' , infos=infos , freq='1d' )

    # vsurf
    infos = [ 'Sea Surface V (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.3, vmax=0.3), lambda x: x ]
    main( filepath='NEVERWORLD.1d_gridVsurf.nc' , var_name='somecrty' , infos=infos , freq='1d' )

    # uf
    infos = [ 'U-grid forcing (m/s2)' , cmocean.cm.balance , colors.Normalize(vmin=-0.00005, vmax=0.00005), lambda x: x ]
    main( filepath='NEVERWORLD.1d_gridUsurf.nc' , var_name='soext_uf' , infos=infos , freq='1d' )

    # vf
    infos = [ 'V-grid forcing (m/s2)' , cmocean.cm.balance , colors.Normalize(vmin=-0.00005, vmax=0.00005), lambda x: x ]
    main( filepath='NEVERWORLD.1d_gridVsurf.nc' , var_name='soext_vf' , infos=infos , freq='1d' )

    # yearly mean
    # kinetic energy
    infos = [ 'KE (m2/s2)' , cmocean.cm.haline , colors.Normalize(vmin=0.0, vmax=0.05), lambda x: x ]
    main( filepath='NEVERWORLD.1y_gridT.nc' , var_name='ke' , infos=infos , freq='1y' )

    # usurf
    infos = [ 'Sea Surface U (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.3, vmax=0.3), lambda x: x ]
    main( filepath='NEVERWORLD.1y_gridU.nc' , var_name='uoce' , infos=infos , freq='1y' )

    # vsurf
    infos = [ 'Sea Surface V (m/s)' , cmocean.cm.diff , colors.Normalize(vmin=-0.3, vmax=0.3), lambda x: x ]
    main( filepath='NEVERWORLD.1y_gridV.nc' , var_name='voce' , infos=infos , freq='1y' )

    # uf
    infos = [ 'U-grid forcing (m/s2)' , cmocean.cm.balance , colors.Normalize(vmin=-0.00005, vmax=0.00005), lambda x: x ]
    main( filepath='NEVERWORLD.1y_gridU.nc' , var_name='ext_uf' , infos=infos , freq='1y' )

    # vf
    infos = [ 'V-grid forcing (m/s2)' , cmocean.cm.balance , colors.Normalize(vmin=-0.00005, vmax=0.00005), lambda x: x ]
    main( filepath='NEVERWORLD.1y_gridV.nc' , var_name='ext_vf' , infos=infos , freq='1y' )
