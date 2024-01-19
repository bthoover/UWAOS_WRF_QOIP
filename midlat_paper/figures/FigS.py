# Figure S: 500 hPa geopotential height and sea-level pressure of unperturbed simulation
# (a) 0-hr forecast
# (b) 12-hr forecast
# (c) 24-hr forecast
# (d) 36-hr forecast

# load modules (UWAOS_WRF_QOIP compliant)
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
from netCDF4 import Dataset
import wrf
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import plan_section_plot
#
# define internal functions
#
def generate_figure_panel(wrfHdl, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format
    lat = np.asarray(wrfHdl.variables['XLAT']).squeeze()
    lon = np.asarray(wrfHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(wrfHdl)
    plotProj = ccrs.PlateCarree()
    # extract pressure for interpolation
    prs = np.asarray(wrf.getvar(wrfHdl,'p')).squeeze()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(wrfHdl)).squeeze()
    # interpolate heights to 500 hPa
    z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP and thickness
    slpContours = np.arange(1004., 1024.1, 4.)
    hgtContours = np.arange(4800.,6200.1,80.)
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[slp, z500],
                                             contIntervalList=[slpContours, hgtContours],
                                             contColorList=['#a10acf', 'black'],
                                             contLineThicknessList=[1.5, 2.0],
                                             shadVariable=None,
                                             shadInterval=None,
                                             shadAlpha=None,
                                             datProj=datProj,
                                             plotProj=plotProj,
                                             shadCmap=None,
                                             uVecVariable=None,
                                             vVecVariable=None,
                                             vectorThinning=None,
                                             vecColor=None,
                                             figax=ax)
    # add contour labels to slp
    ax.clabel(cons[0], levels=slpContours[::2])
    # save file
    fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return

#
# begin
#
if __name__ == "__main__":
    # define directory of unperturbed WRF forecast
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    # define datetime of WRF initialization
    dtInit = datetime.datetime(2019, 11, 25, 12)

    # FIG Sa: 0-hr unperturbed WRF forecast SLP and 500 hPa geop hgt
    fcstHr = 0
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGS_panel_A')

    # FIG Sb: 12-hr unperturbed WRF forecast SLP and 500 hPa geop hgt
    fcstHr = 12
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGS_panel_B')
    
    # FIG Sc: 24-hr unperturbed WRF forecast SLP and 500 hPa geop hgt
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGS_panel_C')
    
    # FIG Sd: 36-hr unperturbed WRF forecast SLP and 500 hPa geop hgt
    fcstHr = 36
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGS_panel_D')


#
# end
#
