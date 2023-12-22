# Figure A:
# (a) OPC East Atlantic surface analysis for 1200 UTC 06 March 2020
# (b) OPC East Atlantic surface analysis for 1200 UTC 07 March 2020
# (c) WRF 0-hr unperturbed forecast of sea level pressure and 850-500 hPa thickness
# (d) WRF 24-hr unperturbed forecast of sea level pressure and 850-500 hPa thickness

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
    # interpolate heights to 850 and 500 hPa
    z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                           vert=prs,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # compute 850-500 thickness
    thk = z500 - z850
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP and thickness
    slpContours =np.arange(900., 1040.1, 4.)
    thkContours =np.arange(3700., 4500.1, 50.)
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[slp, thk],
                                             contIntervalList=[slpContours, thkContours],
                                             contColorList=['black', '#19a83d'],
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
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    # define datetime of WRF initialization
    dtInit = datetime.datetime(2020, 3, 6, 12)

    # FIG Ac: 0-hr unperturbed WRF forecast SLP and 850-500 hPa thickness
    fcstHr = 0
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGA_panel_C')

    # FIG Ad: 24-hr unperturbed WRF forecast SLP and 850-500 hPa thickness
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGA_panel_D')
#
# end
#
