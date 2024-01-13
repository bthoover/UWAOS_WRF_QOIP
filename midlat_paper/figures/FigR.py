# Figure R:
# (a) WRF 0-hr unperturbed forecast of 2.0 PVU potential temperature (shaded), wind speed
#     at 335.0 K surface, and sea-level pressure
# (b) As per panel-a, but for WRF 12-hr unperturbed forecast
# (c) As per panel-a, but for WRF 24-hr unperturbed forecast
# (d) As per panel-a, but for WRF 36-hr unperturbed forecast
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
from analysis_dependencies import get_wrf_th
from analysis_dependencies import plan_section_plot
from analysis_dependencies import get_uvmet
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
    # extract potential temperature
    thta = np.asarray(get_wrf_th(wrfHdl)).squeeze()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(wrfHdl)).squeeze()
    # extract potential vorticity
    pvor = np.asarray(wrf.getvar(wrfHdl,'pvo')).squeeze()
    # extract wind components (rotated to Earth-relative)
    uwd, vwd = get_uvmet(wrfHdl)
    # interpolate potential temperature to 2.0 PVU surface
    dynThta = wrf.interplevel(field3d=thta,
                              vert=pvor,
                              desiredlev=2.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    # interpolate wind components to 335.0 K surface: this is roughly the potential
    # temperature that delineates the center of the subtropical jet
    uwd335 = wrf.interplevel(field3d=uwd,
                             vert=thta,
                             desiredlev=335.,
                             missing=np.nan,
                             squeeze=True,
                             meta=False)
    vwd335 = wrf.interplevel(field3d=vwd,
                             vert=thta,
                             desiredlev=335.,
                             missing=np.nan,
                             squeeze=True,
                             meta=False)
    # compute wind speed on 335.0 K surface
    spd335 = (uwd335**2. + vwd335**2.)**0.5
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP and thickness
    slpContours = np.arange(1004., 1024.1, 4.)  # these contours are reduced from the full range of
                                              # sea-level pressure to pick out the low-pressure
                                              # centers
    thtContours = np.arange(270., 375.1, 5.)
    spdContours = np.arange(36., 150.1, 6.)
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[slp, spd335],
                                             contIntervalList=[slpContours, spdContours],
                                             contColorList=['yellow', 'red'],
                                             contLineThicknessList=[1.0, 2.0],
                                             shadVariable=dynThta,
                                             shadInterval=thtContours,
                                             shadAlpha=1.0,
                                             datProj=datProj,
                                             plotProj=plotProj,
                                             shadCmap='gray',
                                             uVecVariable=None,
                                             vVecVariable=None,
                                             vectorThinning=None,
                                             vecColor=None,
                                             figax=ax)
    # add contour labels to spd335
    ax.clabel(cons[1], levels=spdContours[::2])
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

    # FIG Ra: 0-hr unperturbed WRF forecast 2.0 PVU pot. temp, 335K wind speed, and SLP
    fcstHr = 0
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGR_panel_A')

    # FIG Ra: 12-hr unperturbed WRF forecast 2.0 PVU pot. temp, 335K wind speed, and SLP
    fcstHr = 12
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGR_panel_B')

    # FIG Rc: 24-hr unperturbed WRF forecast 2.0 PVU pot. temp, 335K wind speed, and SLP
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGR_panel_C')

    # FIG Rd: 36-hr unperturbed WRF forecast 2.0 PVU pot. temp, 335K wind speed, and SLP
    fcstHr = 36
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGR_panel_D')
#
# end
#
