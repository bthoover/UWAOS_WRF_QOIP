# Figure E:
# (a) WRF 24-hr least-intense perturbed forecast of SLP (shaded), unperturbed SLP, and unperturbed
#     850-500 hPa thickness
# (b) WRF 24-hr least-intense perturbed 300 hPa geopotential height (shaded), unperturbed 300 hPa
#     geopotential height, and unperturbed 300 hPa wind speed
# (c) As per panel-a, but for most-intense perturbed forecast
# (d) As per panel-b, but for most-intense perturbed forecast
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
from analysis_dependencies import get_uvmet
from analysis_dependencies import plan_section_plot
#
# define internal functions
#
def generate_figure_panel_left(unpHdl, ptdHdl, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    ptdSlp = np.asarray(get_wrf_slp(ptdHdl)).squeeze()
    # extract unperturbed geopotential height
    unpHgt = np.asarray(wrf.getvar(unpHdl,'z')).squeeze()
    # interpolate geopotential height to 850 hPa surface
    unpHgt850 = wrf.interplevel(field3d=unpHgt,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # interpolate geopotential height to 500 hPa surface
    unpHgt500 = wrf.interplevel(field3d=unpHgt,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute 850-500 hPa thickness
    unpThk = unpHgt500 - unpHgt850
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP, thickness, and SLP difference
    slpContours = np.arange(900., 1040.1, 4.)
    thkContours = np.arange(3700., 4500.1, 50.)
    slpDiffContours = np.arange(-30., 30.1, 4.).astype('float16')
    mask = np.ones(np.shape(slpDiffContours)).astype('bool')
    mask[np.where(slpDiffContours==0.)] = False
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[ptdSlp-unpSlp, unpSlp, unpThk],
                                             contIntervalList=[slpDiffContours, slpContours, thkContours],
                                             contColorList=['white', 'black', '#19a83d'],
                                             contLineThicknessList=[1.0, 1.5, 2.0],
                                             shadVariable=ptdSlp-unpSlp,
                                             shadInterval=slpDiffContours[mask],
                                             shadAlpha=1.0,
                                             datProj=datProj,
                                             plotProj=plotProj,
                                             shadCmap='seismic',
                                             uVecVariable=None,
                                             vVecVariable=None,
                                             vectorThinning=None,
                                             vecColor=None,
                                             figax=ax)
    # add contour labels to slp
    ax.clabel(cons[1], levels=slpContours[::2])
    # save file
    fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')


def generate_figure_panel_right(unpHdl, ptdHdl, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract geopotential height
    unpHgt = np.asarray(wrf.getvar(unpHdl,'z')).squeeze()
    ptdHgt = np.asarray(wrf.getvar(ptdHdl,'z')).squeeze()
    # extract unperturbed wind components
    unpUwd, unpVwd = get_uvmet(unpHdl)
    # interpolate geopotential height to 300 hPa surface
    unpHgt300 = wrf.interplevel(field3d=unpHgt,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=30000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt300 = wrf.interplevel(field3d=ptdHgt,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=30000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # interpolate wind components to 300 hPa
    unpUwd300 = wrf.interplevel(field3d=unpUwd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=30000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpVwd300 = wrf.interplevel(field3d=unpVwd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=30000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute 300 hPa wind speed
    unpSpd300 = (unpUwd300**2. + unpVwd300**2.)**0.5
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP, thickness, and SLP difference
    hgtContours = np.arange(8300., 9800.1, 80.)
    spdContours = np.arange(40., 78.1, 8.)
    hgtDiffContours = np.arange(-180., 180.1, 15.).astype('float16')
    mask = np.ones(np.shape(hgtDiffContours)).astype('bool')
    mask[np.where(hgtDiffContours==0.)] = False
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[ptdHgt300-unpHgt300, unpHgt300, unpSpd300],
                                             contIntervalList=[hgtDiffContours, hgtContours, spdContours],
                                             contColorList=['white', 'black', 'green'],
                                             contLineThicknessList=[1.0, 1.5, 2.0],
                                             shadVariable=ptdHgt300-unpHgt300,
                                             shadInterval=hgtDiffContours[mask],
                                             shadAlpha=1.0,
                                             datProj=datProj,
                                             plotProj=plotProj,
                                             shadCmap='seismic',
                                             uVecVariable=None,
                                             vVecVariable=None,
                                             vectorThinning=None,
                                             vecColor=None,
                                             figax=ax)
    # add contour labels to wind speed
    ax.clabel(cons[2], levels=spdContours[::2])
    # save file
    fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
#
# begin
#
if __name__ == "__main__":
    # define directory of unperturbed, perturbed WRF forecasts
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
    # define datetime of WRF initialization
    dtInit = datetime.datetime(2020, 3, 6, 12)

    # FIG Ea: 24-hr forecast of least-intense SLP perturbation, unperturbed SLP and 850-500 thickness
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_figure_panel_left(unpHdl, ptdHdl, 'FIGE_panel_A')

    # FIG Eb: 24-hr forecast of least-intense 300 hPa geop. hgt.  perturbation, unperturbed 300 hPa geopt. hgt. and wind speed
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_figure_panel_right(unpHdl, ptdHdl, 'FIGE_panel_B')

    # FIG Ec: 24-hr forecast of most-intense SLP perturbation, unperturbed SLP and 850-500 thickness
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_figure_panel_left(unpHdl, ptdHdl, 'FIGE_panel_C')

    # FIG Ed: 24-hr forecast of most-intense 300 hPa geop. hgt.  perturbation, unperturbed 300 hPa geopt. hgt. and wind speed
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_figure_panel_right(unpHdl, ptdHdl, 'FIGE_panel_D')
#
# end
#
