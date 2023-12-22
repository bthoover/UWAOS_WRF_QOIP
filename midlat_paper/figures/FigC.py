# Figure C:
# (a) WRF 6-hr unperturbed forecast of 850 hPa potential temperature advection (shaded),
#     300 hPa vorticity, 850 hPa potential temperature, and sea-level pressure
# (b) As per panel-a, but for WRF 12-hr unperturbed forecast
# (c) As per panel-a, but for WRF 18-hr unperturbed forecast
# (d) As per panel-a, but for WRF 24-hr unperturbed forecast
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
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import get_wrf_grad
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
    # extract potential temperature
    thta = np.asarray(get_wrf_th(wrfHdl)).squeeze()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(wrfHdl)).squeeze()
    # extract vorticity
    vor = np.asarray(get_wrf_kinematic(wrfHdl,'vor')).squeeze()
    # extract wind components (non-rotated to Earth-relative, to align with grid-relative
    # gradients in potential temperature when computing advection)
    uwdGrd = np.asarray(wrfHdl.variables['U']).squeeze()
    vwdGrd = np.asarray(wrfHdl.variables['V']).squeeze()
    # destagger wind components to mass-points
    uwdGrdM = wrf.destagger(uwdGrd, stagger_dim=2)
    vwdGrdM = wrf.destagger(vwdGrd, stagger_dim=1)
    # interpolate potential temperature to 850 hPa surface
    thta850 = wrf.interplevel(field3d=thta,
                              vert=wrf.getvar(wrfHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    # interpolate wind components to 850 hPa surface
    uwdGrd850 = wrf.interplevel(field3d=uwdGrdM,
                                vert=wrf.getvar(wrfHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    vwdGrd850 = wrf.interplevel(field3d=vwdGrdM,
                                vert=wrf.getvar(wrfHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # interpolate vorticity to 300 hPa surface
    vor300 = wrf.interplevel(field3d=vor,
                             vert=wrf.getvar(wrfHdl,'p'),
                             desiredlev=30000.,
                             missing=np.nan,
                             squeeze=True,
                             meta=False)
    # compute gradients of 850 hPa potential temperature
    dThdX, dThdY = get_wrf_grad(wrfHdl, thta850)
    # compute advection of 850 hPa potential temperature
    thtaAdv850 = np.multiply(-uwdGrd850,dThdX) + np.multiply(-vwdGrd850,dThdY)
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP and thickness
    slpContours = np.arange(900., 1012.1, 4.)  # these contours are reduced from the full range of
                                              # sea-level pressure to pick out the low-pressure
                                              # centers
    thtContours = np.arange(270., 375.1, 5.)
    vorContours = 1.0E-05 * np.arange(5., 35.1, 5.)
    advContours = 1.0E-03 * np.arange(-5., 5.01, 0.5)
    mask = np.ones(np.shape(advContours), dtype=bool)
    mask[np.where(advContours==0.)] = False
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                             lat=lat,
                                             lon=lon,
                                             contVariableList=[slp, thta850, vor300],
                                             contIntervalList=[slpContours, thtContours, vorContours],
                                             contColorList=['black', 'gray', 'green'],
                                             contLineThicknessList=[1.0, 1.0, 2.0],
                                             shadVariable=thtaAdv850,
                                             shadInterval=advContours[mask],
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

    # FIG Ca: 6-hr unperturbed WRF forecast 850 pot. temp. and adv., 300 hPa vorticity, and SLP
    fcstHr = 6
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGC_panel_A')
    # FIG Cb: 12-hr unperturbed WRF forecast 850 pot. temp. and adv., 300 hPa vorticity, and SLP
    fcstHr = 12
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGC_panel_B')
    # FIG Cc: 18-hr unperturbed WRF forecast 850 pot. temp. and adv., 300 hPa vorticity, and SLP
    fcstHr = 18
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGC_panel_C')
    # FIG Cd: 24-hr unperturbed WRF forecast 850 pot. temp. and adv., 300 hPa vorticity, and SLP
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 'FIGC_panel_D')
#
# end
#
