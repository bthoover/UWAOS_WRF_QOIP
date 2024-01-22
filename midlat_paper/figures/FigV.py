# FIG V: Cross-sections
# a) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 400-700 hPa for
#    most-intense simulation (along-shear cross-section)
# b) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 500-900 hPa for
#    most-intense simulation (along-shear cross-section)
# c) Cross-section perturbation wind speed and plan-section mean wind speed and wind speed perturbation 250-400 hPa for
#    most-intense simulation (along-shear cross-section)
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_tk
from analysis_dependencies import get_wrf_th
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import get_wrf_grad
from analysis_dependencies import get_xsect
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import plan_section_plot
from analysis_dependencies import cross_section_plot
from analysis_dependencies import interpolate_sigma_levels
import datetime
import wrf
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import matplotlib
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
from scipy.interpolate import interp1d
#
# define internal functions
#
# define function for plan-section plot: mean wind-speed and temperature between 2 pressure-levels,
#                                        and perturbation mean wind-speed OR temperature
def right_panel(ax, payloadTuple):
    # expand payloadTuple into unpHdl and ptdHdl, and interpolation level
    unpHdl = payloadTuple[0]
    ptdHdl = payloadTuple[1]
    intLevLow = payloadTuple[2]
    intLevHigh = payloadTuple[3]
    shdType = payloadTuple[4]
    # define intLevs as vector of levels between intLevLow and intLevHigh
    # at intervals intLevInt
    intLevInt = 2500.  # Pa
    intLevs = np.arange(intLevLow, intLevHigh+0.01, intLevInt)
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # extract wind and compute speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    # loop through intLevs and compute wind speed and temperature
    unpTmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdTmean = np.zeros(np.shape(lat))
    unpSmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdSmean = np.zeros(np.shape(lat))
    for intLev in intLevs:
        # compute the wind speed at intLev
        unpSlev = wrf.interplevel(field3d=unpSpd,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdSlev = wrf.interplevel(field3d=ptdSpd,
                                  vert=wrf.getvar(ptdHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # compute the temperature at intLev
        unpTlev = wrf.interplevel(field3d=get_wrf_tk(unpHdl),
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdTlev = wrf.interplevel(field3d=get_wrf_tk(ptdHdl),
                                  vert=wrf.getvar(ptdHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # add temperature and wind terms to 2d mean fields
        unpTmean = unpTmean + unpTlev  #np.nansum(np.dstack((unpTmean,unpTlev)),2)  #unpTmean + unpTlev 
        ptdTmean = ptdTmean + ptdTlev  #np.nansum(np.dstack((ptdTmean,ptdTlev)),2)  #ptdTmean + ptdTlev
        unpSmean = unpSmean + unpSlev  #np.nansum(np.dstack((unpSmean,unpSlev)),2)  #unpSmean + unpSlev
        ptdSmean = ptdSmean + ptdSlev  #np.nansum(np.dstack((ptdSmean,ptdSlev)),2)  #ptdSmean + ptdSlev
    # divide by number of levels to produce mean-values
    unpTmean = unpTmean / np.size(intLevs)
    ptdTmean = ptdTmean / np.size(intLevs)
    unpSmean = unpSmean / np.size(intLevs)
    ptdSmean = ptdSmean / np.size(intLevs)
    # determine shading type and assign shading variable
    if shdType == "S":
        shdVar = ptdSmean - unpSmean
    if shdType == "T":
        shdVar = ptdTmean - unpTmean
    # generate plan-section plot
    spdRange = np.arange(32.,100.1,8.)
    tmpRange = np.arange(200., 400.1, 4.)
    shdRange = np.arange(-2., 2.01, 0.25).astype('float16')
    mask = np.ones(np.shape(shdRange),dtype='bool')
    mask[np.where(shdRange==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpTmean, unpSmean],
                                            contIntervalList=[tmpRange, spdRange], 
                                            contColorList=['orange','green'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=shdVar,
                                            shadInterval=shdRange[mask],
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title('')
    # add contour labels to spd
    ax.clabel(cons[1],levels=spdRange[::2])
    return ax

def generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lev1, lev2, lineColor, figureName):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # compute temperature
    unpT = get_wrf_tk(unpHdl)
    ptdT = get_wrf_tk(ptdHdl)
    # compute potential vorticity
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    # compute (unperturbed) potential temperature
    unpThta = get_wrf_th(unpHdl)
    # compute (unperturbed) wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed temperature and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdT_int = interpolate_sigma_levels(ptdT, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # generate cross-section plot
    xSectShadInterval=np.arange(-5., 5.01, 0.5)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    fig, latLists, lonLists = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, pvorrng, pvorrng, spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdT_int-unpT,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, lev1, lev2, "T")),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=[lineColor]
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')

def generate_cross_speed_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lev1, lev2, lineColor, figureName):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # compute potential vorticity
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    # compute (unperturbed) potential temperature
    unpThta = get_wrf_th(unpHdl)
    # compute wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed wind speed and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdSpd_int = interpolate_sigma_levels(ptdSpd, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # generate cross-section plot
    xSectShadInterval=np.arange(-5., 5.01, 0.5)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    fig, latLists, lonLists = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, pvorrng, pvorrng, spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdSpd_int-unpSpd,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, lev1, lev2, "S")),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=[lineColor]
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')
#
# begin
#
if __name__ == "__main__":
    # define directories for unperturbe and most-intense (pos) simulations
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initialization time and forecast time
    dtInit = datetime.datetime(2019, 11, 25, 12)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # FIG Va: most-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (along-shear)
    latBeg = 56.75
    lonBeg = -175.5
    latEnd = 48.75
    lonEnd = -142.5
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 40000., 70000., 'black', 'FIGV_panel_A')
    # FIG Vb: most-intense simulation cross-section of perturbation temperature and 500-900 hPa mean
    #         temperature perturbation (along-shear)
    latBeg = 42.75
    lonBeg = -172.5
    latEnd = 48.75
    lonEnd = -142.5
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 50000., 90000., 'black', 'FIGV_panel_B')
    # FIG Vc: most-intense simulation cross-section of perturbation wind speed and 250-400 hPa mean
    #         wind-speed perturbation (along-shear)
    latBeg = 65.75
    lonBeg = -180.0
    latEnd = 32.75
    lonEnd = -158.0
    generate_cross_speed_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 25000., 40000., 'black', 'FIGV_panel_C')
#
# end
#
