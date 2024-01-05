# FIG I: Cross-sections
# a) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 400-700 hPa for
#    least-intense simulation (NW-->SE cross-section)
# b) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 400-700 hPa for
#    least-intense simulation (cross-shear cross-section)
# c) Cross-section perturbation wind speed and plan-section mean wind speed and wind speed perturbation 400-700 hPa for
#    least-intense simulation (cross-shear cross-section)
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
        unpTmean = unpTmean + unpTlev
        ptdTmean = ptdTmean + ptdTlev
        unpSmean = unpSmean + unpSlev
        ptdSmean = ptdSmean + ptdSlev
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
    spdRange = np.arange(24.,100.1,8.)
    tmpRange = np.arange(230., 280.1, 4.)
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

def generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lineColor, figureName):
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 40000., 70000., "T")),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=[lineColor]
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')

def generate_cross_speed_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lineColor, figureName):
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 40000., 70000., "S")),
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
    # define directories for unperturbe and least-intense (neg) simulations
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
    # define initialization time and forecast time
    dtInit = datetime.datetime(2020, 3, 6, 12)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # FIG Ia: least-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (NW-->SE)
    latBeg = 47.0
    lonBeg = -102.0
    latEnd = 27.0
    lonEnd = -72.0
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGI_panel_A')
    # FIG Ib: least-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    #         temperature perturbation (cross-shear)
    latBeg = 45.0
    lonBeg = -83.0
    latEnd = 25.0
    lonEnd = -88.0
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGI_panel_B')
    # FIG Ic: least-intense simulation cross-section of perturbation wind speed and 400-700 hPa mean
    #         wind-speed perturbation (cross-shear)
    latBeg = 45.0
    lonBeg = -83.0
    latEnd = 25.0
    lonEnd = -94.0
    generate_cross_speed_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGI_panel_C')
#
# end
#

