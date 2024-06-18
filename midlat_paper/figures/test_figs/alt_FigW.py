# FIG W: Cross-sections
# a) Cross-section perturbation PV and plan-section PV perturbation and temperature perturbation 700-950 hPa for
#    most-intense simulation at 0-hrs (across-shear cross-section)
# b) Cross-section perturbation PV and plan-section PV perturbation and temperature perturbation 700-950 hPa for
#    most-intense simulation at 6-hrs (across-shear cross-section)
# c) Cross-section perturbation PV and plan-section PV perturbation and temperature perturbation 700-950 hPa for
#    most-intense simulation at 12-hrs (across-shear cross-section)
# d) Cross-section perturbation PV and plan-section PV perturbation and temperature perturbation 700-950 hPa for
#    most-intense simulation at 18-hrs (across-shear cross-section)
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
from analysis_dependencies import extend_xsect_point
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
# define function for plan-section plot: mean temperature between 2 pressure-levels,
#                                        and perturbation temperature and PV
def right_panel(ax, payloadTuple):
    # expand payloadTuple into unpHdl and ptdHdl, and interpolation level
    unpHdl = payloadTuple[0]
    ptdHdl = payloadTuple[1]
    intLevLow = payloadTuple[2]
    intLevHigh = payloadTuple[3]
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
    # loop through intLevs and compute wind speed and temperature
    unpTmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdTmean = np.zeros(np.shape(lat))
    unpPVmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdPVmean = np.zeros(np.shape(lat))
    for intLev in intLevs:
        # compute the PV at intLev
        unpPVlev = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdPVlev = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
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
        # add temperature and PV terms to 2d mean fields
        unpTmean = unpTmean + unpTlev
        ptdTmean = ptdTmean + ptdTlev
        unpPVmean = unpPVmean + unpPVlev
        ptdPVmean = ptdPVmean + ptdPVlev
    # divide by number of levels to produce mean-values
    unpTmean = unpTmean / np.size(intLevs)
    ptdTmean = ptdTmean / np.size(intLevs)
    unpPVmean = unpPVmean / np.size(intLevs)
    ptdPVmean = ptdPVmean / np.size(intLevs)
    # assign shading variable
    shdVar = ptdTmean - unpTmean
    # generate plan-section plot
    pvRange = np.arange(0.,4.01,0.2).astype('float16')
    pvRange = pvRange[np.where(pvRange!=0.)]
    tmpRange = np.arange(230., 280.1, 4.)
    shdRange = np.arange(-4., 4.01, 0.5).astype('float16')
    mask = np.ones(np.shape(shdRange),dtype='bool')
    mask[np.where(shdRange==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpTmean, ptdPVmean-unpPVmean, ptdPVmean-unpPVmean],
                                            contIntervalList=[tmpRange, pvRange, pvRange], 
                                            contColorList=['orange','white','black'],
                                            contLineThicknessList=[1.0, 3.0, 2.0],
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
    return ax


# generate_cross_PV_panel: produce figure-panel with cross-section of perturbation PV, unperturbed wind speed
#                          and potential temperature, and unperturbed/perturbed 2.0 PVU surface, with plan-
#                          section plot of cross-section line with features defined by right_panel().
def generate_cross_PV_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lineColor, figureName):
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
    # compute (unperturbed) wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed temperature and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # generate cross-section plot
    xSectShadInterval= np.arange(-4.,4.01,0.5).astype('float16')
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
                                 xSectShadVariable=ptdPvor_int-unpPvor,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 70000., 95000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=[lineColor]
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')
    return
#
# begin
#
if __name__ == "__main__":
    # define directories for unperturbe and most-intense (pos) simulations
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initialization time and forecast time
    dtInit = datetime.datetime(2019, 11, 25, 12)
    # FIG Wa: 0-hr forecast most-intense simulation cross-section of perturbation PV and 700-950 hPa mean
    # PV perturbation and temperature perturbation (across-shear)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latCen = 45.75
    lonCen = -155.5
    bearing = 0.
    kmDist = 1200.
    latBeg, lonBeg = extend_xsect_point(latCen, lonCen, bearing, kmDist)
    latEnd, lonEnd = extend_xsect_point(latCen, lonCen, bearing+180., kmDist)
    generate_cross_PV_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGW_panel_A')
    # FIG Wb: 6-hr forecast most-intense simulation cross-section of perturbation PV and 700-950 hPa mean
    # PV perturbation and temperature perturbation (across-shear)
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latCen = 48.0
    lonCen = -149.5
    bearing = 5.
    kmDist = 1200.
    latBeg, lonBeg = extend_xsect_point(latCen, lonCen, bearing, kmDist)
    latEnd, lonEnd = extend_xsect_point(latCen, lonCen, bearing+180., kmDist)
    generate_cross_PV_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGW_panel_B')
    # FIG Wc: 12-hr forecast most-intense simulation cross-section of perturbation PV and 700-950 hPa mean
    # PV perturbation and temperature perturbation (across-shear)
    fcstHr = 12
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latCen = 48.25
    lonCen = -146.0
    bearing = 15.
    kmDist = 1200.
    latBeg, lonBeg = extend_xsect_point(latCen, lonCen, bearing, kmDist)
    latEnd, lonEnd = extend_xsect_point(latCen, lonCen, bearing+180., kmDist)
    generate_cross_PV_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGW_panel_C')
    # FIG Wd: 18-hr forecast most-intense simulation cross-section of perturbation PV and 700-950 hPa mean
    # PV perturbation and temperature perturbation (across-shear)
    fcstHr = 18
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latCen = 48.0
    lonCen = -142.25
    bearing = 18.
    kmDist = 1200.
    latBeg, lonBeg = extend_xsect_point(latCen, lonCen, bearing, kmDist)
    latEnd, lonEnd = extend_xsect_point(latCen, lonCen, bearing+180., kmDist)
    generate_cross_PV_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGW_panel_D')
#
# end
#
