# FIG V: Cross-sections
# a) Cross-section perturbation temperature and plan-section mean wind and temperature perturbation 700-795 hPa for
#    most-intense simulation (along-shear) for hour-0
# b) Plan-section perturbation mean 700-950 hPa temperature advection and unperturbed temperature and mean
#    sea-level pressure for hour-0
# c) Plan-section perturbation mean 700-950 hPa temperature advection and unperturbed temperature and mean
#    sea-level pressure for hour-3
# d) Plan-section perturbation mean 700-950 hPa temperature advection and unperturbed temperature and mean
#    sea-level pressure for hour-6
# e) Plan-section perturbation mean 700-950 hPa temperature advection and unperturbed temperature and mean
#    sea-level pressure for hour-9
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import get_wrf_tk
from analysis_dependencies import get_wrf_th
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import get_wrf_grad
from analysis_dependencies import get_wrf_kinematic
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
    # extract wind
    u,v = get_uvmet(unpHdl)
    unpUwd = u
    unpVwd = v
    u,v = get_uvmet(ptdHdl)
    ptdUwd = u
    ptdVwd = v
    # extract unperturbed sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl))
    # loop through intLevs and compute wind, temperature and temp-adv
    unpTmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdTmean = np.zeros(np.shape(lat))
    unpUmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdUmean = np.zeros(np.shape(lat))
    unpVmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdVmean = np.zeros(np.shape(lat))
    for intLev in intLevs:
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
        # compute the wind components at intLev
        unpUlev = wrf.interplevel(field3d=unpUwd,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdUlev = wrf.interplevel(field3d=ptdUwd,
                                  vert=wrf.getvar(ptdHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        unpVlev = wrf.interplevel(field3d=unpVwd,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdVlev = wrf.interplevel(field3d=ptdVwd,
                                  vert=wrf.getvar(ptdHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # add speed, temperature and wind terms to 2d mean fields
        unpTmean = unpTmean + unpTlev
        ptdTmean = ptdTmean + ptdTlev
        unpUmean = unpUmean + unpUlev
        ptdUmean = ptdUmean + ptdUlev
        unpVmean = unpVmean + unpVlev
        ptdVmean = ptdVmean + ptdVlev
    # divide by number of levels to produce mean-values
    unpTmean = unpTmean / np.size(intLevs)
    ptdTmean = ptdTmean / np.size(intLevs)
    unpUmean = unpUmean / np.size(intLevs)
    ptdUmean = ptdUmean / np.size(intLevs)
    unpVmean = unpVmean / np.size(intLevs)
    ptdVmean = ptdVmean / np.size(intLevs)
    # shading and vector settings
    shdVar = ptdTmean - unpTmean
    shdRange = np.arange(-2., 2.01, 0.25).astype('float16')
    uVec = ptdUmean-unpUmean
    vVec = ptdVmean-unpVmean
    vThin = 8
    vCol = 'black'
    vScale = 20
    # generate plan-section plot
    tmpRange = np.arange(230., 280.1, 4.)
    slpRange = np.arange(900., 1040.1, 6.)
    mask = np.ones(np.shape(shdRange),dtype='bool')
    mask[np.where(shdRange==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp, unpTmean],
                                            contIntervalList=[slpRange, tmpRange], 
                                            contColorList=['black','#0c8212'],
                                            contLineThicknessList=[1.0, 0.8],
                                            shadVariable=shdVar,
                                            shadInterval=shdRange[mask],
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=uVec,
                                            vVecVariable=vVec,
                                            vectorThinning=vThin,
                                            vecColor=vCol,
                                            vectorScale=vScale,
                                            figax=ax)
    # add a title
    ax.set_title('')
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 70000., 95000., "T")),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=[lineColor]
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')

    
def generate_plan_advection_panel(unpHdl, ptdHdl, figureName):
    # define intLevs as vector of levels between intLevLow and intLevHigh
    # at intervals intLevInt
    intLevLow = 70000.   # Pa
    intLevHigh = 95000. # Pa
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
    # extract wind
    u,v = get_uvmet(unpHdl)
    unpUwd = u
    unpVwd = v
    u,v = get_uvmet(ptdHdl)
    ptdUwd = u
    ptdVwd = v
    # extract unperturbed sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl))
    # loop through intLevs and compute temp-adv and unperturbed temperature
    unpAmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdAmean = np.zeros(np.shape(lat))
    unpTmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    for intLev in intLevs:
        # compute the temperature advection at intLev
        u,v = get_uvmet(unpHdl)
        T = get_wrf_tk(unpHdl)
        Z = get_wrf_kinematic(unpHdl,'vor')
        dTdx, dTdy = get_wrf_grad(unpHdl, T)
        unpTadv = np.multiply(-u,dTdx) + np.multiply(-v,dTdy)
        unpAlev = wrf.interplevel(field3d=unpTadv,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # compute the unperturbed temperature at intLev
        unpTlev = wrf.interplevel(field3d=T,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        u,v = get_uvmet(ptdHdl)
        T = get_wrf_tk(ptdHdl)
        Z = get_wrf_kinematic(ptdHdl,'vor')
        dTdx, dTdy = get_wrf_grad(ptdHdl, T)
        ptdTadv = np.multiply(-u,dTdx) + np.multiply(-v,dTdy)
        ptdAlev = wrf.interplevel(field3d=ptdTadv,
                                  vert=wrf.getvar(ptdHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # add level to 2d mean fields
        unpAmean = unpAmean + unpAlev
        ptdAmean = ptdAmean + ptdAlev
        unpTmean = unpTmean + unpTlev
    # divide by number of levels to produce mean-values
    unpTmean = unpTmean / np.size(intLevs)
    unpAmean = unpAmean / np.size(intLevs)
    ptdAmean = ptdAmean / np.size(intLevs)
    # determine shading type and assign shading variable
    shdVar = ptdAmean - unpAmean
    shdRange = 1.0E-04 * np.arange(-2.,2.01,0.25).astype('float16')
    uVec = None
    vVec = None
    vThin = None
    vCol = None
    vScale = None
    # generate plan-section plot
    tmpRange = np.arange(210., 300.1, 4.)
    slpRange = np.arange(900., 1040.1, 6.)
    mask = np.ones(np.shape(shdRange),dtype='bool')
    mask[np.where(shdRange==0.)] = False
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,6),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp, unpTmean],
                                            contIntervalList=[slpRange, tmpRange], 
                                            contColorList=['black', '#0c8212'],
                                            contLineThicknessList=[1.0, 0.85],
                                            shadVariable=shdVar,
                                            shadInterval=shdRange[mask],
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=uVec,
                                            vVecVariable=vVec,
                                            vectorThinning=vThin,
                                            vecColor=vCol,
                                            vectorScale=vScale,
                                            figax=ax)
    # add a title
    ax.set_title('')
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
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # FIG Va: most-intense simulation cross-section of perturbation temperature and 700-950 hPa mean
    # temperature perturbation (along-shear) at 0-hrs
    latBeg = 43.75
    lonBeg = -172.5
    latEnd = 46.75
    lonEnd = -142.5
    lonEnd = -142.5
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGV_panel_A')
    # FIG Vb: most-intense simulation plan-section of 700-950 hPa mean perturbation temperature advection,
    #         unperturbed temperature and SLP at 0-hrs
    dtInit = datetime.datetime(2019, 11, 25, 12)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_plan_advection_panel(unpHdl,ptdHdl,'FIGV_Panel_B')
    # FIG Vc: most-intense simulation plan-section of 700-950 hPa mean perturbation temperature advection,
    #         unperturbed temperature and SLP at 3-hrs
    dtInit = datetime.datetime(2019, 11, 25, 12)
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_plan_advection_panel(unpHdl,ptdHdl,'FIGV_Panel_C')
    # FIG Vd: most-intense simulation plan-section of 700-950 hPa mean perturbation temperature advection,
    #         unperturbed temperature and SLP at 6-hrs
    dtInit = datetime.datetime(2019, 11, 25, 12)
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_plan_advection_panel(unpHdl,ptdHdl,'FIGV_Panel_D')
    # FIG Ve: most-intense simulation plan-section of 700-950 hPa mean perturbation temperature advection,
    #         unperturbed temperature and SLP at 9-hrs
    dtInit = datetime.datetime(2019, 11, 25, 12)
    fcstHr = 9
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    generate_plan_advection_panel(unpHdl,ptdHdl,'FIGV_Panel_E')
#
# end
#
