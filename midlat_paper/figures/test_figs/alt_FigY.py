# Figure Y: Cross-section of perturbation omega, with plan-section perturbation mean 400-700 hPa geopotential height
#           perturbation and perturbation temperature advection by the geostrophic wind, and mean wind speed for
# a) most-intense simulation 3-hr forecast
# b) most-intense simulation 6-hr forecast
# c) most-intense simulation 9-hr forecast
# d) most-intense simulation 12-hr forecast
#
# Cross-sections are designed to cut across height/t-adv mean perturbation and align across-shear
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
# define function for plan-section plot: 250 hPa geopotential height and wind-speed
def right_panel(ax, payloadTuple):
    # expand payloadTuple into cross-section bounds, unpHdl and ptdHdl, and interpolation level
    latMin = payloadTuple[0]
    lonMin = payloadTuple[1] if payloadTuple[1] > 0. else payloadTuple[1] + 360.  # fix longitude convention
    latMax = payloadTuple[2]
    lonMax = payloadTuple[3] if payloadTuple[3] > 0. else payloadTuple[3] + 360.  # fix longitude convention
    unpHdl = payloadTuple[4]
    ptdHdl = payloadTuple[5]
    intLevLow = payloadTuple[6]
    intLevHigh = payloadTuple[7]
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
    # extract unperturbed wind and compute speed
    u,v = get_uvmet(unpHdl)
    spd = np.sqrt(u**2. + v**2.)
    # define f and g for geostrophic wind calculation
    f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)  # 2*omega*sin(phi), s^-1
    g = 9.80665  # m/s^2
    # loop through intLevs and compute geostrophic advection of temperature
    unpTADVmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdTADVmean = np.zeros(np.shape(lat))
    unpZmean = np.zeros(np.shape(lat))
    ptdZmean = np.zeros(np.shape(lat))
    unpSmean = np.zeros(np.shape(lat))
    for intLev in intLevs:
        # compute the geopotential height at intLev
        unpZlev = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        ptdZlev = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
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
        # compute the wind (speed) at intLev
        u,v = get_uvmet(unpHdl)
        spd = np.sqrt(u**2. + v**2.)
        unpSlev = wrf.interplevel(field3d=spd,
                                  vert=wrf.getvar(unpHdl,'p'),
                                  desiredlev=intLev,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
        # compute temperature gradients at intLev
        unpDTDX, unpDTDY = get_wrf_grad(unpHdl, unpTlev)
        ptdDTDX, ptdDTDY = get_wrf_grad(ptdHdl, ptdTlev)
        # compute geopotential height gradients at intLev
        unpDZDX, unpDZDY = get_wrf_grad(unpHdl, unpZlev)
        ptdDZDX, ptdDZDY = get_wrf_grad(ptdHdl, ptdZlev)
        # compute geostrophic wind components
        unpUGEO = np.multiply(-g * f**-1., unpDZDY)
        unpVGEO = np.multiply(g * f**-1., unpDZDX)
        ptdUGEO = np.multiply(-g * f**-1., ptdDZDY)
        ptdVGEO = np.multiply(g * f**-1., ptdDZDX)
        # compute temperature advection by the geostrophic wind at intLev
        unpTADVlev = np.multiply(-unpUGEO, unpDTDX) + np.multiply(-unpVGEO, unpDTDY)
        ptdTADVlev = np.multiply(-ptdUGEO, ptdDTDX) + np.multiply(-ptdVGEO, ptdDTDY)
        # add selected terms to 2d mean fields
        unpTADVmean = unpTADVmean + unpTADVlev
        ptdTADVmean = ptdTADVmean + ptdTADVlev
        unpZmean = unpZmean + unpZlev
        ptdZmean = ptdZmean + ptdZlev
        unpSmean = unpSmean + unpSlev
    # divide by number of levels to produce mean-values
    unpTADVmean = unpTADVmean / np.size(intLevs)
    ptdTADVmean = ptdTADVmean / np.size(intLevs)
    unpZmean = unpZmean / np.size(intLevs)
    ptdZmean = ptdZmean / np.size(intLevs)
    unpSmean = unpSmean / np.size(intLevs)
    # generate plan-section plot
    spdrng = np.arange(24.,100.1,8.)
    hgtrng = np.arange(-100., 100.1, 10.).astype('float16')
    hgtrng = hgtrng[np.where(hgtrng!=0.)]
    shdrng=1.0E-04*np.arange(-16.,16.01,2.).astype('float16')
    mask = np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)]=False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSmean, ptdZmean-unpZmean],
                                            contIntervalList=[spdrng, hgtrng], 
                                            contColorList=['green', 'black'],
                                            contLineThicknessList=[1.5, 1.5],
                                            shadVariable=ptdTADVmean-unpTADVmean,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=0.7,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # restrict plot to subregion around cross-section
    offsetLat = 0.
    offsetLon = 0.
    lat1 = np.max([np.min([latMin, latMax]) - offsetLat, np.min(lat)])
    lat2 = np.min([np.max([latMin, latMax]) + offsetLat, np.max(lat)])
    lon1 = np.max([np.min([lonMin, lonMax]) - offsetLon, np.min(lon)])
    lon2 = np.min([np.max([lonMin, lonMax]) + offsetLon, np.max(lon)])
    ax.set_extent([lon1, lon2, lat1, lat2],crs=plotProj)
    # add a title
    ax.set_title('')
    # add contour labels to spd
    ax.clabel(cons[0],levels=spdrng[::2])
    return ax

def generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, figureName):
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
    # compute omega
    unpW = wrf.getvar(unpHdl,'omega')
    ptdW = wrf.getvar(ptdHdl,'omega')
    # compute potential vorticity
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    # compute (unperturbed) potential temperature
    unpThta = get_wrf_th(unpHdl)
    # compute (unperturbed) wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed speed, omega and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdW_int = interpolate_sigma_levels(ptdW, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # generate cross-section plot
    xSectShadInterval=np.arange(-2.5, 2.51, 0.25)
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
                                 xSectShadVariable=ptdW_int-unpW,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, (33.0, 171.0, 69.0, 255.0, unpHdl, ptdHdl, 40000., 70000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')
#
# begin
#
if __name__ == "__main__":
    # define unperturbed and most-intense (positive) perturbed file subdirectories
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2019, 11, 25, 12)
    
    # FIG Ya: most-intense simulation 0-hr cross section of perturbation omega
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 64.0
    lonBeg = -179.0
    latEnd = 40.0
    lonEnd = -151.5
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGY_panel_A')
    
    # FIG Yb: most-intense simulation 3-hr cross section of perturbation omega
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 64.0
    lonBeg = -178.0
    latEnd = 40.0
    lonEnd = -150.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGY_panel_B')
    
    # FIG Yc: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.0
    lonBeg = -174.0
    latEnd = 38.0
    lonEnd = -147.5
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGY_panel_C')
    
    # FIG Yd: most-intense simulation 9-hr cross section of perturbation omega
    fcstHr = 9
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.0
    lonBeg = -168.0
    latEnd = 38.0
    lonEnd = -145.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGY_panel_D')
#
# end
#
