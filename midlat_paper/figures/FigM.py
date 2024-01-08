# Figure M: Cross-section of perturbation temperature and plan-section plot of 850-500 hPa thickness perturbation for
# a) initial condition of least-intense simulation
# b) initial condition of most-intense simulation
# c) 12-hr forecast of least-intense simulation
# d) 12-hr forecast of most-intense simulation
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_tk
from analysis_dependencies import get_wrf_th
from analysis_dependencies import get_wrf_slp
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

# define function for plan-section plot: 250 hPa geopotential height and wind-speed
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
    # compute unperturbed sea-level pressure
    unpSlp = get_wrf_slp(unpHdl)
    # compute unperturbed 850-500 hPa thickness
    unpZ850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              meta=False)
    unpZ500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              meta=False)
    unpThk = unpZ500 - unpZ850
    # loop through intLevs and compute wind speed and temperature
    unpTmean = np.zeros(np.shape(lat))  # using any 2d field as a guide for dimensions
    ptdTmean = np.zeros(np.shape(lat))
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
        # add temperature and wind terms to 2d mean fields
        unpTmean = unpTmean + unpTlev
        ptdTmean = ptdTmean + ptdTlev
    # divide by number of levels to produce mean-values
    unpTmean = unpTmean / np.size(intLevs)
    ptdTmean = ptdTmean / np.size(intLevs)
    # define shading variable
    shdVar = ptdTmean - unpTmean
    # generate plan-section plot
    slprng = np.arange(900., 1040.1, 4.)
    thkrng = np.arange(3700.,4500.1,50.)
    shdRange = np.arange(-2.5, 2.51, 0.25).astype('float16')
    mask = np.ones(np.shape(shdRange),dtype='bool')
    mask[np.where(shdRange==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp, unpThk],
                                            contIntervalList=[slprng, thkrng],
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[1.5, 1.5],
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
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    return ax

def generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, lineColor, figureName):
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
    # compute unperturbed wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # compute temperature
    unpT = get_wrf_tk(unpHdl)
    ptdT = get_wrf_tk(ptdHdl)
    # interpolate perturbed temperature and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    ptdT_int = interpolate_sigma_levels(ptdT, p, ps, pt, s, unpHdl)
    # generate cross-section plot
    xSectShadInterval=np.arange(-5., 5.01, 0.5)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(270.,450.,4.)
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 50000., 85000.)),
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
    # define unperturbed and perturbed simulation subdirectories
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
    dtInit = datetime.datetime(2020, 3, 6, 12)

    # Fig Ma: initial condition perturbation of least-intense simulation
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # CROSS-SHEAR THROUGH UPSTREAM 850-500 hPa THICKNESS PERTUBATION
    latBeg=55.0
    lonBeg=-80.
    latEnd=37.0
    lonEnd=-50.
    generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                                latEnd, lonEnd, 'black', 'FIGM_panel_A')
    # Fig Mb: initial condition perturbation of most-intense simulation
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # CROSS-SHEAR THROUGH UPSTREAM 850-500 hPa THICKNESS PERTUBATION
    latBeg=55.0
    lonBeg=-80.
    latEnd=37.0
    lonEnd=-50.
    generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                                latEnd, lonEnd, 'black', 'FIGM_panel_B')
    # Fig Mc: 12-hr forecast perturbation of least-intense simulatin
    fcstHr = 12
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # CROSS-SHEAR THROUGH UPSTREAM 850-500 hPa THICKNESS PERTUBATION
    latBeg=50.0
    lonBeg=-83.5
    latEnd=40.0
    lonEnd=-46.5
    generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                                latEnd, lonEnd, 'black', 'FIGM_panel_C')
    # Fig Md: 12-hr forecast perturbation of most-intense simulatin
    fcstHr = 12
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # CROSS-SHEAR THROUGH UPSTREAM 850-500 hPa THICKNESS PERTUBATION
    latBeg=50.0
    lonBeg=-83.5
    latEnd=40.0
    lonEnd=-46.5
    generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                                latEnd, lonEnd, 'black', 'FIGM_panel_D')
#
# end
#
