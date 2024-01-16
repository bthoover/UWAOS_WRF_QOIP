# Figure K: Cross-section of perturbation potential vorticity, with plan-section perturbation 500 hPa potential
#           vorticity and basic-state PV, and 500 hPa wind speed for
# a) most-intense simulation 0-hr forecast
# b) most-intense simulation 3-hr forecast
# c) most-intense simulation 6-hr forecast
# d) most-intense simulation 9-hr forecast
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
    # expand payloadTuple into unpHdl and ptdHdl, and interpolation level
    unpHdl = payloadTuple[0]
    ptdHdl = payloadTuple[1]
    intLev = payloadTuple[2]
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # compute unperturbed wind
    u,v = get_uvmet(unpHdl)
    # compute potential vorticity
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    # interpolate potential vorticity to intLev
    unpPvorLev = wrf.interplevel(field3d=unpPvor,
                                 vert=wrf.getvar(unpHdl,'p'),
                                 desiredlev=intLev,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    ptdPvorLev = wrf.interplevel(field3d=ptdPvor,
                                 vert=wrf.getvar(ptdHdl,'p'),
                                 desiredlev=intLev,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    # interpolate wind compontents to 300 hPa surface
    unpUwdLev = wrf.interplevel(field3d=u,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=intLev,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpVwdLev = wrf.interplevel(field3d=v,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=intLev,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute wind speed on intLev
    unpSpdLev = np.sqrt(unpUwdLev**2. + unpVwdLev**2.)
    #
    dynrng = np.arange(-2., 10.1, 0.5)
    #
    shdrng = np.arange(-4.,4.01,0.5).astype('float16')
    negMask = np.ones(np.shape(shdrng),dtype='bool')
    negMask[np.where(shdrng<=0.)] = False
    posMask = np.ones(np.shape(shdrng),dtype='bool')
    posMask[np.where(shdrng>=0.)] = False
    spdrng=[36., 54., 72., 90.]
    #
    shd=ax.contourf(lon, lat, unpPvorLev, levels=dynrng, cmap='gray', vmin=np.min(dynrng), vmax=np.max(dynrng), transform=plotProj)
    ax.contourf(lon, lat, ptdPvorLev-unpPvorLev, levels=shdrng[posMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    ax.contourf(lon, lat, ptdPvorLev-unpPvorLev, levels=shdrng[negMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    con1=ax.contour(lon, lat, unpSpdLev, levels=spdrng, colors='black', transform=plotProj, linewidths=3.)
    con2=ax.contour(lon, lat, unpSpdLev, levels=spdrng, colors='#ffd500', transform=plotProj, linewidths=1.)
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
    ax.clabel(con2,levels=spdrng)
    # define lat/lon lines
    latLines = np.arange(-90., 90., 5.)
    lonLines = np.arange(-180., 180. ,5.)
    gl = ax.gridlines(crs=plotProj, draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lonLines)
    gl.ylocator = mticker.FixedLocator(latLines)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'alpha' : 0.}
    gl.ylabel_style = {'size' : 9, 'color' : 'gray'}
    plt.colorbar(mappable=shd, ax=ax)
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
                                 xSectShadVariable=ptdPvor_int-unpPvor,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 50000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=['lime']
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
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2020, 3, 6, 12)
    
    # FIG Ka: most-intense simulation 0-hr cross section of perturbation PV
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 50.0
    lonBeg = -81.0
    latEnd = 27.0
    lonEnd = -88.5
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGK_panel_A')
    
    # FIG Kb: most-intense simulation 3-hr cross section of perturbation PV
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 50.0
    lonBeg = -81.0
    latEnd = 27.0
    lonEnd = -84.5
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGK_panel_B')
    
    # FIG Kc: most-intense simulation 6-hr cross section of perturbation PV
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 50.0
    lonBeg = -85.0
    latEnd = 27.0
    lonEnd = -80.5
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGK_panel_C')
    
    # FIG Kd: most-intense simulation 9-hr cross section of perturbation PV
    fcstHr = 9
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 50.0
    lonBeg = -87.0
    latEnd = 27.0
    lonEnd = -78.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGK_panel_D')
#
# end
#
