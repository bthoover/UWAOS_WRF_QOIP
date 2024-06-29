# Figure X: Cross-section of perturbation PV, with plan-section perturbation 450 hPa PV and omega
# and mean wind speed for
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
# define function for plan-section plot
def right_panel(ax, payloadTuple):
    # expand payloadTuple into cross-section bounds, unpHdl and ptdHdl, and interpolation level
    latMin = payloadTuple[0]
    lonMin = payloadTuple[1] if payloadTuple[1] > 0. else payloadTuple[1] + 360.  # fix longitude convention
    latMax = payloadTuple[2]
    lonMax = payloadTuple[3] if payloadTuple[3] > 0. else payloadTuple[3] + 360.  # fix longitude convention
    unpHdl = payloadTuple[4]
    ptdHdl = payloadTuple[5]
    intLev = payloadTuple[6]
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
    # compute unperturbed omega
    w = wrf.getvar(unpHdl,'omega')
    # interpolate omega to intLev
    unpOmegaLev = wrf.interplevel(field3d=w,
                            vert=wrf.getvar(unpHdl,'p'),
                            desiredlev=intLev,
                            missing=np.nan,
                            squeeze=True,
                            meta=False)
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
    shdrng = np.arange(-4.,4.01,0.5).astype('float16')
    mask = np.ones(np.shape(shdrng),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    spdrng=np.arange(36., 100.1, 8.)
    omgrng=np.arange(0.5,3.01,0.25)
    slprng=np.arange(1004., 1024.1, 4.)
    #
    shd=ax.contourf(lon, lat, ptdPvorLev-unpPvorLev, levels=shdrng[mask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    ax.contour(lon, lat, unpOmegaLev, levels=omgrng, colors='black', transform=plotProj, linewidths=2.0)
    con=ax.contour(lon, lat, unpSpdLev, levels=spdrng, colors='green', transform=plotProj, linewidths=1.0)
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
    ax.clabel(con,levels=spdrng)
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
    # restrict plot to subregion around cross-section
    offsetLat = 0.
    offsetLon = 0.
    lat1 = np.max([np.min([latMin, latMax]) - offsetLat, np.min(lat)])
    lat2 = np.min([np.max([latMin, latMax]) + offsetLat, np.max(lat)])
    lon1 = np.max([np.min([lonMin, lonMax]) - offsetLon, np.min(lon)])
    lon2 = np.min([np.max([lonMin, lonMax]) + offsetLon, np.max(lon)])
    ax.set_extent([lon1, lon2, lat1, lat2],crs=plotProj)
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
                                 planSectPlotTuple=(right_panel, (33.0, 176.0, 69.0, 220.0, unpHdl, ptdHdl, 45000.)),
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
    
    # FIG Xa: most-intense simulation 0-hr cross section of perturbation PV
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
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGX_panel_A')
    
    # FIG Xb: most-intense simulation 3-hr cross section of perturbation PV
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
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGX_panel_B')
    
    # FIG Xc: most-intense simulation 6-hr cross section of perturbation PV
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
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGX_panel_C')
    
    # FIG Xd: most-intense simulation 9-hr cross section of perturbation PV
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
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGX_panel_D')
#
# end
#
