#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import get_wrf_rh
from analysis_dependencies import get_wrf_tk
from analysis_dependencies import get_wrf_th
from analysis_dependencies import get_wrf_ss
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import get_wrf_grad
from analysis_dependencies import get_xsect
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import compute_inverse_laplacian
from analysis_dependencies import plan_section_plot
from analysis_dependencies import cross_section_plot
from analysis_dependencies import gen_time_avg
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
from glob import glob
from scipy.interpolate import interp1d


# In[2]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
d10Dir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/10day_files/'
dtInit = datetime.datetime(2020, 3, 6, 12)
dtAvgBeg = datetime.datetime(2020, 3, 2, 12)
dtAvgEnd = datetime.datetime(2020, 3, 12, 12)


# In[3]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
for fcstHr in [0]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define prior datetime stamp, if fcstHr==0, set to 0
    if fcstHr == 0:
        dtPrior = dtFcst
        dtPriorStr = dtFcstStr
    else:
        dtPrior = dtFcst - datetime.timedelta(hours=1)
        dtPriorStr = datetime.datetime.strftime(dtPrior,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePrior = unpDir + 'wrfout_d01_' + dtPriorStr
    wrfHdl = Dataset(fileFcst)
    wrfHdlPrior = Dataset(filePrior)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(wrfHdl.variables['XLAT']).squeeze()
    lon = np.asarray(wrfHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(wrfHdl)
    plotProj = ccrs.PlateCarree()
    # extract pressure for interpolation
    prs = np.asarray(wrf.getvar(wrfHdl,'p')).squeeze()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(wrfHdl)).squeeze()
    # extract accumulated non-convective precipitation, compute hourly
    hrPrecipNC = np.asarray(wrfHdl.variables['RAINNC']).squeeze() - np.asarray(wrfHdlPrior.variables['RAINNC']).squeeze()
    # extract accumulated convective precipitation, compute hourly
    hrPrecipC = np.asarray(wrfHdl.variables['RAINC']).squeeze() - np.asarray(wrfHdlPrior.variables['RAINC']).squeeze()
    # interpolate heights to 850 and 500 hPa
    z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                           vert=prs,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # compute 850-500 thickness
    thk = z500 - z850
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp,thk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=hrPrecipC+hrPrecipNC,
                                            shadInterval=np.arange(1.,12.1,0.5),
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='cubehelix_r',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed sea level pressure, 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[4]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, precipitation, and 
# SLP-perturbation
for fcstHr in [24]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define prior datetime stamp, if fcstHr==0, set to 0
    if fcstHr == 0:
        dtPrior = dtFcst
        dtPriorStr = dtFcstStr
    else:
        dtPrior = dtFcst - datetime.timedelta(hours=1)
        dtPriorStr = datetime.datetime.strftime(dtPrior,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = posDir + 'wrfout_d01_' + dtFcstStr
    filePrior = posDir + 'wrfout_d01_' + dtPriorStr
    unpHdl = Dataset(fileFcst)
    ptdHdl = Dataset(filePert)
    ptdHdlPrior = Dataset(filePrior)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract pressure for interpolation
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    # extract sea-level pressure
    slp0 = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    slp = np.asarray(get_wrf_slp(ptdHdl)).squeeze()
    # extract accumulated non-convective precipitation, compute hourly
    hrPrecipNC = np.asarray(ptdHdl.variables['RAINNC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINNC']).squeeze()
    # extract accumulated convective precipitation, compute hourly
    hrPrecipC = np.asarray(ptdHdl.variables['RAINC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINC']).squeeze()
    # interpolate heights to 850 and 500 hPa
    z850 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=prs,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    z500 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # compute 850-500 thickness
    thk = z500 - z850
    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(30,7), subplot_kw={'projection' : datProj})

    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)
    shdrng=np.arange(-30.,30.1,3.)
    prcrng=np.arange(1.,12.1,0.5)
    mask=np.ones(np.size(shdrng),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    ax=axs[0]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp,thk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=hrPrecipNC+hrPrecipC,
                                            shadInterval=prcrng,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='cubehelix_r',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed sea level pressure, 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    
    ax=axs[1]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp,thk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=slp-slp0,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=0.5,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed sea level pressure, 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[5]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and 
# SLP-perturbation
for fcstHr in [24]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define prior datetime stamp, if fcstHr==0, set to 0
    if fcstHr == 0:
        dtPrior = dtFcst
        dtPriorStr = dtFcstStr
    else:
        dtPrior = dtFcst - datetime.timedelta(hours=1)
        dtPriorStr = datetime.datetime.strftime(dtPrior,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = posDir + 'wrfout_d01_' + dtFcstStr
    filePrior = posDir + 'wrfout_d01_' + dtPriorStr
    unpHdl = Dataset(fileFcst)
    ptdHdl = Dataset(filePert)
    ptdHdlPrior = Dataset(filePrior)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract pressure for interpolation
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    # extract sea-level pressure
    slp0 = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    slp = np.asarray(get_wrf_slp(ptdHdl)).squeeze()
    # extract accumulated non-convective precipitation, compute hourly
    hrPrecipNC = np.asarray(ptdHdl.variables['RAINNC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINNC']).squeeze()
    # extract accumulated convective precipitation, compute hourly
    hrPrecipC = np.asarray(ptdHdl.variables['RAINC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINC']).squeeze()
    # interpolate heights to 850 and 500 hPa
    uz850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    uz500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz850 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=prs,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz500 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # compute 850-500 thickness
    uthk = uz500 - uz850
    pthk = pz500 - pz850
    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(30,7), subplot_kw={'projection' : datProj})

    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)
    shdrng=np.arange(-30.,30.1,3.)
    prcrng=np.arange(1.,12.1,0.5)
    mask=np.ones(np.size(shdrng),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    ax=axs[0]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp0,uthk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap=None,
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed sea level pressure, 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    
    ax=axs[1]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp,pthk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=slp-slp0,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=0.5,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed sea level pressure, 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[6]:


# For a selected forecast time, plot the 500 hPa geopotential heights, perturbation vorticity and heights
for fcstHr in [24]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(fileFcst)
    ptdHdl = Dataset(filePert)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # interpolate vorticity and heights to 500 hPa
    uz500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz500 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=wrf.getvar(ptdHdl,'p'),
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    slp = get_wrf_slp(ptdHdl)
    slp0 = get_wrf_slp(unpHdl)
    # Plot unperturbed 500 hPa height, height perturbation, vorticity perturbation
    slprng=np.arange(900.,1012.1,4.)
    hgtrng=np.arange(4800.,6200.1,60.)
    zrng=np.arange(-80.,80.1,4.)
    
    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(30,7), subplot_kw={'projection' : datProj})
    
    ax=axs[0]
    
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[uz500,slp0],
                                            contIntervalList=[hgtrng,slprng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.5,1.],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=0.5,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 500 hPa geopotential height'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[1]
    
    mask=np.ones(np.size(zrng),dtype='bool')
    mask[np.where(zrng==0.)] = False
    
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[pz500,slp],
                                            contIntervalList=[hgtrng,slprng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.5,1.],
                                            shadVariable=pz500-uz500,
                                            shadInterval=zrng[mask],
                                            shadAlpha=0.5,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed 500 hPa geopotential height'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[7]:


# For a selected forecast time, plot the 250/350/450 hPa geopotential heights, wind speed, and perturbation wind speed
for fcstHr in [0]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(fileFcst)
    ptdHdl = Dataset(filePert)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # define wind and wind speed
    u, v = get_uvmet(unpHdl)
    spd = np.sqrt(u**2. + v**2.)
    unpWspd250 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(unpHdl,'p'),
                                     desiredlev=25000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    unpWspd350 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(unpHdl,'p'),
                                     desiredlev=35000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    unpWspd450 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(unpHdl,'p'),
                                     desiredlev=45000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    u, v = get_uvmet(ptdHdl)
    spd = np.sqrt(u**2. + v**2.)
    ptdWspd250 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(ptdHdl,'p'),
                                     desiredlev=25000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    ptdWspd350 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(ptdHdl,'p'),
                                     desiredlev=35000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    ptdWspd450 = wrf.interplevel(field3d=spd,
                                     vert=wrf.getvar(ptdHdl,'p'),
                                     desiredlev=45000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    
    # define heights
    uz250 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=25000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    uz350 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=35000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    uz450 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=45000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz250 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=wrf.getvar(ptdHdl,'p'),
                           desiredlev=25000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz350 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=wrf.getvar(ptdHdl,'p'),
                           desiredlev=35000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    pz450 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                           vert=wrf.getvar(ptdHdl,'p'),
                           desiredlev=45000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # plan-section figure: perturbation 300 hPa wind speed and winds

    shdrng = np.arange(-3.5,3.51,0.25)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(5500.,12000.1,90.)
    spdrng = np.arange(40.,90.1,10.)
    shdalpha = 0.6

    fig, axs = plt.subplots(ncols=2,nrows=3,figsize=(30,27), subplot_kw={'projection' : datProj})

    ax=axs[0][0]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[uz250, unpWspd250],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 250 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[0][1]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[pz250, ptdWspd250],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=ptdWspd250-unpWspd250,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed 250 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[1][0]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[uz350, unpWspd350],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 350 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[1][1]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[pz350, ptdWspd350],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=ptdWspd350-unpWspd350,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed 350 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[2][0]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[uz450, unpWspd450],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 450 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    ax=axs[2][1]

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[pz450, ptdWspd450],
                                            contIntervalList=[hgtrng, spdrng], 
                                            contColorList=['black', '#35821b'],
                                            contLineThicknessList=[1.0, 2.0],
                                            shadVariable=ptdWspd450-unpWspd450,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=shdalpha,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            vectorScale=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed 450 hPa geopotential height and wind speed'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/pos_f00_initSpdPert.png',bbox_inches='tight',facecolor='white')


# In[8]:


# Plot the 450 hPa geopotential height perturbations at 0, 6, and 12 hrs with a cross-section line, and contrast
# with cross-sections taken hourly along trough axis
#
# ALONG TROUGH AXIS CROSS SECTIONS (roughly tracks neg. pert. Tadv, appears to show PV intrusion)
sampleHrs=[0., 3., 6., 9., 12.]
sampleLatBegList=[45., 45., 45., 45., 45.]
sampleLonBegList=[-83., -81.5, -80., -78., -77.5]
sampleLatEndList=[25., 25., 25., 25., 25.]
sampleLonEndList=[-88., -85., -82.5, -81.5, -79.5]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(13.))

fcstHr = 0
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
wrfHdl = Dataset(unpDir + 'wrfout_d01_' + dtFcstStr)
datProj = gen_cartopy_proj(wrfHdl)
plotProj = ccrs.PlateCarree()

fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

for fcstHr in [0,6,12]:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    posFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    negFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    posHdl = Dataset(posFileFcst)
    negHdl = Dataset(negFileFcst)
    #extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpHgt450 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    prs = np.asarray(wrf.getvar(posHdl,'p')).squeeze()
    posHgt450 = wrf.interplevel(field3d=wrf.getvar(posHdl,'z'),
                                vert=prs,
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    prs = np.asarray(wrf.getvar(negHdl,'p')).squeeze()
    negHgt450 = wrf.interplevel(field3d=wrf.getvar(negHdl,'z'),
                                vert=prs,
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    hgtrng=np.arange(-120.,120.1,12.)
    hmask=np.ones(np.size(hgtrng),dtype='bool')
    hmask[np.where(hgtrng==0.)]=False
    
    ax=axs
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[negHgt450-unpHgt450],
                                            contIntervalList=[hgtrng[hmask]], 
                                            contColorList=['black'],
                                            contLineThicknessList=[1.0],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=10,
                                            vecColor='#35821b',
                                            vectorScale=30,
                                            figax=ax)
    ax.set_title('(0/6/12 hrs) weakening perturbed Hght 450 hPa')
# presentation cross-section values
# latBeg = 47.5
# lonBeg = -98.5
# latEnd = 28.5
# lonEnd = -70.
# modified cross-section values
# latBeg = 48.0
# lonBeg = -98.5
# latEnd = 27.0
# lonEnd = -70.
# final(?) cross-section values
latBeg=48.0
lonBeg=-94.
latEnd=27.0
lonEnd=-74.
# collect the latitude and longitude values along cross-section from get_xsect(), for some
# arbitrary cross-section data (we aren't using the data so it doesn't really matter)
xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), latBeg, lonBeg, latEnd, lonEnd)
# plot end-points of cross section
axs.plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs.plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
# along cross-section, plot segments defined by latList, lonList
for i in range(len(latList)-1):
    axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')
for j in range(len(hourlyLatBegList)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegList[j], hourlyLonBegList[j],
                                        hourlyLatEndList[j], hourlyLonEndList[j])
    axs.plot(lonList[0],latList[0],'o',transform=plotProj,color='green',alpha=0.5)
    axs.plot(lonList[-1],latList[-1],'o',transform=plotProj,color='green',alpha=0.5)
    for i in range(len(latList)-1):
        axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='green',alpha=0.5)
fig.savefig('fig_tank/cross_section_plan_hgtPert.png',bbox_inches='tight',facecolor='white')


# In[10]:


# Plot the 450 hPa geopotential height perturbations at 0, 6, and 12 hrs with a cross-section line, and contrast
# with cross-sections taken hourly along trough axis
#
# ALONG TROUGH AXIS CROSS SECTIONS (roughly tracks neg. pert. Tadv, appears to show PV intrusion)
sampleHrs=[0., 3., 6., 9., 12.]
sampleLatBegList=[45., 45., 45., 45., 45.]
sampleLonBegList=[-83., -81.5, -80., -78., -77.5]
sampleLatEndList=[25., 25., 25., 25., 25.]
sampleLonEndList=[-88., -85., -82.5, -81.5, -79.5]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(13.))

fcstHr = 0
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
wrfHdl = Dataset(unpDir + 'wrfout_d01_' + dtFcstStr)
datProj = gen_cartopy_proj(wrfHdl)
plotProj = ccrs.PlateCarree()

fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

for fcstHr in [12]:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    posFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    negFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    posHdl = Dataset(posFileFcst)
    negHdl = Dataset(negFileFcst)
    #extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpSlp = get_wrf_slp(unpHdl)
    posSlp = get_wrf_slp(posHdl)
    negSlp = get_wrf_slp(negHdl)
    unpH850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpH500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    slprng=np.arange(-18.,18.1,2.)
    
    ax=axs
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[negSlp-unpSlp,unpH500-unpH850],
                                            contIntervalList=[slprng,np.arange(3700.,4500.1,50.)],
                                            contColorList=['black','red'],
                                            contLineThicknessList=[1.0,0.75],
                                            shadVariable=None,
                                            shadInterval=None,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='seismic',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=10,
                                            vecColor='#35821b',
                                            vectorScale=30,
                                            figax=ax)
    ax.set_title('(0/6/12 hrs) intensifying perturbed Hght 450 hPa')
# presentation cross-section values
# latBeg = 47.5
# lonBeg = -98.5
# latEnd = 28.5
# lonEnd = -70.
# modified cross-section values
# latBeg = 48.0
# lonBeg = -98.5
# latEnd = 27.0
# lonEnd = -70.
# final(?) cross-section values
latBeg=48.0
lonBeg=-94.
latEnd=27.0
lonEnd=-74.
# collect the latitude and longitude values along cross-section from get_xsect(), for some
# arbitrary cross-section data (we aren't using the data so it doesn't really matter)
xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), latBeg, lonBeg, latEnd, lonEnd)
# plot end-points of cross section
axs.plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs.plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
# along cross-section, plot segments defined by latList, lonList
for i in range(len(latList)-1):
    axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')
for j in range(len(hourlyLatBegList)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegList[j], hourlyLonBegList[j],
                                        hourlyLatEndList[j], hourlyLonEndList[j])
    axs.plot(lonList[0],latList[0],'o',transform=plotProj,color='green',alpha=0.5)
    axs.plot(lonList[-1],latList[-1],'o',transform=plotProj,color='green',alpha=0.5)
    for i in range(len(latList)-1):
        axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='green',alpha=0.5)
fig.savefig('fig_tank/cross_section_plan_hgtPert.png',bbox_inches='tight',facecolor='white')


# In[11]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential temperature (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in range(13):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # compute potential temperature
    unpThta = get_wrf_th(unpHdl)
    ptdThta = get_wrf_th(ptdHdl)
    # compute (unperturbed) wind and wind-speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed pvor and thta to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    ptdThta_int = interpolate_sigma_levels(ptdThta, p, ps, pt, s, unpHdl)
    
    # define function to produce plan-section plot: SLP and 850-500 hPa thickness
    def right_panel(ax, wrfHdl):
        # get latitude and longitude
        lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
        lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
        # fix longitude to 0 to 360 degree format
        fix = np.where(lon < 0.)
        lon[fix] = lon[fix] + 360.
        # get data and plot projections
        datProj = gen_cartopy_proj(wrfHdl)
        plotProj = ccrs.PlateCarree()
        # interpolate heights to 850 and 500 hPa
        z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=85000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=50000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # compute 850-500 thickness
        thk = z500 - z850
        # get SLP
        slp = get_wrf_slp(wrfHdl)
        # generate figure on ax
        slprng=np.arange(900.,1030.1,4.)
        thkrng=np.arange(3700.,4500.1,50.)
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[slp,thk],
                                                contIntervalList=[slprng,thkrng], 
                                                contColorList=['black','#b06407'],
                                                contLineThicknessList=[0.75,0.75],
                                                shadVariable=None,
                                                shadInterval=None,
                                                shadAlpha=None,
                                                datProj=datProj,
                                                plotProj=plotProj,
                                                shadCmap=None,
                                                uVecVariable=None,
                                                vVecVariable=None,
                                                vectorThinning=None,
                                                vecColor=None,
                                                figax=ax)
        # add a title
        ax.set_title(dtFcstStr + 'unperturbed sea level pressure, 850-500 hPa thickness')
        # add contour labels to slp
        ax.clabel(cons[0],levels=slprng[::2])
        # return ax
        return ax
    
    # generate cross-section plot
    xSectShadInterval = np.arange(-10., 10.1, 1.)
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
                                 xSectShadVariable=ptdThta_int-unpThta,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, unpHdl),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed potential temperature'.format(fcstHr),
                                 xLineColorList=['black']
                                 )
    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[12]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in range(13):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # compute (unperturbed) potential temperature, winds, and wind-speed
    unpThta = get_wrf_th(unpHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed pvor onto unperturbed sigma levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # define function to produce plan-section plot: SLP and 850-500 hPa thickness
    def right_panel(ax, wrfHdl):
        # get latitude and longitude
        lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
        lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
        # fix longitude to 0 to 360 degree format
        fix = np.where(lon < 0.)
        lon[fix] = lon[fix] + 360.
        # get data and plot projections
        datProj = gen_cartopy_proj(wrfHdl)
        plotProj = ccrs.PlateCarree()
        # interpolate heights to 850 and 500 hPa
        z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=85000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=50000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # compute 850-500 thickness
        thk = z500 - z850
        # get SLP
        slp = get_wrf_slp(wrfHdl)
        # generate figure on ax
        slprng=np.arange(900.,1030.1,4.)
        thkrng=np.arange(3700.,4500.1,50.)
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[slp,thk],
                                                contIntervalList=[slprng,thkrng], 
                                                contColorList=['black','#b06407'],
                                                contLineThicknessList=[0.75,0.75],
                                                shadVariable=None,
                                                shadInterval=None,
                                                shadAlpha=None,
                                                datProj=datProj,
                                                plotProj=plotProj,
                                                shadCmap=None,
                                                uVecVariable=None,
                                                vVecVariable=None,
                                                vectorThinning=None,
                                                vecColor=None,
                                                figax=ax)
        # add a title
        ax.set_title(dtFcstStr + 'unperturbed sea level pressure, 850-500 hPa thickness')
        # add contour labels to slp
        ax.clabel(cons[0],levels=slprng[::2])
        # return ax
        return ax
    
    
    # generate cross-section plot
    xSectShadInterval = np.arange(-4., 4.01, 0.5)
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
                                 planSectPlotTuple=(right_panel, unpHdl),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed potential vorticity'.format(fcstHr),
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[13]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# geop. height (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in range(13):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # compute potential temperature
    unpThta = get_wrf_th(unpHdl)
    ptdThta = get_wrf_th(ptdHdl)
    # compute geopotential height
    unpHgt = wrf.getvar(unpHdl,'z')
    ptdHgt = wrf.getvar(ptdHdl,'z')
    # compute (unperturbed) wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed pvor and hgt to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdHgt_int = interpolate_sigma_levels(ptdHgt, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # define function to produce plan-section plot: SLP and 850-500 hPa thickness
    def right_panel(ax, wrfHdl):
        # get latitude and longitude
        lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
        lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
        # fix longitude to 0 to 360 degree format
        fix = np.where(lon < 0.)
        lon[fix] = lon[fix] + 360.
        # get data and plot projections
        datProj = gen_cartopy_proj(wrfHdl)
        plotProj = ccrs.PlateCarree()
        # interpolate heights to 850 and 500 hPa
        z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=85000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=50000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # compute 850-500 thickness
        thk = z500 - z850
        # get SLP
        slp = get_wrf_slp(wrfHdl)
        # generate figure on ax
        slprng=np.arange(900.,1030.1,4.)
        thkrng=np.arange(3700.,4500.1,50.)
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[slp,thk],
                                                contIntervalList=[slprng,thkrng], 
                                                contColorList=['black','#b06407'],
                                                contLineThicknessList=[0.75,0.75],
                                                shadVariable=None,
                                                shadInterval=None,
                                                shadAlpha=None,
                                                datProj=datProj,
                                                plotProj=plotProj,
                                                shadCmap=None,
                                                uVecVariable=None,
                                                vVecVariable=None,
                                                vectorThinning=None,
                                                vecColor=None,
                                                figax=ax)
        # add a title
        ax.set_title(dtFcstStr + 'unperturbed sea level pressure, 850-500 hPa thickness')
        # add contour labels to slp
        ax.clabel(cons[0],levels=slprng[::2])
        # return ax
        return ax
    # generate cross-section plot
    xSectShadInterval=np.arange(-120., 120.1, 12.)
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
                                 xSectShadVariable=ptdHgt_int-unpHgt,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 planSectPlotTuple=(right_panel, unpHdl),
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed geopotential height'.format(fcstHr),
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[14]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in range(13):
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # compute relative vorticity
    unpVor = get_wrf_kinematic(unpHdl,'vor')
    ptdVor = get_wrf_kinematic(unpHdl,'vor')
    # compute (unperturbed) potential temperature
    unpThta = get_wrf_th(unpHdl)
    # compute (unperturbed) wind and wind speed
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    # interpolate perturbed pvor and vor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdVor_int = interpolate_sigma_levels(ptdVor, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # define function to produce plan-section plot: SLP and 850-500 hPa thickness
    def right_panel(ax, wrfHdl):
        # get latitude and longitude
        lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
        lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
        # fix longitude to 0 to 360 degree format
        fix = np.where(lon < 0.)
        lon[fix] = lon[fix] + 360.
        # get data and plot projections
        datProj = gen_cartopy_proj(wrfHdl)
        plotProj = ccrs.PlateCarree()
        # interpolate heights to 850 and 500 hPa
        z850 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=85000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        z500 = wrf.interplevel(field3d=wrf.getvar(wrfHdl,'z'),
                               vert=wrf.getvar(wrfHdl,'p'),
                               desiredlev=50000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # compute 850-500 thickness
        thk = z500 - z850
        # get SLP
        slp = get_wrf_slp(wrfHdl)
        # generate figure on ax
        slprng=np.arange(900.,1030.1,4.)
        thkrng=np.arange(3700.,4500.1,50.)
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[slp,thk],
                                                contIntervalList=[slprng,thkrng], 
                                                contColorList=['black','#b06407'],
                                                contLineThicknessList=[0.75,0.75],
                                                shadVariable=None,
                                                shadInterval=None,
                                                shadAlpha=None,
                                                datProj=datProj,
                                                plotProj=plotProj,
                                                shadCmap=None,
                                                uVecVariable=None,
                                                vVecVariable=None,
                                                vectorThinning=None,
                                                vecColor=None,
                                                figax=ax)
        # add a title
        ax.set_title(dtFcstStr + 'unperturbed sea level pressure, 850-500 hPa thickness')
        # add contour labels to slp
        ax.clabel(cons[0],levels=slprng[::2])
        # return ax
        return ax
    # generate cross-section plot
    xSectShadInterval=1.0E-06 * np.arange(-8., 8.1, 1.)
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
                                 xSectShadVariable=ptdVor_int-unpVor,
                                 xSectShadInterval=xSectShadInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 planSectPlotTuple=(right_panel, unpHdl),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed vorticity'.format(fcstHr),
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[15]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# omega (interpolated to the unperturbed sigma-levels). Plan-section plot is 250 hPa unperturbed geop. hgt and wind
# speed with perturbation temperature advection by the geostrophic wind at a chosen interpolation level (shaded)
for fcstHr in [12]:
    latBegList = [latBeg]
    lonBegList = [lonBeg]
    latEndList = [latEnd]
    lonEndList = [lonEnd]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # interpolate perturbed omega and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdW_int = interpolate_sigma_levels(ptdW, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # define function for plan-section plot: 250 hPa geopotential height and wind-speed
    def right_panel(ax, payloadTuple):
        # expand payloadTuple into unpHdl and ptdHdl, and interpolation level
        unpHdl = payloadTuple[0]
        ptdHdl = payloadTuple[1]
        intLev = payloadTuple[2]
        # define data and plot projection
        datProj = gen_cartopy_proj(unpHdl)
        plotProj = ccrs.PlateCarree()
        # extract unperturbed wind and compute speed
        u,v = get_uvmet(unpHdl)
        spd = np.sqrt(u**2. + v**2.)
        # interpolate unperturbed heights and speed to 250 hPa
        z250 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=25000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        s250 = wrf.interplevel(field3d=spd,
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=25000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # define f and g for geostrophic wind calculation
        f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)  # 2*omega*sin(phi), s^-1
        g = 9.80665  # m/s^2
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
        # generate plan-section plot
        hgtrng=np.arange(9500.,11500.1,120.)
        spdrng = np.arange(36.,100.1,8.)
        shdrng=1.0E-03*np.arange(-2.,2.1,0.2).astype('float16')
        mask = np.ones(np.shape(shdrng),dtype=bool)
        mask[np.where(shdrng==0.)]=False
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[z250,s250],
                                                contIntervalList=[hgtrng,spdrng], 
                                                contColorList=['black','green'],
                                                contLineThicknessList=[0.75,1.5],
                                                shadVariable=ptdTADVlev-unpTADVlev,
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
        # add a title
        ax.set_title('(unperturbed 250 hPa geopt. height, isotachs')
        # add contour labels to spd
        ax.clabel(cons[1],levels=spdrng[::2])
        return ax
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 55000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbation omega'.format(fcstHr),
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # highlight cross-section x-axis (lev=18) in intervals along cross-section of 12 grid-points
    fig.axes[0].plot([0.,11.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([12.,23.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([24.,35.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([36.,47.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([48.,59.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([60.,71.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([72.,83.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    fig.axes[0].plot([84.,95.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    # highlight cross-section line in orange segments (line beneath is entirely black)
    fig.axes[2].plot([lonLists[0][12],lonLists[0][23]],[latLists[0][12],latLists[0][23]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    fig.axes[2].plot([lonLists[0][36],lonLists[0][47]],[latLists[0][36],latLists[0][47]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    fig.axes[2].plot([lonLists[0][60],lonLists[0][71]],[latLists[0][60],latLists[0][71]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    fig.axes[2].plot([lonLists[0][84],lonLists[0][95]],[latLists[0][84],latLists[0][95]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[16]:


# NOTE: There may be a significant contribution to perturbation PV in the direct west-east direction, based
#       on the plots of geostrophic temperature advection and their location within the cyclonic shear. The
#       large negative perturbations to temp adv will drive downward vertical motion, drawing down the dynamic
#       tropopause (plan-section plot of perturbation potential temperature along dynamic trop?), which can
#       feed the upper front PV intrusion into the middle troposphere. Consider west-east cross sections through
#       the zone of temp adv differences.


# In[17]:


fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,6),subplot_kw={'projection':datProj})
ax = right_panel(ax, (unpHdl, ptdHdl, 55000.))
lat1=45.
lon1=-77.5
lat2=25.
lon2=-79.5
xs,latList,lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), lat1, lon1, lat2, lon2)
# plot end-points of cross section
ax.plot(lon1,lat1,'o',transform=plotProj,color='magenta')
ax.plot(lon2,lat2,'o',transform=plotProj,color='magenta')
# along cross-section, plot segments defined by latList, lonList
for i in range(len(latList)-1):
    ax.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')


# In[18]:


# ALONG TROUGH AXIS CROSS SECTIONS (roughly tracks neg. pert. Tadv, appears to show PV intrusion)
sampleHrs=[0., 3., 6., 9., 12.]
sampleLatBegList=[45., 45., 45., 45., 45.]
sampleLonBegList=[-83., -81.5, -80., -78., -77.5]
sampleLatEndList=[25., 25., 25., 25., 25.]
sampleLonEndList=[-88., -85., -82.5, -81.5, -79.5]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(13.))


# In[19]:


for fcstHr in range(13):
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,6),subplot_kw={'projection':datProj})
    lat1=hourlyLatBegList[fcstHr]
    lon1=hourlyLonBegList[fcstHr]
    lat2=hourlyLatEndList[fcstHr]
    lon2=hourlyLonEndList[fcstHr]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    xs,latList,lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), lat1, lon1, lat2, lon2)
    ax = right_panel(ax, (unpHdl, ptdHdl, 55000.))
    # plot end-points of cross section
    ax.plot(lon1,lat1,'o',transform=plotProj,color='magenta')
    ax.plot(lon2,lat2,'o',transform=plotProj,color='magenta')
    # along cross-section, plot segments defined by latList, lonList
    for i in range(len(latList)-1):
        ax.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')


# In[132]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# omega (interpolated to the unperturbed sigma-levels). Plan-section plot is 250 hPa unperturbed geop. hgt and wind
# speed with perturbation temperature advection by the geostrophic wind at a chosen interpolation level (shaded)
for fcstHr in range(13):
    latBegList = [hourlyLatBegList[fcstHr]]
    lonBegList = [hourlyLonBegList[fcstHr]]
    latEndList = [hourlyLatEndList[fcstHr]]
    lonEndList = [hourlyLonEndList[fcstHr]]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
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
    # interpolate perturbed omega and pvor to unperturbed sigma-levels
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdW_int = interpolate_sigma_levels(ptdW, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    # define function for plan-section plot: 250 hPa geopotential height and wind-speed
    def right_panel(ax, payloadTuple):
        # expand payloadTuple into unpHdl and ptdHdl, and interpolation level
        unpHdl = payloadTuple[0]
        ptdHdl = payloadTuple[1]
        intLev = payloadTuple[2]
        # define data and plot projection
        datProj = gen_cartopy_proj(unpHdl)
        plotProj = ccrs.PlateCarree()
        # extract unperturbed wind and compute speed
        u,v = get_uvmet(unpHdl)
        spd = np.sqrt(u**2. + v**2.)
        # interpolate unperturbed heights and speed to 250 hPa
        z250 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=25000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        s250 = wrf.interplevel(field3d=spd,
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=25000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # define f and g for geostrophic wind calculation
        f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)  # 2*omega*sin(phi), s^-1
        g = 9.80665  # m/s^2
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
        # generate plan-section plot
        hgtrng=np.arange(9500.,11500.1,120.)
        spdrng = np.arange(36.,100.1,8.)
        shdrng=1.0E-03*np.arange(-2.,2.1,0.2).astype('float16')
        mask = np.ones(np.shape(shdrng),dtype=bool)
        mask[np.where(shdrng==0.)]=False
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[z250,s250],
                                                contIntervalList=[hgtrng,spdrng], 
                                                contColorList=['black','green'],
                                                contLineThicknessList=[0.75,1.5],
                                                shadVariable=ptdTADVlev-unpTADVlev,
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
        # add a title
        ax.set_title('(unperturbed 250 hPa geopt. height, isotachs')
        # add contour labels to spd
        ax.clabel(cons[1],levels=spdrng[::2])
        return ax
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 55000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbation omega'.format(fcstHr),
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # highlight cross-section x-axis (lev=18) in intervals along cross-section of 12 grid-points
    #fig.axes[0].plot([0.,11.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([12.,23.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([24.,35.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([36.,47.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([48.,59.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([60.,71.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([72.,83.],[18.,18.],color='black',linewidth=4.0,alpha=1.)
    #fig.axes[0].plot([84.,95.],[18.,18.],color='orange',linewidth=4.0,alpha=1.)
    # highlight cross-section line in orange segments (line beneath is entirely black)
    #fig.axes[2].plot([lonLists[0][12],lonLists[0][23]],[latLists[0][12],latLists[0][23]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    #fig.axes[2].plot([lonLists[0][36],lonLists[0][47]],[latLists[0][36],latLists[0][47]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    #fig.axes[2].plot([lonLists[0][60],lonLists[0][71]],[latLists[0][60],latLists[0][71]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    #fig.axes[2].plot([lonLists[0][84],lonLists[0][95]],[latLists[0][84],latLists[0][95]],color='orange',linewidth=2.5,transform=plotProj,alpha=1.0)
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[122]:





# In[34]:


# For a selected forecast time, plot the unperturbed potential temperature of the 2 PVU isosurface and the
# perturbation potential temperature of the 2 PVU isosurface
for fcstHr in [3]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast files and open netCDF4 file-handles
    unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFile)
    ptdHdl = Dataset(ptdFile)
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
    # interpolate potential temperature to 2.0 PVU surface
    unpDynThta = wrf.interplevel(field3d=get_wrf_th(unpHdl),
                                 vert=unpPvor,
                                 desiredlev=2.0,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    ptdDynThta = wrf.interplevel(field3d=get_wrf_th(ptdHdl),
                                 vert=ptdPvor,
                                 desiredlev=2.0,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    #
    dynrng=np.arange(275.,425.1,5.)
    #
    shdrng=np.arange(-25.,25.1,2.5).astype('float16')
    mask=np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)]=False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[ptdDynThta-unpDynThta,ptdDynThta-unpDynThta],
                                            contIntervalList=[np.arange(-25.,-2.4,2.5),np.arange(2.5,25.1,2.5)], 
                                            contColorList=['black','white'],
                                            contLineThicknessList=[1.5,1.5],
                                            shadVariable=unpDynThta,
                                            shadInterval=dynrng,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='gist_rainbow_r',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add a title
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbation pot. temp. on 2 PVU isosurface'.format(fcstHr))
    # add contour labels to slp
    #ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[62]:


for fcstHr in [12]:
    intLev=45000.
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast files and open netCDF4 file-handles
    unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFile = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFile)
    ptdHdl = Dataset(ptdFile)
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract unperturbed wind and compute speed
    u,v = get_uvmet(unpHdl)
    spd = np.sqrt(u**2. + v**2.)
    # interpolate unperturbed heights and speed to 250 hPa
    z250 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=25000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    s250 = wrf.interplevel(field3d=spd,
                           vert=wrf.getvar(unpHdl,'p'),
                           desiredlev=25000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # define f and g for geostrophic wind calculation
    f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)  # 2*omega*sin(phi), s^-1
    g = 9.80665  # m/s^2
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
    # compute advection of PERTURBED temperature by UNPERTURBED geostrophic wind
    ptdTADVlev1 = np.multiply(-unpUGEO, ptdDTDX-unpDTDX) + np.multiply(-unpVGEO, ptdDTDY-unpDTDY)
    # compute advection of UNPERTURBED temperature by PERTURBED geostrophic wind
    ptdTADVlev2 = np.multiply(-(ptdUGEO-unpUGEO), unpDTDX) + np.multiply(-(ptdVGEO-unpVGEO), unpDTDY)
    
    ptdTADVlev3 = np.multiply(-(ptdUGEO-unpUGEO), ptdDTDX-unpDTDX) + np.multiply(-(ptdVGEO-unpVGEO), ptdDTDY-unpDTDY)
    # generate plan-section plot
    hgtrng=np.arange(9500.,11500.1,120.)
    spdrng = np.arange(36.,100.1,8.)
    shdrng=1.0E-03*np.arange(-2.,2.1,0.2).astype('float16')
    mask = np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)]=False
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[z250,s250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTADVlev-unpTADVlev,
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
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[z250,s250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTADVlev1+ptdTADVlev2+ptdTADVlev3,
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
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[z250,s250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTADVlev1,
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
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[z250,s250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTADVlev2,
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
    fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(12,8),subplot_kw={'projection':datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[z250,s250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTADVlev3,
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


# In[48]:


# For a selected forecast time, plot the unperturbed potential temperature of the 2 PVU isosurface and the
# perturbation potential temperature of the 2 PVU isosurface
for fcstHr in [0]:
    latBeg = 48.0
    lonBeg = -94.0
    latEnd = 27.0
    lonEnd = -74.0
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast files and open netCDF4 file-handles
    unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFile)
    ptdHdl = Dataset(ptdFile)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # compute potential temperature
    unpThta = get_wrf_th(unpHdl)
    # compute potential vorticity
    unpPvor = wrf.getvar(unpHdl,'pvo')
    # compute wind components
    unpUwd, unpVwd = get_uvmet(unpHdl)
    # interpolate potential temperature to 2.0 PVU surface
    unpDynThta = wrf.interplevel(field3d=unpThta,
                                 vert=unpPvor,
                                 desiredlev=2.0,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    # interpolate wind components to 335.0 K surface
    unpDynUwd = wrf.interplevel(field3d=unpUwd,
                                vert=unpThta,
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpDynVwd = wrf.interplevel(field3d=unpVwd,
                                vert=unpThta,
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute sea-level pressure
    unpSlp = get_wrf_slp(unpHdl)
    #    
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    #
    spdrng=np.arange(36.,150.1,6.)
    dynrng=np.arange(270.,375.1,5.)
    slprng=np.arange(900.,1012.1,4.)
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp,(unpDynUwd**2. + unpDynVwd**2.)**0.5],
                                            contIntervalList=[slprng,spdrng], 
                                            contColorList=['yellow','red'],
                                            contLineThicknessList=[1.0,2.0],
                                            shadVariable=unpDynThta,
                                            shadInterval=dynrng,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='gray',#'gist_rainbow_r',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    # add contour labels to wind speed
    ax.clabel(cons[1],levels=spdrng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')
    
    


# In[21]:


dynrng


# In[86]:


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
    advContours = 1.0E-03 * np.arange(-3., 3.01, 0.25)
    grdContours = 1.0E-04 * np.arange(0.4, 2.01, 0.2)
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
    #fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return fig

fig = generate_figure_panel(unpHdl,'test')


# In[167]:


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
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
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
    #fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return fig
fcstHr=24
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)
fig = generate_figure_panel_left(unpHdl,ptdHdl,'test')


# In[187]:


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
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
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
    #fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return fig
fcstHr=24
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)
fig = generate_figure_panel_right(unpHdl,ptdHdl,'test')


# In[205]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import matplotlib.cm as cm
# define local functions
#
# plot_normalized_iterprofiles: generates a plot of normalized profiles (sums to 1.0 for each profile)
# for each iteration on a given colormap / plot axis
#
# INPUTS
#
# profVar (float): profile variable in (nLev,nIter) dimension
# profLev (float): profile levels in (nLev,) dimension
# titleStr (string): title for plot
# colMap (string): name of colormap
# ax (plt.axes): matplotlib.pyplot axis
#
# OUTPUTS
#
# no explicit outputs, but places figure on axis=ax
#
# DEPENDENCIES
#
# matplotlib.pyplot
# matplotlib.cm
# numpy
def plot_normalized_iterprofiles(profVar, profLev, titleStr, colMap, ax):
    # define number of levels (nl) and number of iterations (ni) based on
    # implicit dimensions of profVar [nl,ni]
    nl,ni = np.shape(profVar)
    # define colormap for each profile based on a colormap of colMap with
    # a length of [0,ni-1]
    scalarMap = cm.ScalarMappable(cmap=colMap)
    scalarMapList = scalarMap.to_rgba(range(ni))
    # loop through iterations
    for i in range(ni):
        # define profile color as ith element of scalarMapList
        profColor = list(scalarMapList[i][0:3])
        # define profile as ith iteration's profile [nl,]
        prof = profVar[:,i].squeeze()
        # plot normalized profile
        ax.plot(prof/np.abs(np.nansum(prof)),profLev,color=profColor,linewidth=2)
        #ax.plot(prof,profLev,color=profColor,linewidth=2)
    # plot a dashed zero-line profile for reference
    ax.plot(np.zeros((nl,)),profLev,color='black',linestyle='dashed',linewidth=2)
    # add title
    ax.set_title(titleStr)
    # plotting on eta-levels: reverse y-axis
    ax.invert_yaxis()
    return
#
# begin
#
if __name__ == "__main__":
    cp = 1004.
    tr = 270.
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq'
    # check to make sure dataDir ends in '/', if not, add it to end of string
    if dataDir[-1] != '/':
        dataDir = dataDir + '/'
    # list of initial condition unperturbed files for each iteration
    unpFileList = glob.glob(dataDir + 'wrfinput_d01_unpi*')
    unpFileList.sort()
    # list of initial condition perturbed files for each iteration
    ptdFileList = glob.glob(dataDir + 'wrfinput_d01_ptdi*')
    ptdFileList.sort()
    # number of files in lists (should all be equal, let's check)
    if len(unpFileList) != len(ptdFileList):
        print('LIST MISMATCH: unpFileList size={:d} ptdFileList={:d}'.format(len(unpFileList), len(ptdFileList)))
        nFiles = None
    else:
        print('{:d} files discovered'.format(len(unpFileList)))
        nFiles = len(unpFileList)
    # use wrfinput_d01_unpi00 to pick eta values on each level
    # eta value at each level is computed as etaMidLev, computed as the average between
    # the eta values at each half-level between [1.,0.]
    dn = np.asarray(Dataset(dataDir+'wrfinput_d01_unpi00').variables['DN']).squeeze()  # d(eta) on half levels
    eta = 1.0+np.cumsum(dn)
    # add 0-top level
    eta = np.append(eta, 0.)
    # compute eta mid-levels
    etaMidLev = 0.5*(eta[0:-1] + eta[1:])
    # loop through files, generating ke and ape on each level
    nIter = nFiles
    for i in range(nIter):
        print('processing iteration {:d} of {:d}'.format(i + 1, nFiles))
        unpHdl = Dataset(unpFileList[i])
        ptdHdl = Dataset(ptdFileList[i])
        if i == 0:
            lat = np.asarray(sens_hdl.variables['XLAT_M']).squeeze()
            lon = np.asarray(sens_hdl.variables['XLONG_M']).squeeze()
            nz = np.shape(np.asarray(unpHdl.variables['P']).squeeze())[0]
            ke = np.nan * np.ones((nz, nFiles))
            ape = np.nan * np.ones((nz, nFiles))
        # vorticity impact for iteration i
        u0 = np.asarray(unpHdl.variables['U']).squeeze()
        u1 = np.asarray(ptdHdl.variables['U']).squeeze()
        v0 = np.asarray(unpHdl.variables['V']).squeeze()
        v1 = np.asarray(ptdHdl.variables['V']).squeeze()
        t0 = np.asarray(get_wrf_tk(unpHdl)).squeeze()
        t1 = np.asarray(get_wrf_tk(ptdHdl)).squeeze()
        for k in range(nz):
            up = u1[k,:,:].squeeze() - u0[k,:,:].squeeze()
            vp = v1[k,:,:].squeeze() - v0[k,:,:].squeeze()
            tp = t1[k,:,:].squeeze() - t0[k,:,:].squeeze()
            ke[k,i] = 0.
            ke[k,i] = ke[k,i] + 0.5*np.sum(up**2.)
            ke[k,i] = ke[k,i] + 0.5*np.sum(vp**2.)
            ape[k,i] = 0.
            ape[k,i] = 0.5*(cp/tr)*np.sum(tp**2.)
     # 4-panel plot of each iteration's normalized profile for each kinematic term
    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    plot_normalized_iterprofiles(ke, etaMidLev, 'KE', 'gist_rainbow_r', axs.flatten()[0])
    plot_normalized_iterprofiles(ape, etaMidLev, 'APE', 'gist_rainbow_r', axs.flatten()[1])
    # save figure
    #fig.savefig('iterprof_figure.png', bbox_inches='tight', facecolor='white')
    fig
    #
    # end
    #


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import matplotlib.cm as cm
# define local functions
#
# plot_normalized_iterprofiles: generates a plot of normalized profiles (sums to 1.0 for each profile)
# for each iteration on a given colormap / plot axis
#
# INPUTS
#
# profVar (float): profile variable in (nLev,nIter) dimension
# profLev (float): profile levels in (nLev,) dimension
# titleStr (string): title for plot
# colMap (string): name of colormap
# ax (plt.axes): matplotlib.pyplot axis
#
# OUTPUTS
#
# no explicit outputs, but places figure on axis=ax
#
# DEPENDENCIES
#
# matplotlib.pyplot
# matplotlib.cm
# numpy
def plot_normalized_iterprofiles(profVar, profLev, titleStr, colMap, ax):
    # define number of levels (nl) and number of iterations (ni) based on
    # implicit dimensions of profVar [nl,ni]
    nl,ni = np.shape(profVar)
    # define colormap for each profile based on a colormap of colMap with
    # a length of [0,ni-1]
    scalarMap = cm.ScalarMappable(cmap=colMap)
    scalarMapList = scalarMap.to_rgba(range(ni))
    # loop through iterations
    for i in range(ni):
        # define profile color as ith element of scalarMapList
        profColor = list(scalarMapList[i][0:3])
        # define profile as ith iteration's profile [nl,]
        prof = profVar[:,i].squeeze()
        # plot normalized profile
        ax.plot(prof/np.abs(np.nansum(prof)),profLev,color=profColor,linewidth=2)
        #ax.plot(prof,profLev,color=profColor,linewidth=2)
    # plot a dashed zero-line profile for reference
    ax.plot(np.zeros((nl,)),profLev,color='black',linestyle='dashed',linewidth=2)
    # add title
    ax.set_title(titleStr)
    # plotting on eta-levels: reverse y-axis
    ax.invert_yaxis()
    return
#
# begin
#
if __name__ == "__main__":
    cp = 1004.
    tr = 270.
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/positive/uvTq'
    # check to make sure dataDir ends in '/', if not, add it to end of string
    if dataDir[-1] != '/':
        dataDir = dataDir + '/'
    # list of initial condition unperturbed files for each iteration
    unpFileList = glob.glob(dataDir + 'wrfout_d01_unpi*')
    unpFileList.sort()
    # list of initial condition perturbed files for each iteration
    ptdFileList = glob.glob(dataDir + 'wrfout_d01_ptdi*')
    ptdFileList.sort()
    # number of files in lists (should all be equal, let's check)
    if len(unpFileList) != len(ptdFileList):
        print('LIST MISMATCH: unpFileList size={:d} ptdFileList={:d}'.format(len(unpFileList), len(ptdFileList)))
        nFiles = None
    else:
        print('{:d} files discovered'.format(len(unpFileList)))
        nFiles = len(unpFileList)
    # use wrfinput_d01_unpi00 to pick eta values on each level
    # eta value at each level is computed as etaMidLev, computed as the average between
    # the eta values at each half-level between [1.,0.]
    dn = np.asarray(Dataset(dataDir+'wrfout_d01_unpi00').variables['DN']).squeeze()  # d(eta) on half levels
    eta = 1.0+np.cumsum(dn)
    # add 0-top level
    eta = np.append(eta, 0.)
    # compute eta mid-levels
    etaMidLev = 0.5*(eta[0:-1] + eta[1:])
    # loop through files, generating ke and ape on each level
    nIter = nFiles
    for i in range(nIter):
        print('processing iteration {:d} of {:d}'.format(i + 1, nFiles))
        unpHdl = Dataset(unpFileList[i])
        ptdHdl = Dataset(ptdFileList[i])
        if i == 0:
            lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
            lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
            nz = np.shape(np.asarray(unpHdl.variables['P']).squeeze())[0]
            ke = np.nan * np.ones((nz, nFiles))
            ape = np.nan * np.ones((nz, nFiles))
        # vorticity impact for iteration i
        u0 = np.asarray(unpHdl.variables['U']).squeeze()
        u1 = np.asarray(ptdHdl.variables['U']).squeeze()
        v0 = np.asarray(unpHdl.variables['V']).squeeze()
        v1 = np.asarray(ptdHdl.variables['V']).squeeze()
        t0 = np.asarray(get_wrf_tk(unpHdl)).squeeze()
        t1 = np.asarray(get_wrf_tk(ptdHdl)).squeeze()
        for k in range(nz):
            up = u1[k,:,:].squeeze() - u0[k,:,:].squeeze()
            vp = v1[k,:,:].squeeze() - v0[k,:,:].squeeze()
            tp = t1[k,:,:].squeeze() - t0[k,:,:].squeeze()
            ke[k,i] = 0.
            ke[k,i] = ke[k,i] + 0.5*np.sum(up**2.)
            ke[k,i] = ke[k,i] + 0.5*np.sum(vp**2.)
            ape[k,i] = 0.
            ape[k,i] = 0.5*(cp/tr)*np.sum(tp**2.)
     # 4-panel plot of each iteration's normalized profile for each kinematic term
    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    plot_normalized_iterprofiles(ke, etaMidLev, 'KE', 'gist_rainbow_r', axs.flatten()[0])
    plot_normalized_iterprofiles(ape, etaMidLev, 'APE', 'gist_rainbow_r', axs.flatten()[1])
    # save figure
    #fig.savefig('iterprof_figure.png', bbox_inches='tight', facecolor='white')
    fig
    #
    # end
    #


# In[199]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import matplotlib.cm as cm
# define local functions
#
# plot_normalized_iterprofiles: generates a plot of normalized profiles (sums to 1.0 for each profile)
# for each iteration on a given colormap / plot axis
#
# INPUTS
#
# profVar (float): profile variable in (nLev,nIter) dimension
# profLev (float): profile levels in (nLev,) dimension
# titleStr (string): title for plot
# colMap (string): name of colormap
# ax (plt.axes): matplotlib.pyplot axis
#
# OUTPUTS
#
# no explicit outputs, but places figure on axis=ax
#
# DEPENDENCIES
#
# matplotlib.pyplot
# matplotlib.cm
# numpy
def plot_normalized_iterprofiles(profVar, profLev, titleStr, colMap, ax):
    # define number of levels (nl) and number of iterations (ni) based on
    # implicit dimensions of profVar [nl,ni]
    nl,ni = np.shape(profVar)
    # define colormap for each profile based on a colormap of colMap with
    # a length of [0,ni-1]
    scalarMap = cm.ScalarMappable(cmap=colMap)
    scalarMapList = scalarMap.to_rgba(range(ni))
    # loop through iterations
    for i in range(ni):
        # define profile color as ith element of scalarMapList
        profColor = list(scalarMapList[i][0:3])
        # define profile as ith iteration's profile [nl,]
        prof = profVar[:,i].squeeze()
        # plot normalized profile
        ax.plot(prof/np.abs(np.nansum(prof)),profLev,color=profColor,linewidth=2)
    # plot a dashed zero-line profile for reference
    ax.plot(np.zeros((nl,)),profLev,color='black',linestyle='dashed',linewidth=2)
    # add title
    ax.set_title(titleStr)
    # plotting on eta-levels: reverse y-axis
    ax.invert_yaxis()
    return
#
# begin
#
if __name__ == "__main__":
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq'
    # check to make sure dataDir ends in '/', if not, add it to end of string
    if dataDir[-1] != '/':
        dataDir = dataDir + '/'
    # list of kinematic sensitivity files for each iteration
    sensFileList = glob.glob(dataDir + 'gradient_kinematic_d01_unpi*')
    sensFileList.sort()
    # list of initial condition kinematic perturbation files for each iteration
    pertFileList = glob.glob(dataDir + 'wrfinput_kinematic_d01_perti*')
    pertFileList.sort()
    # number of files in lists (should all be equal, let's check)
    if len(sensFileList) != len(pertFileList):
        print('LIST MISMATCH: sensFileList size={:d} pertFileList={:d}'.format(len(sensFileList), len(pertFileList)))
        nFiles = None
    else:
        print('{:d} files discovered'.format(len(sensFileList)))
        nFiles = len(sensFileList)
    # use wrfinput_d01_unpi00 to pick eta values on each level
    # eta value at each level is computed as etaMidLev, computed as the average between
    # the eta values at each half-level between [1.,0.]
    dn = np.asarray(Dataset(dataDir+'wrfinput_d01_unpi00').variables['DN']).squeeze()  # d(eta) on half levels
    eta = 1.0+np.cumsum(dn)
    # add 0-top level
    eta = np.append(eta, 0.)
    # compute eta mid-levels
    etaMidLev = 0.5*(eta[0:-1] + eta[1:])
    # loop through files, generating impacts (sens*pert) at each grid-point for each
    # kinematic variable: VOR, DIV, STR, SHR
    # lat and lon are picked on the first sens-file, should be identical for all files
    # impact arrays are defined on first sens-file (dimensions copied from sens data)
    nIter = nFiles
    for i in range(nIter):
        print('processing iteration {:d} of {:d}'.format(i + 1, nFiles))
        sens_hdl = Dataset(sensFileList[i])
        pert_hdl = Dataset(pertFileList[i])
        if i == 0:
            lat = np.asarray(sens_hdl.variables['XLAT_M']).squeeze()
            lon = np.asarray(sens_hdl.variables['XLONG_M']).squeeze()
            nz, ny, nx = np.shape(np.asarray(sens_hdl.variables['A_VOR']).squeeze())
            impVOR = np.nan * np.ones((nz, ny, nx, nFiles))
            impDIV = np.nan * np.ones((nz, ny, nx, nFiles))
            impSTR = np.nan * np.ones((nz, ny, nx, nFiles))
            impSHR = np.nan * np.ones((nz, ny, nx, nFiles))
        # vorticity impact for iteration i
        impVOR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_VOR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_VOR']).squeeze())
        # divergence impact for iteration i
        impDIV[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_DIV']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_DIV']).squeeze())
        # stretching deformation impact for iteration i
        impSTR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_STR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_STR']).squeeze())
        # shearing deformation impact for iteration i
        impSHR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_SHR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_SHR']).squeeze())
    # fix longitudes
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # generate vertical-profiles of impact: (nz,nIter) dimensions
    impVORVertProf = np.nan * np.ones((nz, nIter))
    impDIVVertProf = np.nan * np.ones((nz, nIter))
    impSTRVertProf = np.nan * np.ones((nz, nIter))
    impSHRVertProf = np.nan * np.ones((nz, nIter))
    for i in range(nIter):
        for z in range(nz):
            impVORVertProf[z, i] = np.nansum(impVOR[z, :, :, i].squeeze().flatten())
            impDIVVertProf[z, i] = np.nansum(impDIV[z, :, :, i].squeeze().flatten())
            impSTRVertProf[z, i] = np.nansum(impSTR[z, :, :, i].squeeze().flatten())
            impSHRVertProf[z, i] = np.nansum(impSHR[z, :, :, i].squeeze().flatten())
     # 4-panel plot of each iteration's normalized profile for each kinematic term
    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    plot_normalized_iterprofiles(impVORVertProf, etaMidLev, 'Vorticity', 'gist_rainbow_r', axs.flatten()[0])
    plot_normalized_iterprofiles(impDIVVertProf, etaMidLev, 'Divergence', 'gist_rainbow_r', axs.flatten()[1])
    plot_normalized_iterprofiles(impSTRVertProf, etaMidLev, 'Stretching Def.', 'gist_rainbow_r', axs.flatten()[2])
    plot_normalized_iterprofiles(impSHRVertProf, etaMidLev, 'Shearing Def.', 'gist_rainbow_r', axs.flatten()[3])
    # save figure
    #fig.savefig('iterprof_figure.png', bbox_inches='tight', facecolor='white')
    fig
    #
    # end
    #


# In[108]:


def generate_figure_panel(unpInitHdl, ptdInitHdl, unpFcstHdl, ptdFcstHdl):
    # constants
    cp = 1004.  # specific heat at constant pressure
    tr = 270.   # reference temperature
    L = 2.5104E+06  # latent heat of condensation
    eps = 1.0  # latent heat coefficient
    # extract eta-levels from unpInitHdl (should be identical for all files)
    eta = np.asarray(unpInitHdl.variables['ZNU']).squeeze()  # d(eta) on half levels
    # extract (u,v,T,q) from each WRF file handle
    unpInitU = np.asarray(unpInitHdl.variables['U']).squeeze()
    ptdInitU = np.asarray(ptdInitHdl.variables['U']).squeeze()
    unpInitV = np.asarray(unpInitHdl.variables['V']).squeeze()
    ptdInitV = np.asarray(ptdInitHdl.variables['V']).squeeze()
    unpInitT = np.asarray(get_wrf_tk(unpInitHdl)).squeeze()
    ptdInitT = np.asarray(get_wrf_tk(ptdInitHdl)).squeeze()
    unpInitQ = np.asarray(unpInitHdl.variables['QVAPOR']).squeeze()
    ptdInitQ = np.asarray(ptdInitHdl.variables['QVAPOR']).squeeze()
    unpFcstU = np.asarray(unpFcstHdl.variables['U']).squeeze()
    ptdFcstU = np.asarray(ptdFcstHdl.variables['U']).squeeze()
    unpFcstV = np.asarray(unpFcstHdl.variables['V']).squeeze()
    ptdFcstV = np.asarray(ptdFcstHdl.variables['V']).squeeze()
    unpFcstT = np.asarray(get_wrf_tk(unpFcstHdl)).squeeze()
    ptdFcstT = np.asarray(get_wrf_tk(ptdFcstHdl)).squeeze()
    unpFcstQ = np.asarray(unpFcstHdl.variables['QVAPOR']).squeeze()
    ptdFcstQ = np.asarray(ptdFcstHdl.variables['QVAPOR']).squeeze()
    # compute perturbation quantities
    pInitU = ptdInitU - unpInitU
    pInitV = ptdInitV - unpInitV
    pInitT = ptdInitT - unpInitT
    pInitQ = ptdInitQ - unpInitQ
    pFcstU = ptdFcstU - unpFcstU
    pFcstV = ptdFcstV - unpFcstV
    pFcstT = ptdFcstT - unpFcstT
    pFcstQ = ptdFcstQ - unpFcstQ
    # compute initial and final energy profiles
    pInitKE = np.nan * np.ones(np.shape(eta))
    pInitAPE = np.nan * np.ones(np.shape(eta))
    pInitQE = np.nan * np.ones(np.shape(eta))
    pFcstKE = np.nan * np.ones(np.shape(eta))
    pFcstAPE = np.nan * np.ones(np.shape(eta))
    pFcstQE = np.nan * np.ones(np.shape(eta))
    for k in range(np.size(eta)):
        up = pInitU[k,:,:].squeeze()
        vp = pInitV[k,:,:].squeeze()
        tp = pInitT[k,:,:].squeeze()
        qp = pInitQ[k,:,:].squeeze()
        pInitKE[k] = 0.
        pInitKE[k] = pInitKE[k] + 0.5 * np.sum(up**2.)
        pInitKE[k] = pInitKE[k] + 0.5 * np.sum(vp**2.)
        pInitAPE[k] = 0.
        pInitAPE[k] = pInitAPE[k] + 0.5 * (cp/tr) * np.sum(tp**2.)
        pInitQE[k] = 0.
        pInitQE[k] = pInitKE[k] + 0.5 * eps * L**2./(cp*tr) * np.sum(qp**2.)
        
        up = pFcstU[k,:,:].squeeze()
        vp = pFcstV[k,:,:].squeeze()
        tp = pFcstT[k,:,:].squeeze()
        qp = pFcstQ[k,:,:].squeeze()
        pFcstKE[k] = 0.
        pFcstKE[k] = pFcstKE[k] + 0.5 * np.sum(up**2.)
        pFcstKE[k] = pFcstKE[k] + 0.5 * np.sum(vp**2.)
        pFcstAPE[k] = 0.
        pFcstAPE[k] = pFcstAPE[k] + 0.5 * (cp/tr) * np.sum(tp**2.)
        pFcstQE[k] = 0.
        pFcstQE[k] = pFcstKE[k] + 0.5 * eps * L**2./(cp*tr) * np.sum(qp**2.)
    # compute initial/forecast total energy (norm) profiles
    pInitTOT = pInitKE + pInitAPE
    pFcstTOT = pFcstKE + pFcstAPE
    # plot figure panel: initial/forecast energy norm profile, with and without QE term
    fig = plt.figure(figsize=(4,8))
    plt.plot(5. * pInitTOT, eta, color='black', linewidth=2.0)
    plt.plot(5. * (pInitTOT + pInitQE), eta, color='black', linewidth=2.0, linestyle='dotted')
    plt.plot(pFcstTOT, eta, color='orange', linewidth=2.0)
    plt.legend(['norm init (mul. 5)', 'norm init + QE (mul. 5)', 'norm final'])
    plt.gca().invert_yaxis()
    plt.show()
    return
    
dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/positive/uvTq'
unpInitHdl = Dataset(dataDir + '/wrfinput_d01_unpi00')
ptdInitHdl = Dataset(dataDir + '/wrfinput_d01_ptdi14')
unpFcstHdl = Dataset(dataDir + '/wrfout_d01_unpi00')
ptdFcstHdl = Dataset(dataDir + '/wrfout_d01_ptdi14')
generate_figure_panel(unpInitHdl, ptdInitHdl, unpFcstHdl, ptdFcstHdl)


# In[131]:


def generate_figure_panel(unpHdls, ptdHdls, sensHdls):
    # constants
    cp = 1004.  # specific heat at constant pressure
    tr = 270.   # reference temperature
    L = 2.5104E+06  # latent heat of condensation
    eps = 1.0  # latent heat coefficient
    # extract eta-levels from unpInitHdl (should be identical for all files)
    eta = np.asarray(unpInitHdl.variables['ZNU']).squeeze()  # d(eta) on half levels
    fig = plt.figure(figsize=(4,8))
    for i in range(len(unpHdls)):
        unpHdl = unpHdls[i]
        ptdHdl = ptdHdls[i]
        sensHdl = sensHdls[i]
        # extract sensitivity to (u,v,T,q) from sensitivity file
        sensQ = np.asarray(sensHdl.variables['A_QVAPOR']).squeeze()
        # define perturbations to (u,v,T,q) equating to 1 J/kg KE, APE, QE
        pIdealQ = np.sqrt(2.*cp*tr/(eps*L**2.))
        pInitQ = np.asarray(ptdHdl.variables['QVAPOR']).squeeze() - np.asarray(unpHdl.variables['QVAPOR']).squeeze()
        # compute impact profile
        impQIdeal = np.nan * np.ones(np.shape(eta))
        impQInit = np.nan * np.ones(np.shape(eta))
        for k in range(np.size(eta)):
            impQIdeal[k] = np.sum(np.abs(sensQ[k,:,:].squeeze() * pIdealQ))
            impQInit[k] = np.sum(np.multiply(sensQ[k,:,:].squeeze(), pInitQ[k,:,:].squeeze()))
        # plot figure panel: impact profiles
        plt.plot(impQInit, eta, color='black', linewidth=2.0)
        plt.plot(impQIdeal, eta, color='black', linewidth=2.0, linestyle='dotted')
    plt.gca().invert_yaxis()
    plt.show()
    return
    
dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq'
unpHdls = [
           Dataset(dataDir + '/wrfinput_d01_unpi00'),
           Dataset(dataDir + '/wrfinput_d01_unpi01'),
           Dataset(dataDir + '/wrfinput_d01_unpi02'),
           Dataset(dataDir + '/wrfinput_d01_unpi03'),
           Dataset(dataDir + '/wrfinput_d01_unpi04'),
           Dataset(dataDir + '/wrfinput_d01_unpi05'),
           Dataset(dataDir + '/wrfinput_d01_unpi06'),
           Dataset(dataDir + '/wrfinput_d01_unpi07'),
           Dataset(dataDir + '/wrfinput_d01_unpi08'),
           Dataset(dataDir + '/wrfinput_d01_unpi09'),
           Dataset(dataDir + '/wrfinput_d01_unpi10'),
           Dataset(dataDir + '/wrfinput_d01_unpi11'),
           Dataset(dataDir + '/wrfinput_d01_unpi12'),
           Dataset(dataDir + '/wrfinput_d01_unpi13'),
           Dataset(dataDir + '/wrfinput_d01_unpi14')
          ]
ptdHdls = [
           Dataset(dataDir + '/wrfinput_d01_ptdi00'),
           Dataset(dataDir + '/wrfinput_d01_ptdi01'),
           Dataset(dataDir + '/wrfinput_d01_ptdi02'),
           Dataset(dataDir + '/wrfinput_d01_ptdi03'),
           Dataset(dataDir + '/wrfinput_d01_ptdi04'),
           Dataset(dataDir + '/wrfinput_d01_ptdi05'),
           Dataset(dataDir + '/wrfinput_d01_ptdi06'),
           Dataset(dataDir + '/wrfinput_d01_ptdi07'),
           Dataset(dataDir + '/wrfinput_d01_ptdi08'),
           Dataset(dataDir + '/wrfinput_d01_ptdi09'),
           Dataset(dataDir + '/wrfinput_d01_ptdi10'),
           Dataset(dataDir + '/wrfinput_d01_ptdi11'),
           Dataset(dataDir + '/wrfinput_d01_ptdi12'),
           Dataset(dataDir + '/wrfinput_d01_ptdi13'),
           Dataset(dataDir + '/wrfinput_d01_ptdi14')
          ]
sensHdls = [
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi00'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi01'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi02'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi03'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi04'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi05'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi06'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi07'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi08'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi09'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi10'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi11'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi12'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi13'),
            Dataset(dataDir + '/gradient_wrfplus_d01_unpi14')
           ]
generate_figure_panel(unpHdls, ptdHdls, sensHdls)


# In[109]:


unpHdl.variables


# In[21]:


fig,ax = plt.subplots(nrows=1,ncols=1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = right_panel(ax,(unpHdl,ptdHdl))


# In[19]:


for fcstHr in range(13):
    latBegList = [47.5]
    lonBegList = [-98.5]
    latEndList = [28.5]
    lonEndList = [-70.]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    unpW = wrf.getvar(unpHdl,'omega')
    ptdW = wrf.getvar(ptdHdl,'omega')
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    unpThta = get_wrf_th(unpHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpSpd350 = wrf.interplevel(field3d=unpSpd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u,v = get_uvmet(ptdHdl)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdW_int = interpolate_sigma_levels(ptdW, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    xSectShadInterval=np.arange(-2.5, 2.51, 0.25)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig, latLists, lonLists = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=unpW,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpSpd350,
                                 thkInterval=spdrng,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) unperturbed omega'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')
    
    fig, latLists, lonLists = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdW_int,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpSpd350,
                                 thkInterval=spdrng,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed omega'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[80]:


# For a selected time, generate a plan section of heights and isotachs at 250 hPa (core of subtropical jet)
# and temperature advection by the geostrophic wind at 450 hPa (beneath the core, roughly at a level that
# includes the upper front)
for fcstHr in range(13):
    latBegList = [47.5]
    lonBegList = [-98.5]
    latEndList = [28.5]
    lonEndList = [-70.]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # unperturbed variables
    unpHgt = wrf.getvar(unpHdl,'z')
    unpHgt250 = wrf.interplevel(field3d=unpHgt,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=25000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt450 = wrf.interplevel(field3d=unpHgt,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpdZdx450, unpdZdy450 = get_wrf_grad(unpHdl,unpHgt450)
    f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)
    g = 9.81
    unpUgeo450 = np.multiply(-g*f**-1., unpdZdy450)
    unpVgeo450 = np.multiply(g*f**-1., unpdZdx450)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpSpd250 = wrf.interplevel(field3d=unpSpd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=25000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpUwd450 = wrf.interplevel(field3d=u,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpVwd450 = wrf.interplevel(field3d=v,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpTmp = get_wrf_tk(unpHdl)
    unpTmp450 = wrf.interplevel(field3d=unpTmp,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpdTdx450, unpdTdy450 = get_wrf_grad(unpHdl,unpTmp450)
    unpTmpAdv450 = np.multiply(-unpUgeo450,unpdTdx450) + np.multiply(-unpVgeo450,unpdTdy450)
    # perturbed variables
    ptdHgt = wrf.getvar(ptdHdl,'z')
    ptdHgt250 = wrf.interplevel(field3d=ptdHgt,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=25000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt450 = wrf.interplevel(field3d=ptdHgt,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptddZdx450, ptddZdy450 = get_wrf_grad(ptdHdl,ptdHgt450)
    f = 2. * 7.292E-05 * np.sin(lat * np.pi/180.)
    g = 9.81
    ptdUgeo450 = np.multiply(-g*f**-1., ptddZdy450)
    ptdVgeo450 = np.multiply(g*f**-1., ptddZdx450)
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    ptdSpd250 = wrf.interplevel(field3d=ptdSpd,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=25000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdUwd450 = wrf.interplevel(field3d=u,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdVwd450 = wrf.interplevel(field3d=v,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdTmp = get_wrf_tk(ptdHdl)
    ptdTmp450 = wrf.interplevel(field3d=ptdTmp,
                                vert=wrf.getvar(ptdHdl,'p'),
                                desiredlev=45000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptddTdx450, ptddTdy450 = get_wrf_grad(ptdHdl,ptdTmp450)
    ptdTmpAdv450 = np.multiply(-ptdUgeo450,ptddTdx450) + np.multiply(-ptdVgeo450,ptddTdy450)
    
    fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(30,7), subplot_kw={'projection' : datProj})

    hgtrng=np.arange(9500.,11500.1,120.)
    spdrng = np.arange(36.,100.1,8.)
    shdrng=1.0E-03*np.arange(-2.,2.1,0.2).astype('float16')
    mask = np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)]=False
    # left panel: unperturbed case
    ax =  axs[0]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt250,unpSpd250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=unpTmpAdv450,
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 250 hPa geop. height and isotachs, 450 geostr. temp. adv.'.format(fcstHr))
    
    # right panel: perturbed case
    ax =  axs[1]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[ptdHgt250,ptdSpd250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTmpAdv450,
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed 250 hPa geop. height and isotachs, 450 geostr. temp. adv.'.format(fcstHr))
    # add cross-section line to both panels
    axs[0].plot(lonBegList[0],latBegList[0],'o',transform=plotProj,color='magenta')
    axs[0].plot(lonEndList[0],latEndList[0],'o',transform=plotProj,color='magenta')
    axs[0].plot((lonBegList[0],lonEndList[0]),(latBegList[0],latEndList[0]),transform=plotProj,color='magenta')
    axs[1].plot(lonBegList[0],latBegList[0],'o',transform=plotProj,color='magenta')
    axs[1].plot(lonEndList[0],latEndList[0],'o',transform=plotProj,color='magenta')
    axs[1].plot((lonBegList[0],lonEndList[0]),(latBegList[0],latEndList[0]),transform=plotProj,color='magenta')
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'_2panel.png',bbox_inches='tight',facecolor='white')
    
    # separately, plot the temperature advection difference
    fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

    hgtrng=np.arange(9500.,11500.1,120.)
    spdrng = np.arange(36.,100.1,8.)
    shdrng=1.0E-03*np.arange(-2.,2.1,0.2).astype('float16')
    mask = np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)]=False
    # difference plot of 450 hPa T-adv on unperturbed 250 hPa state
    ax =  axs
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt250,unpSpd250],
                                            contIntervalList=[hgtrng,spdrng], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[0.75,1.5],
                                            shadVariable=ptdTmpAdv450-unpTmpAdv450,
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed 250 hPa geop. height and isotachs, perturbation 450 geostr. temp. adv.'.format(fcstHr))
    # add cross-section line
    axs.plot(lonBegList[0],latBegList[0],'o',transform=plotProj,color='magenta')
    axs.plot(lonEndList[0],latEndList[0],'o',transform=plotProj,color='magenta')
    axs.plot((lonBegList[0],lonEndList[0]),(latBegList[0],latEndList[0]),transform=plotProj,color='magenta')
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'_diff.png',bbox_inches='tight',facecolor='white')


# In[91]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and perturbation precipitation
for fcstHr in range(13):
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define prior datetime stamp, if fcstHr==0, set to 0
    if fcstHr == 0:
        dtPrior = dtFcst
        dtPriorStr = dtFcstStr
    else:
        dtPrior = dtFcst - datetime.timedelta(hours=1)
        dtPriorStr = datetime.datetime.strftime(dtPrior,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    unpFilePrior = unpDir + 'wrfout_d01_' + dtPriorStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    ptdFilePrior = posDir + 'wrfout_d01_' + dtPriorStr
    unpHdl = Dataset(unpFileFcst)
    unpHdlPrior = Dataset(unpFilePrior)
    ptdHdl = Dataset(ptdFileFcst)
    ptdHdlPrior = Dataset(ptdFilePrior)
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract pressure for interpolation
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    # extract accumulated non-convective precipitation, compute hourly
    unpHrPrecipNC = np.asarray(unpHdl.variables['RAINNC']).squeeze() - np.asarray(unpHdlPrior.variables['RAINNC']).squeeze()
    ptdHrPrecipNC = np.asarray(ptdHdl.variables['RAINNC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINNC']).squeeze()
    # extract accumulated convective precipitation, compute hourly
    unpHrPrecipC = np.asarray(unpHdl.variables['RAINC']).squeeze() - np.asarray(unpHdlPrior.variables['RAINC']).squeeze()
    ptdHrPrecipC = np.asarray(ptdHdl.variables['RAINC']).squeeze() - np.asarray(ptdHdlPrior.variables['RAINC']).squeeze()
    # interpolate heights to 850 and 500 hPa
    z850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=prs,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    z500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                           vert=prs,
                           desiredlev=50000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    # compute 850-500 thickness
    thk = z500 - z850
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)
    shdrng=np.arange(-10., 10.1, 1.).astype('float16')
    mask=np.ones(np.shape(shdrng),dtype=bool)
    mask[np.where(shdrng==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[slp,thk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=(ptdHrPrecipC+ptdHrPrecipNC)-(unpHrPrecipC+unpHrPrecipNC),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) unperturbed sea level pressure, 850-500 hPa thickness, perturbation precip.'.format(fcstHr))
    # add contour labels to slp
    ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[92]:


unpHdl.variables


# In[47]:


x=np.asarray(hdl.variables['G_MU']).squeeze()


# In[56]:


plt.contourf(x)
plt.colorbar()
plt.show()


# In[146]:


# Plot all response-function boxes for a given case along with the 24-hr forecast SLP of the last iteration


negSensList = glob('/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq/final_sens_d01_unpi*')
negSensList.sort()
negSensList.reverse()
posSensList = glob('/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/positive/uvTq/final_sens_d01_unpi*')
posSensList.sort()

sensList = negSensList.copy()
senList = sensList.extend(posSensList)



fcstHr = 24
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
FileFcst = negDir + 'wrfout_d01_' + dtFcstStr
wrfHdl = Dataset(FileFcst)
# extract latitude and longitude, set longitude to 0 to 360 deg format 
lat = np.asarray(wrfHdl.variables['XLAT']).squeeze()
lon = np.asarray(wrfHdl.variables['XLONG']).squeeze()
fix = np.where(lon < 0.)
lon[fix] = lon[fix] + 360.
# define data and plot projection
datProj = gen_cartopy_proj(wrfHdl)
plotProj = ccrs.PlateCarree()
cvlist = [get_wrf_slp(wrfHdl)]
cclist = ['green']
ctlist = [1.0]
cilist = [np.arange(900.,1040.1,4.)]
for i in [0]:#range(len(sensList)):
    sensFile = sensList[i]
    hdl = Dataset(sensFile)
    cvlist.append(-1*np.asarray(hdl.variables['G_MU']).squeeze())
    cilist.append([-1.,0.,1.])
    if 'final_sens_d01_unpi00' in sensFile:
        cclist.append('black')
    else:
        cclist.append(matplotlib.colors.to_hex((1.-i/len(sensList), 0., i/len(sensList))))
    if 'final_sens_d01_unpi00' in sensFile:
        ctlist.append(3.0)
    else:
        ctlist.append(1.5)

fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

ax=axs

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=cvlist,
                                        contIntervalList=cilist, 
                                        contColorList=cclist,
                                        contLineThicknessList=ctlist,
                                        shadVariable=None,
                                        shadInterval=None,
                                        shadAlpha=0.5,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap=None,
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=ax)
# add a title
ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed sea level pressure, response function boxes'.format(fcstHr))
# save file
fcstHrStr=str(fcstHr).zfill(2)
fig.savefig('fig_tank/rfboxf'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[126]:


'final_sens_d01_unpi00' in sensList[14]


# In[147]:


sensList[15]


# In[143]:


len(sensList)


# In[118]:


len(cclist)


# In[113]:


ci = cilist[0]
cc = cclist[0]
cv = cvlist[0]
ct = ctlist[0]


fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

ax=axs

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=cv,
                                        contIntervalList=[ci], 
                                        contColorList=cc,
                                        contLineThicknessList=ct,
                                        shadVariable=None,
                                        shadInterval=None,
                                        shadAlpha=0.5,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=ax)


# In[50]:


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
    hgtrng = np.arange(-50., 50.1, 5.).astype('float16')
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
                                            shadVariable=unpTADVmean,#ptdTADVmean-unpTADVmean,
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 40000., 70000.)),
                                 presLevMin=10000.,
                                 xSectTitleStr=None,
                                 xLineColorList=['black']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')


# In[52]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

fcstHr = 0

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# ACROSS SHEAR THROUGH MID-TROP HEIGHT PERTURBATIONS
sampleHrs=[0., 3., 6., 9., 12.]
sampleLatBegList=[45., 45., 45., 45., 45.]
sampleLonBegList=[-83., -79., -75., -75., -81.]
sampleLatEndList=[25., 25., 25., 25., 25.]
sampleLonEndList=[-88., -85., -86., -81.5, -74.]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(13.))

generate_figure_panel(unpHdl, ptdHdl, hourlyLatBegList[fcstHr], hourlyLonBegList[fcstHr],
                      hourlyLatEndList[fcstHr], hourlyLonEndList[fcstHr], 'test')


# In[1]:


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

# define function for plan-section plot: 250 hPa geopotential height and wind-speed
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


# In[2]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

fcstHr = 0

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# ALONG TROUGH AXIS CROSS SECTIONS (roughly tracks neg. pert. Tadv, appears to show PV intrusion)
sampleHrs=[0., 3., 6., 9., 12.]
sampleLatBegList=[45., 45., 45., 45., 45.]
sampleLonBegList=[-83., -81.5, -80., -78., -77.5]
sampleLatEndList=[25., 25., 25., 25., 25.]
sampleLonEndList=[-88., -85., -82.5, -81.5, -79.5]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(13.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(13.))

generate_cross_temperature_panel(unpHdl, ptdHdl, hourlyLatBegList[fcstHr], hourlyLonBegList[fcstHr],
                                 hourlyLatEndList[fcstHr], hourlyLonEndList[fcstHr], 'black', 'test')


# NW-->SE THROUGH MID-TROPOSPHERE TEMPERATURE PERTUBATIONS
latBeg=47.0
lonBeg=-102.
latEnd=27.0
lonEnd=-72.

generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'test')


# CROSS-SHEAR THROUGH MID-TROPOSPHERE SPEED PERTUBATIONS
latBeg=45.0
lonBeg=-83.
latEnd=25.0
lonEnd=-94.

generate_cross_speed_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                            latEnd, lonEnd, 'black', 'test')


# In[18]:


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

# define function for plan-section plot: 250 hPa geopotential height and wind-speed
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


# In[35]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

fcstHr = 0

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# CROSS-SHEAR THROUGH MID-TROPOSPHERE SPEED PERTUBATIONS
latBeg=45.0
lonBeg=-83.
latEnd=25.0
lonEnd=-94.

generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                            latEnd, lonEnd, 'lime', 'test')


# In[101]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import plan_section_plot
import datetime
import wrf
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import matplotlib
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr

# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
def generate_figure_panel(unpHdl, ptdHdl, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract unperturbed sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    # interpolate unperturbed heights to 850 and 500 hPa
    unpZ850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    unpZ500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    # compute unperturbed 850-500 thickness
    unpThk = unpZ500 - unpZ850
    # compute mean temperature of lowest 10 model layers
    t = np.asarray(get_wrf_tk(unpHdl))
    unpTmean = np.mean(t[0:10,:,:], axis=0).squeeze()
    t = np.asarray(get_wrf_tk(ptdHdl))
    ptdTmean = np.mean(t[0:10,:,:], axis=0).squeeze()
    # generate figure
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)
    tDiffRng=np.arange(-5., 5.01, 0.5).astype('float16')
    mask = np.ones(np.shape(tDiffRng)).astype('bool')
    mask[np.where(tDiffRng==0.)] = False

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp,unpThk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=ptdTmean-unpTmean,
                                            shadInterval=tDiffRng[mask],
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
    ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')


# In[107]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

fcstHr = 12

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

generate_figure_panel(unpHdl, ptdHdl, 'test')


# In[149]:


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


# In[150]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

ptdDir = posDir

fcstHr = 0

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# CROSS-SHEAR THROUGH MID-TROPOSPHERE SPEED PERTUBATIONS
latBeg=55.0
lonBeg=-80.
latEnd=37.0
lonEnd=-50.

generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                            latEnd, lonEnd, 'black', 'test')

fcstHr = 6

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# CROSS-SHEAR THROUGH MID-TROPOSPHERE SPEED PERTUBATIONS
latBeg=50.0
lonBeg=-83.
latEnd=41.0
lonEnd=-47.

generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                            latEnd, lonEnd, 'black', 'test')

fcstHr = 12

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

# CROSS-SHEAR THROUGH MID-TROPOSPHERE SPEED PERTUBATIONS
latBeg=50.0
lonBeg=-83.5
latEnd=40.0
lonEnd=-46.5

generate_cross_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg,
                            latEnd, lonEnd, 'black', 'test')


# In[90]:


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import plan_section_plot
import datetime
import wrf
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import matplotlib
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr

# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
def generate_figure_panel(unpHdl, ptdHdl, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl)
    plotProj = ccrs.PlateCarree()
    # extract unperturbed sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    # interpolate unperturbed heights to 850 and 500 hPa
    unpZ850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    unpZ500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    # compute unperturbed 850-500 thickness
    unpThk = unpZ500 - unpZ850
    # compute precipitation
    unpPrcp = np.asarray(unpHdl.variables['RAINC']).squeeze() + np.asarray(unpHdl.variables['RAINNC']).squeeze()
    ptdPrcp = np.asarray(ptdHdl.variables['RAINC']).squeeze() + np.asarray(ptdHdl.variables['RAINNC']).squeeze()
    # generate figure
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    slprng=np.arange(900.,1030.1,4.)
    thkrng=np.arange(3700.,4500.1,50.)
    tDiffRng=np.arange(-5., 5.01, 0.5).astype('float16')
    mask = np.ones(np.shape(tDiffRng)).astype('bool')
    mask[np.where(tDiffRng==0.)] = False

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpSlp,unpThk],
                                            contIntervalList=[slprng,thkrng], 
                                            contColorList=['black','#b06407'],
                                            contLineThicknessList=[0.75,0.75],
                                            shadVariable=ptdPrcp-unpPrcp,
                                            shadInterval=tDiffRng[mask],
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
    ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')


# In[93]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
dtInit = datetime.datetime(2020, 3, 6, 12)

fcstHr = 24

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFileFcst)
ptdHdl = Dataset(ptdFileFcst)

generate_figure_panel(unpHdl, ptdHdl, 'test')


# In[ ]:




