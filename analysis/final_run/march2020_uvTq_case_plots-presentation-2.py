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


# In[2]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
d10Dir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/10day_files/'
dtInit = datetime.datetime(2020, 3, 6, 12)
dtAvgBeg = datetime.datetime(2020, 3, 2, 12)
dtAvgEnd = datetime.datetime(2020, 3, 12, 12)


# In[105]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
for fcstHr in range(25):
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


# In[101]:


np.min(slp)


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
    filePert = negDir + 'wrfout_d01_' + dtFcstStr
    filePrior = negDir + 'wrfout_d01_' + dtPriorStr
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


# In[114]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and 
# SLP-perturbation
for fcstHr in range(25):
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


# In[115]:


# For a selected forecast time, plot the 500 hPa geopotential heights, perturbation vorticity and heights
for fcstHr in [24]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = posDir + 'wrfout_d01_' + dtFcstStr
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


# In[6]:


# For a selected forecast time, plot the 250/350/450 hPa geopotential heights, wind speed, and perturbation wind speed
for fcstHr in [0]:
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    filePert = posDir + 'wrfout_d01_' + dtFcstStr
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


# In[7]:


fig, axs = plt.subplots(ncols=2,nrows=1,figsize=(30,7), subplot_kw={'projection' : datProj})

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
    
    ax=axs[0]
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[posHgt450-unpHgt450],
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
    ax.set_title('(0/6/12 hrs) intensifying perturbed Hght 450 hPa')
    ax=axs[1]
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
    # add a title
    ax.set_title('(0/6/12 hrs) weakening perturbed Hght 450 hPa')
latBeg=47.5
lonBeg=-98.5
latEnd=28.5
lonEnd=-70.
axs[0].plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs[0].plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
axs[0].plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
axs[1].plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs[1].plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
axs[1].plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
fig.savefig('fig_tank/cross_section_plan_hgtPert.png',bbox_inches='tight',facecolor='white')


# In[116]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbed
# potential temperature (interpolated to the unperturbed sigma-levels)
for fcstHr in [24]:
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
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    unpThta = get_wrf_th(unpHdl)
    ptdThta = get_wrf_th(ptdHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    ptdThta_int = interpolate_sigma_levels(ptdThta, p, ps, pt, s, unpHdl)
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    xSectShadInterval = np.arange(-10., 10.1, 1.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdThta_int-unpThta,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed potential temperature'.format(fcstHr)
                                )
    xSectShadInterval = np.arange(-2., 2.01, 0.25)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[117]:


for fcstHr in [24]:
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
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    unpThta = get_wrf_th(unpHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u,v = get_uvmet(ptdHdl)
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    xSectShadInterval = np.arange(-4., 4.01, 0.5)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdPvor_int-unpPvor,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed potential vorticity'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[136]:


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
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    unpThta = get_wrf_th(unpHdl)
    ptdThta = get_wrf_th(ptdHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u,v = get_uvmet(ptdHdl)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdHgt_int = interpolate_sigma_levels(ptdHgt, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    hgtrng = np.arange(-120., 120.1, 12.)
    hgtrng = hgtrng[np.where(hgtrng != 0.)]
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    xSectShadInterval=np.arange(-120., 120.1, 12.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdHgt_int-unpHgt,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed geopotential height'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[137]:


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
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    unpVor = get_wrf_kinematic(unpHdl,'vor')
    ptdVor = get_wrf_kinematic(unpHdl,'vor')
    unpThta = get_wrf_th(unpHdl)
    ptdThta = get_wrf_th(ptdHdl)
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u,v = get_uvmet(ptdHdl)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdVor_int = interpolate_sigma_levels(ptdVor, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    vorrng = np.arange(-10., 10.1, 1.)
    vorrng = vorrng[np.where(vorrng != 0.)]
    spdrng = np.arange(35.,100.,5.)
    thtarng = np.arange(280.,450.,4.)
    pvorrng = [2.]
    
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    xSectShadInterval=1.0E-06 * np.arange(-8., 8.1, 1.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdVor_int-unpVor,
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed vorticity'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[17]:


for fcstHr in [4]:
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
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpThta, unpPvor, ptdPvor_int, unpSpd],
                                 xSectContIntervalList=[thtarng, [2.], [2.], spdrng],
                                 xSectContColorList=['black', '#1d913c', 'gold', '#818281'],
                                 xSectContLineThicknessList=[0.5, 3., 3., 2.],
                                 xSectShadVariable=ptdW_int-unpW,
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


# In[18]:


for fcstHr in [4]:
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
    fig = cross_section_plot(
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
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed omega'.format(fcstHr)
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')
    
    fig = cross_section_plot(
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


# In[97]:


# Plot all response-function boxes for a given case along with the 24-hr forecast SLP of the last iteration

sensList = glob('/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/positive/uvTq/final_sens_d01_unpi*')
sensList.sort()

fcstHr = 24
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
ptdHdl = Dataset(ptdFileFcst)
# extract latitude and longitude, set longitude to 0 to 360 deg format 
lat = np.asarray(ptdHdl.variables['XLAT']).squeeze()
lon = np.asarray(ptdHdl.variables['XLONG']).squeeze()
fix = np.where(lon < 0.)
lon[fix] = lon[fix] + 360.
# define data and plot projection
datProj = gen_cartopy_proj(ptdHdl)
plotProj = ccrs.PlateCarree()
slp = get_wrf_slp(ptdHdl)
slprng=np.arange(900.,1030.1,4.)

cvlist = [slp]
cilist = [slprng]
cclist = ['black']
ctlist = [1.]
for i in range(len(sensList)):
    sensFile = sensList[i]
    hdl = Dataset(sensFile)
    cvlist.append(-1.*np.asarray(hdl.variables['G_MU']).squeeze())
    cilist.append([-9999.,0.,1.])
    cclist.append(matplotlib.colors.to_hex((i/len(sensList), 0., 1.-i/len(sensList))))
    ctlist.append(1.5)

fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

ax=axs

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=ptdHdl,
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
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=ax)
# add a title
ax.set_title(dtFcstStr + ' ({:d} hrs) perturbed sea level pressure, response function boxes'.format(fcstHr))
# add contour labels to slp
ax.clabel(cons[0],levels=slprng[::2])
# save file
fcstHrStr=str(fcstHr).zfill(2)
fig.savefig('fig_tank/rfboxf'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[3]:


# For a selected forecast time, plot the geopoential heights, isotachs, and temperature advection by
# the geostrophic wind
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


dTdx, dTdy = get_wrf_grad(wrfHdl,get_wrf_tk(wrfHdl))
u,v = get_uvmet(wrfHdl)
t = get_wrf_tk(wrfHdl)


# In[13]:


lev=50000.
t0 = wrf.interplevel(field3d=t,
                       vert=wrf.getvar(wrfHdl,'p'),
                       desiredlev=lev,
                       missing=np.nan,
                       squeeze=True,
                       meta=False)
tx0, ty0 = get_wrf_grad(wrfHdl,t0)
u0 = wrf.interplevel(field3d=u,
                       vert=wrf.getvar(wrfHdl,'p'),
                       desiredlev=lev,
                       missing=np.nan,
                       squeeze=True,
                       meta=False)
v0 = wrf.interplevel(field3d=v,
                       vert=wrf.getvar(wrfHdl,'p'),
                       desiredlev=lev,
                       missing=np.nan,
                       squeeze=True,
                       meta=False)
shd=plt.contourf(lon,lat,np.multiply(-u0,tx0)+np.multiply(-v0,ty0),np.arange(-.0024,.0025,0.0001),cmap='seismic',vmin=-.0024,vmax=.0024)
#shd=plt.contourf(lon,lat,np.sqrt(tx0**2. + ty0**2.),cmap='jet')
plt.contour(lon,lat,t0,np.arange(240.,310.,3.),colors='k')
skp=10
plt.quiver(lon[::skp,::skp],lat[::skp,::skp],u0[::skp,::skp],v0[::skp,::skp])
#plt.contourf(ty0)
plt.colorbar(mappable=shd)
plt.show()


# In[8]:


p=wrf.getvar(wrfHdl,'p')
k=18
plt.contourf(lon,lat,p[k,:,:].squeeze())
plt.colorbar()
plt.show()


# In[6]:


tx1,ty1 = get_wrf_grad(wrfHdl,t0)


# In[8]:


plt.contourf(lon,lat,tx0)
plt.colorbar()
plt.show()

plt.contourf(lon,lat,tx1)
plt.colorbar()
plt.show()


# In[ ]:



