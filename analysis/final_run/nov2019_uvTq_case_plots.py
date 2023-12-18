#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/negative/uvTq/ptdi19/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
d10Dir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/10day_files/'
dtInit = datetime.datetime(2019, 11, 25, 12)
#dtAvgBeg = datetime.datetime(2020, 3, 2, 12)
#dtAvgEnd = datetime.datetime(2019, 11, 27, 0)


# In[5]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[5]:


np.asarray(wrfHdl.variables['ZNU']).squeeze()[1:]-np.asarray(wrfHdl.variables['ZNU']).squeeze()[0:-1]


# In[6]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, precipitation, and 
# SLP-perturbation
for fcstHr in [36]:
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


# In[7]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and 
# SLP-perturbation
for fcstHr in [36]:
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


# In[10]:


# For a selected forecast time, plot the 500 hPa geopotential heights, perturbation vorticity and heights
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[11]:


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


# In[209]:


# Generate cross-section lines:
#
# At discrete time-intervals, plot the sea-level pressure perturbation and 850-500 hPa thickness
# Align cross-section centered on SLP minimum and oriented along thickness gradient at center
# Constrain cross-section segment-length to 30 degrees of longitude
#
sampleHrs=[0., 6., 12., 18., 24., 30., 36.]
sampleLatBegList=[42.75, 49.5, 51., 52., 52., 50.25, 49.5]
sampleLonBegList=[-172.5, -165., -162., -158., -154., -149., -146.]
sampleLatEndList=[48.75, 45.5, 46., 43.5, 41., 38.75, 35.]
sampleLonEndList=[-142.5, -135., -132., -128., -124., -119., -116.]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(37.))


fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

for fcstHr in [6]:
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
    unpSlp = get_wrf_slp(unpHdl)
    posSlp = get_wrf_slp(posHdl)
    unpH850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    unpH500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    
    #slprng=np.arange(900.,1030.1,2.)
    slprng=np.arange(-18.,18.1,2.)
    smask=np.ones(np.size(slprng),dtype='bool')
    smask[np.where(slprng==0.)]=False
    
    ax=axs
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[posSlp-unpSlp,unpH500-unpH850],
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
    ax.set_title('SLP and 850-500 thickness')
# presentation cross-section values
# latBeg = 47.5
# lonBeg = -98.5
# latEnd = 28.5
# lonEnd = -70.
# modified cross-section values
latBeg = 49.5
lonBeg = -165.
latEnd = 45.5
lonEnd = -135.
# final(?) cross-section values
latBeg=hourlyLatBegList[fcstHr]
lonBeg=hourlyLonBegList[fcstHr]
latEnd=hourlyLatEndList[fcstHr]
lonEnd=hourlyLonEndList[fcstHr]
#hourlyLatBegList=[latBeg]
#hourlyLonBegList=[lonBeg]
#hourlyLatEndList=[latEnd]
#hourlyLonEndList=[lonEnd]

for j in range(len(hourlyLatBegList)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegList[j], hourlyLonBegList[j],
                                        hourlyLatEndList[j], hourlyLonEndList[j])
    axs.plot(lonList[0],latList[0],'o',transform=plotProj,color='green',alpha=0.5)
    axs.plot(lonList[-1],latList[-1],'o',transform=plotProj,color='green',alpha=0.5)
    for i in range(len(latList)-1):
        axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='green',alpha=0.5)

# collect the latitude and longitude values along cross-section from get_xsect(), for some
# arbitrary cross-section data (we aren't using the data so it doesn't really matter)
xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), latBeg, lonBeg, latEnd, lonEnd)
# plot end-points of cross section
axs.plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs.plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
# along cross-section, plot segments defined by latList, lonList
for i in range(len(latList)-1):
    axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')        

fig.savefig('fig_tank/cross_section_plan_hgtPert.png',bbox_inches='tight',facecolor='white')


# In[57]:


# Generate cross-section lines:
#
# At discrete time-intervals, plot the sea-level pressure perturbation and 850-500 hPa thickness
# Align cross-section centered on SLP minimum and oriented 90 degrees to the thickness gradient at center
# Constrain cross-section segment-length to roughly the same length as the along-gradient cross-sections
#
sampleHrs=[0., 6., 12., 18., 24., 30., 36.]
sampleLatBegList=[57., 58., 60., 60., 59., 56., 52.]
sampleLonBegList=[-161.5, -144., -140., -133.75, -125.5, -120., -118.]
sampleLatEndList=[36., 37., 39., 38., 36., 34., 33.]
sampleLonEndList=[-153.5, -152., -150., -146.75, -143.5, -138., -137.]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndList=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndList=f(np.arange(37.))


fig, axs = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})

for fcstHr in [36]:
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
    unpSlp = get_wrf_slp(unpHdl)
    posSlp = get_wrf_slp(posHdl)
    unpH850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    unpH500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                              vert=wrf.getvar(unpHdl,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    
    #slprng=np.arange(900.,1030.1,2.)
    slprng=np.arange(-18.,18.1,2.)
    smask=np.ones(np.size(slprng),dtype='bool')
    smask[np.where(slprng==0.)]=False
    
    ax=axs
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[posSlp-unpSlp,unpH500-unpH850],
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
    ax.set_title('SLP and 850-500 thickness')
# presentation cross-section values
# latBeg = 47.5
# lonBeg = -98.5
# latEnd = 28.5
# lonEnd = -70.
# modified cross-section values
#latBeg = 52.
#lonBeg = -118.
#latEnd = 33.
#lonEnd = -137.
# final(?) cross-section values
latBeg=hourlyLatBegList[fcstHr]
lonBeg=hourlyLonBegList[fcstHr]
latEnd=hourlyLatEndList[fcstHr]
lonEnd=hourlyLonEndList[fcstHr]
#hourlyLatBegList=[latBeg]
#hourlyLonBegList=[lonBeg]
#hourlyLatEndList=[latEnd]
#hourlyLonEndList=[lonEnd]

for j in range(len(hourlyLatBegList)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegList[j], hourlyLonBegList[j],
                                        hourlyLatEndList[j], hourlyLonEndList[j])
    axs.plot(lonList[0],latList[0],'o',transform=plotProj,color='green',alpha=0.5)
    axs.plot(lonList[-1],latList[-1],'o',transform=plotProj,color='green',alpha=0.5)
    for i in range(len(latList)-1):
        axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='green',alpha=0.5)

# collect the latitude and longitude values along cross-section from get_xsect(), for some
# arbitrary cross-section data (we aren't using the data so it doesn't really matter)
xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), latBeg, lonBeg, latEnd, lonEnd)
# plot end-points of cross section
axs.plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
axs.plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
# along cross-section, plot segments defined by latList, lonList
for i in range(len(latList)-1):
    axs.plot((lonList[i],lonList[i+1]),(latList[i],latList[i+1]),transform=plotProj,color='magenta')        

fig.savefig('fig_tank/cross_section_plan_hgtPert.png',bbox_inches='tight',facecolor='white')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential temperature (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[59]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[60]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# geop. height (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[61]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[62]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# omega (interpolated to the unperturbed sigma-levels). Plan-section plot is 250 hPa unperturbed geop. hgt and wind
# speed with perturbation temperature advection by the geostrophic wind at a chosen interpolation level (shaded)
for fcstHr in [0,6,12,18,24,30,36]:
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


# In[202]:


# NOTE: There may be a significant contribution to perturbation PV in the direct west-east direction, based
#       on the plots of geostrophic temperature advection and their location within the cyclonic shear. The
#       large negative perturbations to temp adv will drive downward vertical motion, drawing down the dynamic
#       tropopause (plan-section plot of perturbation potential temperature along dynamic trop?), which can
#       feed the upper front PV intrusion into the middle troposphere. Consider west-east cross sections through
#       the zone of temp adv differences.


# In[206]:


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


# In[207]:


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


# In[208]:


for fcstHr in range(13):
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,6),subplot_kw={'projection':datProj})
    lat1=hourlyLatBegList[fcstHr]
    lon1=hourlyLonBegList[fcstHr]
    lat2=hourlyLatEndList[fcstHr]
    lon2=hourlyLonEndList[fcstHr]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
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


# In[16]:


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





# In[258]:


# For a selected forecast time, plot the unperturbed potential temperature of the 2 PVU isosurface and the
# perturbation potential temperature of the 2 PVU isosurface
for fcstHr in [0]:
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# PICK UP UPDATES TO CROSS-SECTION ROUTINES HERE


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




