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


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/negative/uvTq/ptdi19/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
d10Dir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/10day_files/'
dtInit = datetime.datetime(2019, 11, 25, 12)
#dtAvgBeg = datetime.datetime(2020, 3, 2, 12)
#dtAvgEnd = datetime.datetime(2019, 11, 27, 0)


# In[4]:


# For a selected forecast time, plot the sea-level pressure, 850-500 hPa thickness, and precipitation
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


# In[8]:


x=np.asarray(get_wrf_slp(wrfHdl))
plt.contourf(x[60:110,160:190])
plt.colorbar()
plt.show()


# In[9]:


np.min(x[60:110,160:190])


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


# In[125]:


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
hourlyLatBegListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndListAlongShear=f(np.arange(37.))


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
# test cross-section values
latBeg = 49.5
lonBeg = -165.
latEnd = 45.5
lonEnd = -135.
# replace test-values with committed values
latBeg=hourlyLatBegListAlongShear[fcstHr]
lonBeg=hourlyLonBegListAlongShear[fcstHr]
latEnd=hourlyLatEndListAlongShear[fcstHr]
lonEnd=hourlyLonEndListAlongShear[fcstHr]

for j in range(len(hourlyLatBegListAlongShear)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegListAlongShear[j], 
                                        hourlyLonBegListAlongShear[j], hourlyLatEndListAlongShear[j],
                                        hourlyLonEndListAlongShear[j])
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


# In[126]:


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
hourlyLatBegListAcrossShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegListAcrossShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndListAcrossShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndListAcrossShear=f(np.arange(37.))


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
# test cross-section values
#latBeg = 52.
#lonBeg = -118.
#latEnd = 33.
#lonEnd = -137.
# replace test-values with committed values
latBeg=hourlyLatBegListAcrossShear[fcstHr]
lonBeg=hourlyLonBegListAcrossShear[fcstHr]
latEnd=hourlyLatEndListAcrossShear[fcstHr]
lonEnd=hourlyLonEndListAcrossShear[fcstHr]

for j in range(len(hourlyLatBegListAcrossShear)):
    xSect, latList, lonList = get_xsect(unpHdl, wrf.getvar(unpHdl,'z'), hourlyLatBegListAcrossShear[j],
                                        hourlyLonBegListAcrossShear[j], hourlyLatEndListAcrossShear[j],
                                        hourlyLonEndListAcrossShear[j])
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





# In[142]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential temperature (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [0,6,12,18]:
    latBegList = [hourlyLatBegListAlongShear[fcstHr]]
    lonBegList = [hourlyLonBegListAlongShear[fcstHr]]
    latEndList = [hourlyLatEndListAlongShear[fcstHr]]
    lonEndList = [hourlyLonEndListAlongShear[fcstHr]]
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
                                 xLineColorList=['black','red']
                                 )
    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[171]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# potential vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [6]:
    latBegList = [hourlyLatBegListAlongShear[fcstHr],hourlyLatBegListAcrossShear[fcstHr]]
    lonBegList = [hourlyLonBegListAlongShear[fcstHr],hourlyLonBegListAcrossShear[fcstHr]]
    latEndList = [hourlyLatEndListAlongShear[fcstHr],hourlyLatEndListAcrossShear[fcstHr]]
    lonEndList = [hourlyLonEndListAlongShear[fcstHr],hourlyLonEndListAcrossShear[fcstHr]]
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
    def right_panel(ax, payloadTuple):
        unpHdl = payloadTuple[0]
        ptdHdl = payloadTuple[1]
        # get latitude and longitude
        lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
        lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
        # fix longitude to 0 to 360 degree format
        fix = np.where(lon < 0.)
        lon[fix] = lon[fix] + 360.
        # get data and plot projections
        datProj = gen_cartopy_proj(unpHdl)
        plotProj = ccrs.PlateCarree()
        # interpolate heights to 850 and 500 hPa
        z850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=85000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        z500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                               vert=wrf.getvar(unpHdl,'p'),
                               desiredlev=50000.,
                               missing=np.nan,
                               squeeze=True,
                               meta=False)
        # compute 850-500 thickness
        thk = z500 - z850
        # get SLP
        unpSlp = get_wrf_slp(unpHdl)
        ptdSlp = get_wrf_slp(ptdHdl)
        # generate figure on ax
        slprng=np.arange(-24.,24.1,1.).astype('float16')
        slprng=slprng[np.where(slprng!=0.)]
        thkrng=np.arange(3700.,4500.1,50.)
        ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                                lat=lat,
                                                lon=lon,
                                                contVariableList=[ptdSlp-unpSlp,thk],
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl)),
                                 presLevMin=10000.,
                                 xSectTitleStr=dtFcstStr + ' ({:d} hrs) perturbed potential vorticity'.format(fcstHr),
                                 xLineColorList=['black','red']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[136]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# geop. height (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [18]:
    latBegList = [hourlyLatBegListAlongShear[fcstHr], hourlyLatBegListAcrossShear[fcstHr]]
    lonBegList = [hourlyLonBegListAlongShear[fcstHr], hourlyLonBegListAcrossShear[fcstHr]]
    latEndList = [hourlyLatEndListAlongShear[fcstHr], hourlyLatEndListAcrossShear[fcstHr]]
    lonEndList = [hourlyLonEndListAlongShear[fcstHr], hourlyLonEndListAcrossShear[fcstHr]]
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
                                 xLineColorList=['black','red']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[72]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# vorticity (interpolated to the unperturbed sigma-levels). Plan-section plot is SLP and 850-500 hPa
# thickness.
for fcstHr in [18]:
    latBegList = [hourlyLatBegListAlongShear[fcstHr], hourlyLatBegListAcrossShear[fcstHr]]
    lonBegList = [hourlyLonBegListAlongShear[fcstHr], hourlyLonBegListAcrossShear[fcstHr]]
    latEndList = [hourlyLatEndListAlongShear[fcstHr], hourlyLatEndListAcrossShear[fcstHr]]
    lonEndList = [hourlyLonEndListAlongShear[fcstHr], hourlyLonEndListAcrossShear[fcstHr]]
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
                                 xLineColorList=['black','red']
                                )

    print('hour {:d}'.format(fcstHr))
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[73]:


# For a selected time, plot the cross-section of unperturbed potential temperature and wind speed, and perturbation
# omega (interpolated to the unperturbed sigma-levels). Plan-section plot is 250 hPa unperturbed geop. hgt and wind
# speed with perturbation temperature advection by the geostrophic wind at a chosen interpolation level (shaded)
for fcstHr in [18]:
    latBegList = [hourlyLatBegListAlongShear[fcstHr], hourlyLatBegListAcrossShear[fcstHr]]
    lonBegList = [hourlyLonBegListAlongShear[fcstHr], hourlyLonBegListAcrossShear[fcstHr]]
    latEndList = [hourlyLatEndListAlongShear[fcstHr], hourlyLatEndListAcrossShear[fcstHr]]
    lonEndList = [hourlyLonEndListAlongShear[fcstHr], hourlyLonEndListAcrossShear[fcstHr]]
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
                                 xLineColorList=['black','red']
                                )

    print('hour {:d}'.format(fcstHr))
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


# In[15]:


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
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/negative/uvTq'
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


# In[20]:


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
    plt.plot(pInitTOT, eta, color='black', linewidth=2.0)
    plt.plot((pInitTOT + pInitQE), eta, color='black', linewidth=2.0, linestyle='dotted')
    plt.plot(pFcstTOT, eta, color='orange', linewidth=2.0)
    plt.legend(['norm init (mul. 5)', 'norm init + QE (mul. 5)', 'norm final'])
    plt.gca().invert_yaxis()
    plt.show()
    return
    
dataDir1 = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq'
dataDir2 = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/negative/uvTq'

unpInitHdl1 = Dataset(dataDir1 + '/wrfout_d01_unpi00')
ptdInitHdl1 = Dataset(dataDir1 + '/wrfout_d01_ptdi14')
unpInitHdl2 = Dataset(dataDir2 + '/wrfout_d01_unpi00')
ptdInitHdl2 = Dataset(dataDir2 + '/wrfout_d01_ptdi19')
generate_figure_panel(unpInitHdl1, ptdInitHdl1, unpInitHdl2, ptdInitHdl2)


# In[22]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/negative/uvTq/ptdi19/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
dtInit = datetime.datetime(2019, 11, 25, 12)


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
    slprng=np.arange(900.,1016.1,4.)
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
    
    


# In[40]:


ux = wrf.interplevel(field3d=unpUwd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=20000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
vx = wrf.interplevel(field3d=unpVwd,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=25000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)

shd=plt.contourf(unpDynThta,30)
plt.contour(np.sqrt(ux**2. + vx**2.),colors='w')
plt.contour(unpDynThta,[-999., 335.],colors='r')
plt.colorbar(mappable=shd)
plt.show()


# In[28]:


# For a selected forecast time, plot the 250/350/450 hPa geopotential heights, wind speed, and perturbation wind speed
for fcstHr in [6]:
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


# In[26]:


# For a selected forecast time, plot the 500 hPa geopotential heights, perturbation vorticity and heights
for fcstHr in [6,12,18,24,30,36]:
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


# In[67]:


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
    # compute unperturbed wind
    u,v = get_uvmet(unpHdl)
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
    # interpolate wind compontents to 335 K surface (roughly jet-level)
    unpDynUwd = wrf.interplevel(field3d=u,
                                vert=get_wrf_th(unpHdl),
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpDynVwd = wrf.interplevel(field3d=v,
                                vert=get_wrf_th(unpHdl),
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute wind speed and total deformation on 335 K surface
    unpDynSpd = np.sqrt(unpDynUwd**2. + unpDynVwd**2.)
    dUdX, dUdY = get_wrf_grad(unpHdl, unpDynUwd)
    dVdX, dVdY = get_wrf_grad(unpHdl, unpDynVwd)
    unpDynShr = dVdX + dUdY
    unpDynStr = dUdX - dVdY
    unpDynDef = np.sqrt(unpDynShr**2. + unpDynStr**2.)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    #
    dynrng=np.arange(270.,375.1,5.)
    #
    shdrng=np.arange(-30.,30.1,4.).astype('float16')
    negMask=np.ones(np.shape(shdrng),dtype='bool')
    negMask[np.where(shdrng<=0.)] = False
    posMask=np.ones(np.shape(shdrng),dtype='bool')
    posMask[np.where(shdrng>=0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=None,
                                            contIntervalList=None, 
                                            contColorList=None,
                                            contLineThicknessList=None,
                                            shadVariable=unpDynThta,
                                            shadInterval=dynrng,
                                            shadAlpha=1.0,
                                            datProj=datProj,
                                            plotProj=plotProj,
                                            shadCmap='gray',
                                            uVecVariable=None,
                                            vVecVariable=None,
                                            vectorThinning=None,
                                            vecColor=None,
                                            figax=ax)
    #ax.contourf(lon,lat,ptdDynThta-unpDynThta,shdrng[negMask],vmin=np.min(shdrng),vmax=np.max(shdrng),cmap='seismic',transform=plotProj)
    #ax.contourf(lon,lat,ptdDynThta-unpDynThta,shdrng[posMask],vmin=np.min(shdrng),vmax=np.max(shdrng),cmap='seismic',transform=plotProj)
    ax.contour(lon,lat,unpDynSpd,colors='yellow',linewidths=1.5)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpDynThta-ptdDynThta,ptdDynThta-unpDynThta, unpDynSpd],
                                            contIntervalList=[np.arange(2.,24.1,4.), np.arange(2.,25.1,4.), [36., 54., 72.]], 
                                            contColorList=['#0095ff','#fc0345','#ffd500'],
                                            contLineThicknessList=[1.5,1.5,1.5],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) perturbation pot. temp. on 2 PVU isosurface'.format(fcstHr))
    # add contour labels to slp
    #ax.clabel(cons[0],levels=slprng[::2])
    # save file
    fcstHrStr=str(fcstHr).zfill(2)
    fig.savefig('fig_tank/f'+fcstHrStr+'.png',bbox_inches='tight',facecolor='white')


# In[121]:


def generate_figure_panel(unpHdl, ptdHdl):
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
    # interpolate wind compontents to 335 K surface (roughly jet-level)
    unpDynUwd = wrf.interplevel(field3d=u,
                                vert=get_wrf_th(unpHdl),
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpDynVwd = wrf.interplevel(field3d=v,
                                vert=get_wrf_th(unpHdl),
                                desiredlev=335.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute wind speed and total deformation on 335 K surface
    unpDynSpd = np.sqrt(unpDynUwd**2. + unpDynVwd**2.)
    dUdX, dUdY = get_wrf_grad(unpHdl, unpDynUwd)
    dVdX, dVdY = get_wrf_grad(unpHdl, unpDynVwd)
    unpDynShr = dVdX + dUdY
    unpDynStr = dUdX - dVdY
    # compute unperturbed sea-level pressure
    unpSlp = get_wrf_slp(unpHdl)
    unpDynDef = np.sqrt(unpDynShr**2. + unpDynStr**2.)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    #
    dynrng = np.arange(270., 375.1, 5.)
    #
    shdrng = np.arange(-30.,30.1,4.).astype('float16')
    negMask = np.ones(np.shape(shdrng),dtype='bool')
    negMask[np.where(shdrng<=0.)] = False
    posMask = np.ones(np.shape(shdrng),dtype='bool')
    posMask[np.where(shdrng>=0.)] = False
    spdrng=[36., 54., 72., 90.]
    slprng=np.arange(1004., 1024.1, 4.)
    #
    shd=ax.contourf(lon, lat, unpDynThta, levels=dynrng, cmap='gray', vmin=np.min(dynrng), vmax=np.max(dynrng), transform=plotProj)
    ax.contourf(lon, lat, ptdDynThta-unpDynThta, levels=shdrng[posMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    ax.contourf(lon, lat, ptdDynThta-unpDynThta, levels=shdrng[negMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    #con1=ax.contour(lon, lat, unpDynSpd, levels=spdrng, colors='#23a625', transform=plotProj, linewidths=3.)
    #con2=ax.contour(lon, lat, unpDynSpd, levels=spdrng, colors='#ffd500', transform=plotProj, linewidths=1.)
    con1=ax.contour(lon, lat, unpSlp, levels=slprng, colors='black', transform=plotProj, linewidths=3.)
    con2=ax.contour(lon, lat, unpSlp, levels=slprng, colors='#ffd500', transform=plotProj, linewidths=1.)
    
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
    ax.clabel(con1,levels=slprng)
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
    plt.show()


# In[123]:


fcstHr = 0
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = negDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
generate_figure_panel(unpHdl, ptdHdl)

fcstHr = 18
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = negDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
generate_figure_panel(unpHdl, ptdHdl)


# In[143]:


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



# In[157]:


# FIG S: Cross-sections
# a) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 400-700 hPa for
#    most-intense simulation (along-shear cross-section)
# b) Cross-section perturbation temperature and plan-section mean wind speed and temperature perturbation 400-700 hPa for
#    most-intense simulation (along-shear cross-section)
# c) Cross-section perturbation wind speed and plan-section mean wind speed and wind speed perturbation 400-700 hPa for
#    most-intense simulation (along-shear cross-section)
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
    shdRange = np.arange(-4., 4.01, 0.5).astype('float16')
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
    xSectShadInterval=np.arange(-10., 10.01, 1.0)
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
#
# begin
#
if __name__ == "__main__":
    # define directories for unperturbe and most-intense (pos) simulations
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initialization time and forecast time
    dtInit = datetime.datetime(2019, 11, 25, 12)
    # FIG Sa: 0-hr forecast most-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (along-shear)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 42.75
    lonBeg = -172.5
    latEnd = 48.75
    lonEnd = -142.5
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGS_panel_A')
    # FIG Sb: 6-hr forecast most-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (along-shear)
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 49.5
    lonBeg = -165.0
    latEnd = 45.5
    lonEnd = -135.0
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGS_panel_B')
    # FIG Sc: 12-hr forecast most-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (along-shear)
    fcstHr = 12
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 51.0
    lonBeg = -162.0
    latEnd = 46.0
    lonEnd = -132.0
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGS_panel_C')
    # FIG Sc: 18-hr forecast most-intense simulation cross-section of perturbation temperature and 400-700 hPa mean
    # temperature perturbation (along-shear)
    fcstHr = 18
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # open WRF files and define file-handles
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 52.0
    lonBeg = -158.0
    latEnd = 43.5
    lonEnd = -128.0
    generate_cross_temperature_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'black', 'FIGS_panel_D')
#
# end
#


# In[144]:


sampleHrs=[0., 6., 12., 18., 24., 30., 36.]
sampleLatBegList=[42.75, 49.5, 51., 52., 52., 50.25, 49.5]
sampleLonBegList=[-172.5, -165., -162., -158., -154., -149., -146.]
sampleLatEndList=[48.75, 45.5, 46., 43.5, 41., 38.75, 35.]
sampleLonEndList=[-142.5, -135., -132., -128., -124., -119., -116.]
# Linearly interpolate for cross-section beg/end points at hourly segments
f = interp1d(sampleHrs, sampleLatBegList)
hourlyLatBegListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonBegList)
hourlyLonBegListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLatEndList)
hourlyLatEndListAlongShear=f(np.arange(37.))
f = interp1d(sampleHrs, sampleLonEndList)
hourlyLonEndListAlongShear=f(np.arange(37.))


# In[154]:


i=18
print(hourlyLatBegListAlongShear[i])
print(hourlyLonBegListAlongShear[i])
print(hourlyLatEndListAlongShear[i])
print(hourlyLonEndListAlongShear[i])


# In[317]:


# Figure L: Cross-section of perturbation omega, with plan-section perturbation mean 400-700 hPa geopotential height
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
#
# begin
#
if __name__ == "__main__":
    # define unperturbed and most-intense (positive) perturbed file subdirectories
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2019, 11, 25, 12)
    
    # FIG Ta: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.0
    lonBeg = -166.0
    latEnd = 38.0
    lonEnd = -150.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGT_panel_A')
    
    # FIG Tb: most-intense simulation 12-hr cross section of perturbation omega
    fcstHr = 12
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 63.0
    lonBeg = -152.0
    latEnd = 37.0
    lonEnd = -154.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGT_panel_B')
    
    # FIG Tc: most-intense simulation 18-hr cross section of perturbation omega
    fcstHr = 18
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 64.0
    lonBeg = -143.0
    latEnd = 35.0
    lonEnd = -149.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGT_panel_C')
    
    # FIG Ld: most-intense simulation 24-hr cross section of perturbation omega
    fcstHr = 24
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.5
    lonBeg = -127.0
    latEnd = 35.0
    lonEnd = -147.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGT_panel_D')
#
# end
#


# In[218]:


# Figure L: Cross-section of perturbation omega, with plan-section perturbation mean 400-700 hPa geopotential height
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
                                 xSectShadVariable=ptdPvor_int-unpPvor,
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
#
# begin
#
if __name__ == "__main__":
    # define unperturbed and most-intense (positive) perturbed file subdirectories
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2019, 11, 25, 12)
    
    # FIG Ua: most-intense simulation 0-hr cross section of perturbation omega
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 60.0
    lonBeg = -171.0
    latEnd = 38.0
    lonEnd = -155.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
    
    # FIG Ua: most-intense simulation 3-hr cross section of perturbation omega
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.0
    lonBeg = -168.0
    latEnd = 38.0
    lonEnd = -154.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
    
    # FIG Ua: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 62.0
    lonBeg = -166.0
    latEnd = 38.0
    lonEnd = -150.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
#
# end
#


# In[217]:


# Type-A case doesn't derive early upper-trop PV from enhanced cold-advection and
# subsidence beneath the jet core like in the Type-B case. Instead, a significant
# portion of the initial PV is inserted directly, roughly at the 310 K / 300 hPa
# level beneath the primary jet.


# In[273]:


# Figure L: Cross-section of perturbation omega, with plan-section perturbation mean 400-700 hPa geopotential height
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
                                 xSectShadVariable=ptdPvor_int-unpPvor,
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
#
# begin
#
if __name__ == "__main__":
    # define unperturbed and most-intense (positive) perturbed file subdirectories
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2019, 11, 25, 12)
    
    # FIG Ua: most-intense simulation 0-hr cross section of perturbation omega
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 48.0
    lonBeg = -179.0
    latEnd = 56.0
    lonEnd = -147.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGV_panel_A')
    
    # FIG Ua: most-intense simulation 3-hr cross section of perturbation omega
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 47.0
    lonBeg = -175.0
    latEnd = 57.0
    lonEnd = -138.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGV_panel_A')
    
    # FIG Ua: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 46.0
    lonBeg = -173.0
    latEnd = 55.0
    lonEnd = -138.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGV_panel_A')
#
# end
#


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[258]:


def generate_figure_panel(unpHdl, ptdHdl):
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
    # interpolate potential vorticity to 300 hPa surface
    unpPvor300 = wrf.interplevel(field3d=unpPvor,
                                 vert=wrf.getvar(unpHdl,'p'),
                                 desiredlev=30000.0,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    ptdPvor300 = wrf.interplevel(field3d=ptdPvor,
                                 vert=wrf.getvar(ptdHdl,'p'),
                                 desiredlev=30000.0,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
    # interpolate wind compontents to 300 hPa surface
    unpUwd300 = wrf.interplevel(field3d=u,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=30000.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpVwd300 = wrf.interplevel(field3d=v,
                                vert=wrf.getvar(unpHdl,'p'),
                                desiredlev=30000.0,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # compute wind speed and total deformation on 300 hPa surface
    unpSpd300 = np.sqrt(unpUwd300**2. + unpVwd300**2.)
    dUdX, dUdY = get_wrf_grad(unpHdl, unpUwd300)
    dVdX, dVdY = get_wrf_grad(unpHdl, unpVwd300)
    unpShr300 = dVdX + dUdY
    unpStr300 = dUdX - dVdY
    # compute unperturbed sea-level pressure
    unpSlp = get_wrf_slp(unpHdl)
    unpDef300 = np.sqrt(unpShr300**2. + unpStr300**2.)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    #
    dynrng = np.arange(-2., 10.1, 0.5)
    #
    shdrng = np.arange(-4.,4.01,0.5).astype('float16')
    negMask = np.ones(np.shape(shdrng),dtype='bool')
    negMask[np.where(shdrng<=0.)] = False
    posMask = np.ones(np.shape(shdrng),dtype='bool')
    posMask[np.where(shdrng>=0.)] = False
    spdrng=[36., 54., 72., 90.]
    defrng=1.0E-05*np.arange(5.,75.1,10.)
    slprng=np.arange(1004., 1024.1, 4.)
    #
    shd=ax.contourf(lon, lat, unpPvor300, levels=dynrng, cmap='gray', vmin=np.min(dynrng), vmax=np.max(dynrng), transform=plotProj)
    ax.contourf(lon, lat, ptdPvor300-unpPvor300, levels=shdrng[posMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    ax.contourf(lon, lat, ptdPvor300-unpPvor300, levels=shdrng[negMask], cmap='seismic', vmin=np.min(shdrng), vmax=np.max(shdrng), transform=plotProj)
    con1=ax.contour(lon, lat, unpDynSpd, levels=spdrng, colors='black', transform=plotProj, linewidths=3.)
    con2=ax.contour(lon, lat, unpDynSpd, levels=spdrng, colors='#ffd500', transform=plotProj, linewidths=1.)
    #con1=ax.contour(lon, lat, unpSlp, levels=slprng, colors='black', transform=plotProj, linewidths=3.)
    #con2=ax.contour(lon, lat, unpSlp, levels=slprng, colors='#ffd500', transform=plotProj, linewidths=1.)
    #con1=ax.contour(lon, lat, unpDef300, levels=defrng, colors='black', transform=plotProj, linewidths=3.)
    #con2=ax.contour(lon, lat, unpDef300, levels=defrng, colors='#ffd500', transform=plotProj, linewidths=1.)
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
    plt.show()


# In[260]:


fcstHr = 0
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
x=generate_figure_panel(unpHdl, ptdHdl)

fcstHr = 4
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
x=generate_figure_panel(unpHdl, ptdHdl)

fcstHr = 8
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
x=generate_figure_panel(unpHdl, ptdHdl)

fcstHr = 12
#define forecast datetime stamp
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
# define WRF forecast files and open netCDF4 file-handles
unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFile = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl = Dataset(unpFile)
ptdHdl = Dataset(ptdFile)
generate_figure_panel(unpHdl, ptdHdl)


# In[318]:


# Figure L: Cross-section of perturbation omega, with plan-section perturbation mean 400-700 hPa geopotential height
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
    # compute unperturbed sea-level pressure
    unpSlp = get_wrf_slp(unpHdl)
    #
    dynrng = np.arange(-2., 10.1, 0.5)
    #
    shdrng = np.arange(-4.,4.01,0.5).astype('float16')
    negMask = np.ones(np.shape(shdrng),dtype='bool')
    negMask[np.where(shdrng<=0.)] = False
    posMask = np.ones(np.shape(shdrng),dtype='bool')
    posMask[np.where(shdrng>=0.)] = False
    spdrng=[36., 54., 72., 90.]
    defrng=1.0E-05*np.arange(5.,75.1,10.)
    slprng=np.arange(1004., 1024.1, 4.)
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
                                 planSectPlotTuple=(right_panel, (unpHdl, ptdHdl, 30000.)),
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
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
    ptdDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
    # define initial-condition datetime
    dtInit = datetime.datetime(2019, 11, 25, 12)
    
    # FIG Ua: most-intense simulation 0-hr cross section of perturbation omega
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 63.0
    lonBeg = -164.0
    latEnd = 40.0
    lonEnd = -167.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
    
    # FIG Ua: most-intense simulation 3-hr cross section of perturbation omega
    fcstHr = 3
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 63.0
    lonBeg = -162.0
    latEnd = 40.0
    lonEnd = -165.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
    
    # FIG Ua: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 6
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 63.0
    lonBeg = -163.0
    latEnd = 40.0
    lonEnd = -157.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
    
    # FIG Ua: most-intense simulation 6-hr cross section of perturbation omega
    fcstHr = 9
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = ptdDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    latBeg = 63.0
    lonBeg = -160.0
    latEnd = 40.0
    lonEnd = -150.0
    generate_figure_panel(unpHdl, ptdHdl, latBeg, lonBeg, latEnd, lonEnd, 'FIGU_panel_A')
#
# end
#


# In[278]:


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
def generate_figure_panel(unpHdl1, ptdHdl1, unpHdl2, ptdHdl2, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format 
    lat = np.asarray(unpHdl1.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl1.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(unpHdl1)
    plotProj = ccrs.PlateCarree()
    # extract unperturbed sea-level pressure
    unpSlp = np.asarray(get_wrf_slp(unpHdl2)).squeeze()
    # interpolate unperturbed heights to 850 and 500 hPa
    unpZ850 = wrf.interplevel(field3d=wrf.getvar(unpHdl2,'z'),
                              vert=wrf.getvar(unpHdl2,'p'),
                              desiredlev=85000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    unpZ500 = wrf.interplevel(field3d=wrf.getvar(unpHdl2,'z'),
                              vert=wrf.getvar(unpHdl2,'p'),
                              desiredlev=50000.,
                              missing=np.nan,
                              squeeze=True,
                              meta=False)
    # compute unperturbed 850-500 thickness
    unpThk = unpZ500 - unpZ850
    # compute precipitation
    unpPrcp1 = np.asarray(unpHdl1.variables['RAINC']).squeeze() + np.asarray(unpHdl1.variables['RAINNC']).squeeze()
    ptdPrcp1 = np.asarray(ptdHdl1.variables['RAINC']).squeeze() + np.asarray(ptdHdl1.variables['RAINNC']).squeeze()
    unpPrcp2 = np.asarray(unpHdl2.variables['RAINC']).squeeze() + np.asarray(unpHdl2.variables['RAINNC']).squeeze()
    ptdPrcp2 = np.asarray(ptdHdl2.variables['RAINC']).squeeze() + np.asarray(ptdHdl2.variables['RAINNC']).squeeze()
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
                                            shadVariable=(ptdPrcp2-ptdPrcp1)-(unpPrcp2-unpPrcp1),
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


# In[281]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/unperturbed/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/nov2019/R_mu/positive/uvTq/ptdi24/'
dtInit = datetime.datetime(2019, 11, 25, 12)

fcstHr1 = 0
fcstHr2 = 1

dtFcst = dtInit + datetime.timedelta(hours=fcstHr1)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl1 = Dataset(unpFileFcst)
ptdHdl1 = Dataset(ptdFileFcst)

dtFcst = dtInit + datetime.timedelta(hours=fcstHr2)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdl2 = Dataset(unpFileFcst)
ptdHdl2 = Dataset(ptdFileFcst)

generate_figure_panel(unpHdl1, ptdHdl1, unpHdl2, ptdHdl2, 'test')


# In[ ]:




