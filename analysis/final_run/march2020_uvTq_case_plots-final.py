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
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import compute_inverse_laplacian
from analysis_dependencies import plan_section_plot
from analysis_dependencies import cross_section_plot
from analysis_dependencies import cross_section_diffplot
from analysis_dependencies import gen_time_avg
from analysis_dependencies import interpolate_sigma_levels
import datetime
import wrf
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr


# In[2]:


unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
negDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/negative/uvTq/ptdi14/'
posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
d10Dir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/10day_files/'
dtInit = datetime.datetime(2020, 3, 6, 12)
dtAvgBeg = datetime.datetime(2020, 3, 2, 12)
dtAvgEnd = datetime.datetime(2020, 3, 12, 12)


# In[3]:


fcstHr = 24
dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtInitStr = datetime.datetime.strftime(dtInit,'%Y-%m-%d_%H:00:00')
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
unpWRFInputFile = unpDir + 'wrfinput_d01'
negWRFInputFile = negDir + 'wrfinput_d01'
posWRFInputFile = posDir + 'wrfinput_d01'
avgFileNameBeg = 'wrfinput_d01_'
avgTimeStamps = []
dt = dtAvgBeg
while dt <= dtAvgEnd:
    avgTimeStamps.append(datetime.datetime.strftime(dt,'%Y-%m-%d_%H:00:00'))
    dt = dt + datetime.timedelta(hours=6)
unpFileInit = unpDir + 'wrfout_d01_' + dtInitStr
negFileInit = negDir + 'wrfout_d01_' + dtInitStr
posFileInit = posDir + 'wrfout_d01_' + dtInitStr
unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
negFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
posFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
unpHdlInit = Dataset(unpFileInit)
negHdlInit = Dataset(negFileInit)
posHdlInit = Dataset(posFileInit)
unpHdlFcst = Dataset(unpFileFcst)
negHdlFcst = Dataset(negFileFcst)
posHdlFcst = Dataset(posFileFcst)


# In[4]:


# draw lat, lon, and projection data from unpHdlInit, draw pressure from all files for interpolation
# all other fields are computed on an as-needed basis
lat = np.asarray(unpHdlInit.variables['XLAT']).squeeze()
lon = np.asarray(unpHdlInit.variables['XLONG']).squeeze()
fix = np.where(lon < 0.)
lon[fix] = lon[fix] + 360.

datProj = gen_cartopy_proj(unpHdlInit)
plotProj = ccrs.PlateCarree()

unpPInit = np.asarray(wrf.getvar(unpHdlInit,'p')).squeeze()
negPInit = np.asarray(wrf.getvar(negHdlInit,'p')).squeeze()
posPInit = np.asarray(wrf.getvar(posHdlInit,'p')).squeeze()
unpPFcst = np.asarray(wrf.getvar(unpHdlFcst,'p')).squeeze()
negPFcst = np.asarray(wrf.getvar(negHdlFcst,'p')).squeeze()
posPFcst = np.asarray(wrf.getvar(posHdlFcst,'p')).squeeze()


# In[5]:


# plan-section figure: unperturbed sea-level pressure and 850-500 hPa thicknesses
hgt = np.asarray(wrf.getvar(unpHdlInit, 'z')).squeeze()
prs = unpPInit
hgt850 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=85000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
hgt500 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpThkInit_850_500 = hgt500 - hgt850
unpSlpInit = np.asarray(get_wrf_slp(unpHdlInit)).squeeze()

slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,7), subplot_kw={'projection' : datProj})

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
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
ax.set_title(dtInitStr + ' unperturbed sea level pressure, 850-500 hPa thickness')
# add contour labels to slp
ax.clabel(cons[0],levels=slprng[::2])


# In[6]:


# plan-section figure: perturbation sea-level pressure (left) and 850-500 thickness (right)
#                      for weakening case (top) and intensifying case (bottom)
hgt = np.asarray(wrf.getvar(posHdlInit, 'z')).squeeze()
prs = posPInit
hgt850 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=85000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
hgt500 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posThkInit_850_500 = hgt500 - hgt850
posSlpInit = np.asarray(get_wrf_slp(posHdlInit)).squeeze()

hgt = np.asarray(wrf.getvar(negHdlInit, 'z')).squeeze()
prs = negPInit
hgt850 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=85000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
hgt500 = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negThkInit_850_500 = hgt500 - hgt850
negSlpInit = np.asarray(get_wrf_slp(negHdlInit)).squeeze()

shdrng = np.arange(-30.,30.1,2.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(24,14), subplot_kw={'projection' : datProj})

#
# Top: weakening experiment slp perturbation (left) and 850-500 hPa thickness perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
                                        shadVariable=negSlpInit - unpSlpInit,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[0][0])
# add a title
axs[0][0].set_title(dtInitStr + ' weakening: sea level pressure perturbation')
# add contour labels to slp
axs[0][0].clabel(cons[0],levels=slprng[::2])

shdrng = np.arange(-200.,200.1,20.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
                                        shadVariable=negThkInit_850_500 - unpThkInit_850_500,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[0][1])
# add a title
axs[0][1].set_title(dtInitStr + ' weakening: 850-500 hPa thickness perturbation')
# add contour labels to slp
axs[0][1].clabel(cons[0],levels=slprng[::2])

#
# Bottom: intensifying experiment slp perturbation (left) and 850-500 thicknesses (right)
#

shdrng = np.arange(-30.,30.1,2.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
                                        shadVariable=posSlpInit - unpSlpInit,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[1][0])
# add a title
axs[1][0].set_title(dtInitStr + ' intensifying: sea level pressure perturbation')
# add contour labels to slp
axs[1][0].clabel(cons[0],levels=slprng[::2])

shdrng = np.arange(-200.,200.1,20.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
                                        shadVariable=posThkInit_850_500 - unpThkInit_850_500,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[1][1])
# add a title
axs[1][1].set_title(dtInitStr + ' intensifying: 850-500 hPa thickness perturbation')
# add contour labels to slp
axs[1][1].clabel(cons[0],levels=slprng[::2])


# In[7]:


# plan-section figure: unperturbed 500 hPa geopotential height, wind vectors, and vorticity
hgt = np.asarray(wrf.getvar(unpHdlInit, 'z')).squeeze()
prs = unpPInit
unpHgt500Init = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
u, v = get_uvmet(unpHdlInit)
vor = get_wrf_kinematic(unpHdlInit, 'vor')
unpVor500Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpUwd500Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpVwd500Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
hgtrng = np.arange(4900.,6550.1,45.)
vorrng = 1.0E-05 * np.arange(-40.,40.01,4.)
mask = np.ones((np.shape(vorrng)),dtype='bool')
mask[np.where(vorrng==0.)] = False

fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,7), subplot_kw={'projection' : datProj})

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt500Init],
                                        contIntervalList=[hgtrng], 
                                        contColorList=['black'],
                                        contLineThicknessList=[1.0],
                                        shadVariable=unpVor500Init,
                                        shadInterval=vorrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=unpUwd500Init,
                                        vVecVariable=unpVwd500Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        figax=ax)
# add a title
ax.set_title(dtInitStr + ' unperturbed 500 hPa geopotential height, vorticity')
# add contour labels to hgt
ax.clabel(cons[0],levels=hgtrng[::2])


# In[8]:


unpSfun500Init = compute_inverse_laplacian(unpHdlInit,unpVor500Init)

u, v = get_uvmet(posHdlInit)
vor = get_wrf_kinematic(posHdlInit, 'vor')
posVor500Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posUwd500Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posVwd500Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posSfun500Init = compute_inverse_laplacian(posHdlInit,posVor500Init)

u, v = get_uvmet(negHdlInit)
vor = get_wrf_kinematic(negHdlInit, 'vor')
negVor500Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negUwd500Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negVwd500Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=50000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negSfun500Init = compute_inverse_laplacian(negHdlInit,negVor500Init)
# plan-section figure: perturbation 500 hPa vorticity and winds (left) and 500 hPa
#                      perturbation streamfunction and winds (right) for weakening case (top) and
#                      intensifying case (bottom)

shdrng = 1.0E-05 * np.arange(-6.,6.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(4900.,6550.1,45.)


fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(24,14), subplot_kw={'projection' : datProj})

#
# Top: weakening experiment potential temperature perturbation (left) and 500 hPa vorticity and wind perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt500Init],
                                        contIntervalList=[hgtrng], 
                                        contColorList=['black'],
                                        contLineThicknessList=[1.0],
                                        shadVariable=negVor500Init - unpVor500Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd500Init - unpUwd500Init,
                                        vVecVariable=negVwd500Init - unpVwd500Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[0][0])
# add a title
axs[0][0].set_title(dtInitStr + ' weakening: 500 hPa perturbation vorticity and winds')
# add contour labels to hgt
axs[0][0].clabel(cons[0],levels=hgtrng[::2])

shdrng = 1.0E+05 * np.arange(-8.,8.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(4900.,6550.1,45.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt500Init],
                                        contIntervalList=[hgtrng], 
                                        contColorList=['black'],
                                        contLineThicknessList=[1.0],
                                        shadVariable=negSfun500Init - unpSfun500Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd500Init - unpUwd500Init,
                                        vVecVariable=negVwd500Init - unpVwd500Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[0][1])
# add a title
axs[0][1].set_title(dtInitStr + ' weakening: 500 hPa perturbation streamfunction and winds')
# add contour labels to hgt
axs[0][1].clabel(cons[0],levels=hgtrng[::2])

#
# Bottom: intensifying experiment potential temperature perturbation (left) and 500 hPa vorticity and wind perturbation (right)
#

shdrng = 1.0E-05 * np.arange(-6.,6.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(4900.,6550.1,45.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt500Init],
                                        contIntervalList=[hgtrng], 
                                        contColorList=['black'],
                                        contLineThicknessList=[1.0],
                                        shadVariable=posVor500Init - unpVor500Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=posUwd500Init - unpUwd500Init,
                                        vVecVariable=posVwd500Init - unpVwd500Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[1][0])
# add a title
axs[1][0].set_title(dtInitStr + ' intensifying: 500 hPa perturbation vorticity and winds')
# add contour labels to hgt
axs[1][0].clabel(cons[0],levels=hgtrng[::2])

shdrng = 1.0E+05 * np.arange(-8.,8.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(4900.,6550.1,45.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt500Init],
                                        contIntervalList=[hgtrng], 
                                        contColorList=['black'],
                                        contLineThicknessList=[1.0],
                                        shadVariable=posSfun500Init - unpSfun500Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=1.0,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=posUwd500Init - unpUwd500Init,
                                        vVecVariable=posVwd500Init - unpVwd500Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[1][1])
# add a title
axs[1][1].set_title(dtInitStr + ' intensifying: 500 hPa perturbation streamfunction and winds')
# add contour labels to hgt
axs[1][1].clabel(cons[0],levels=hgtrng[::2])



# In[9]:


# plan-section figure: unperturbed 300 hPa geopotential height and vorticity
u, v = get_uvmet(unpHdlInit)
hgt = np.asarray(wrf.getvar(unpHdlInit, 'z')).squeeze()
vor = get_wrf_kinematic(unpHdlInit, 'vor')
prs = unpPInit
unpVor300Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpUwd300Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpVwd300Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpHgt300Init = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpWspd300Init = np.sqrt(unpUwd300Init**2. + unpVwd300Init**2.)
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,7), subplot_kw={'projection' : datProj})

shdrng = 1.0E-05 * np.arange(-30.,30.1,6.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)
ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=unpVor300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=ax)
# add a title
ax.set_title(dtInitStr + ' 300 hPa geopotential height, isotachs, vorticity')
# add contour labels to hgt
ax.clabel(cons[0],levels=hgtrng[::2])


# In[10]:


u, v = get_uvmet(unpHdlInit)
prs = unpPInit
unpUwd300Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpVwd300Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpSfun300Init = compute_inverse_laplacian(unpHdlInit,unpVor300Init)

u, v = get_uvmet(posHdlInit)
vor = get_wrf_kinematic(posHdlInit, 'vor')
prs = posPInit
posVor300Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posUwd300Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posVwd300Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posSfun300Init = compute_inverse_laplacian(posHdlInit,posVor300Init)

u, v = get_uvmet(negHdlInit)
vor = get_wrf_kinematic(negHdlInit, 'vor')
prs = negPInit
negVor300Init = wrf.interplevel(field3d=vor,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negUwd300Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negVwd300Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=30000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negSfun300Init = compute_inverse_laplacian(negHdlInit,negVor300Init)

# plan-section figure: perturbation 300 hPa vorticity and winds (left) and 300 hPa
#                      perturbation streamfunction and winds (right) for weakening case (top) and
#                      intensifying case (bottom)

shdrng = 1.0E-05 * np.arange(-4.,4.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)


fig, axs = plt.subplots(ncols=2,nrows=2,figsize=(24,14), subplot_kw={'projection' : datProj})

#
# Top: weakening experiment 300 hPa divergence and windperturbation (left) and 500 hPa velocity potential perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=negVor300Init - unpVor300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd300Init - unpUwd300Init,
                                        vVecVariable=negVwd300Init - unpVwd300Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[0][0])
# add a title
axs[0][0].set_title(dtInitStr + ' weakening: 300 hPa perturbation vorticity and wind')
# add contour labels to hgt
axs[0][0].clabel(cons[0],levels=hgtrng[::2])

shdrng = 1.0E+05 * np.arange(-8.0,8.01,1.0)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=negSfun300Init - unpSfun300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[0][1])

# add a title
axs[0][1].set_title(dtInitStr + ' weakening: 300 hPa perturbation streamfunction and wind')
# add contour labels to hgt
axs[0][1].clabel(cons[0],levels=hgtrng[::2])

#
# Bottom: intensifying experiment potential temperature perturbation (left) and 500 hPa vorticity and wind perturbation (right)
#

shdrng = 1.0E-05 * np.arange(-4.,4.01,0.5)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=posVor300Init - unpVor300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd300Init - unpUwd300Init,
                                        vVecVariable=negVwd300Init - unpVwd300Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[1][0])
# add a title
axs[1][0].set_title(dtInitStr + ' intensifying: 300 hPa perturbation vorticity and wind')
# add contour labels to hgt
axs[1][0].clabel(cons[0],levels=hgtrng[::2])

shdrng = 1.0E+05 * np.arange(-8.0,8.01,1.0)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=posSfun300Init - unpSfun300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=axs[1][1])

# add a title
axs[1][1].set_title(dtInitStr + ' intensifying: 300 hPa perturbation streamfunction and wind')
# add contour labels to hgt
axs[1][1].clabel(cons[0],levels=hgtrng[::2])


# In[11]:


u, v = get_uvmet(posHdlInit)
prs = posPInit
spd = np.sqrt(u**2. + v**2.)
posWspd300Init = wrf.interplevel(field3d=spd,
                                 vert=prs,
                                 desiredlev=30000.,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
u, v = get_uvmet(negHdlInit)
prs = negPInit
spd = np.sqrt(u**2. + v**2.)
negWspd300Init = wrf.interplevel(field3d=spd,
                                 vert=prs,
                                 desiredlev=30000.,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
# plan-section figure: perturbation 300 hPa wind speed and winds for for weakening case (top) and
#                      intensifying case (bottom)

shdrng = np.arange(-3.5,3.51,0.25)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)


fig, axs = plt.subplots(ncols=1,nrows=2,figsize=(12,14), subplot_kw={'projection' : datProj})

#
# Top: weakening experiment 300 hPa divergence and windperturbation (left) and 500 hPa velocity potential perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=negWspd300Init - unpWspd300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd300Init - unpUwd300Init,
                                        vVecVariable=negVwd300Init - unpVwd300Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[0])
# add a title
axs[0].set_title(dtInitStr + ' weakening: 300 hPa perturbation wind speed and wind')
# add contour labels to hgt
axs[0].clabel(cons[0],levels=hgtrng[::2])

#
# Bottom: intensifying experiment 300 hPa wind speed and wind perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=posWspd300Init - unpWspd300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=posUwd300Init - unpUwd300Init,
                                        vVecVariable=posVwd300Init - unpVwd300Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[1])

# add a title
axs[1].set_title(dtInitStr + ' intensifying: 300 hPa perturbation wind speed and wind')
# add contour labels to hgt
axs[1].clabel(cons[0],levels=hgtrng[::2])


# In[12]:


shdrng = np.arange(-3.5,3.51,0.25)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,14), subplot_kw={'projection' : datProj})
ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt300Init, unpWspd300Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=negWspd300Init - unpWspd300Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd300Init - unpUwd300Init,
                                        vVecVariable=negVwd300Init - unpVwd300Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=ax)


# In[13]:


latBeg = 43.
lonBeg = -105.
latEnd = 43.
lonEnd = -80.
fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='green')
fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='green')
fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='green')
latBeg = 43.
lonBeg = -82.5
latEnd = 25.
lonEnd = -75.
fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='orange')
fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='orange')
fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='orange')
fig


# In[14]:


# cross-sections: perturbation potential temperature (intensifying)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
spd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30., 30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(spd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(get_wrf_tk(unpHdlInit),unpHdlInit),
                             xSectShadVariable2=(get_wrf_tk(posHdlInit),posHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(posHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)

# cross-sections: perturbation potential temperature (weakening)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
spd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30., 30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(spd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(get_wrf_tk(unpHdlInit),unpHdlInit),
                             xSectShadVariable2=(get_wrf_tk(negHdlInit),negHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(negHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)


# cross-sections: perturbation wind speed (intensifying)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
u,v = get_uvmet(posHdlInit)
posWspd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(unpWspd,unpHdlInit),
                             xSectShadVariable2=(posWspd,posHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(posHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)

# cross-sections: perturbation wind speed (weakening)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
u,v = get_uvmet(negHdlInit)
negWspd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(unpWspd,unpHdlInit),
                             xSectShadVariable2=(negWspd,negHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(negHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)



# cross-sections: perturbation vorticity (intensifying)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
unpVor = get_wrf_kinematic(unpHdlInit,'vor')
posVor = get_wrf_kinematic(posHdlInit,'vor')
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = 1.0E-05 * np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(unpVor,unpHdlInit),
                             xSectShadVariable2=(posVor,posHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(posHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)


# cross-sections: perturbation vorticity (weakening)
wrfHDL = unpHdlInit
latBegList = [ 43.0, 43.0]
lonBegList = [-105.0, -82.5]
latEndList = [ 43.0, 25.0]
lonEndList = [-80.0, -75.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
unpVor = get_wrf_kinematic(unpHdlInit,'vor')
negVor = get_wrf_kinematic(negHdlInit,'vor')
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = 1.0E-05 * np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(unpVor,unpHdlInit),
                             xSectShadVariable2=(negVor,negHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(negHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.
                            )
plt.show(fig)


# In[15]:


u, v = get_uvmet(unpHdlInit)
hgt = np.asarray(wrf.getvar(unpHdlInit, 'z')).squeeze()
vor = get_wrf_kinematic(unpHdlInit, 'vor')
prs = unpPInit
unpUwd350Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpVwd350Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpHgt350Init = wrf.interplevel(field3d=hgt,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
unpWspd350Init = np.sqrt(unpUwd350Init**2. + unpVwd350Init**2.)

u, v = get_uvmet(posHdlInit)
prs = posPInit
spd = np.sqrt(u**2. + v**2.)
posWspd350Init = wrf.interplevel(field3d=spd,
                                 vert=prs,
                                 desiredlev=35000.,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
posUwd350Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
posVwd350Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
u, v = get_uvmet(negHdlInit)
prs = negPInit
spd = np.sqrt(u**2. + v**2.)
negWspd350Init = wrf.interplevel(field3d=spd,
                                 vert=prs,
                                 desiredlev=35000.,
                                 missing=np.nan,
                                 squeeze=True,
                                 meta=False)
negUwd350Init = wrf.interplevel(field3d=u,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
negVwd350Init = wrf.interplevel(field3d=v,
                         vert=prs,
                         desiredlev=35000.,
                         missing=np.nan,
                         squeeze=True,
                         meta=False)
# plan-section figure: perturbation 350 hPa wind speed and winds for for weakening case (top) and
#                      intensifying case (bottom)

shdrng = np.arange(-3.5,3.51,0.25)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
hgtrng = np.arange(8200.,10000.1,90.)
spdrng = np.arange(40.,90.1,10.)


fig, axs = plt.subplots(ncols=1,nrows=2,figsize=(12,14), subplot_kw={'projection' : datProj})

#
# Top: weakening experiment 350 hPa divergence and windperturbation (left) and 500 hPa velocity potential perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt350Init, unpWspd350Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=negWspd350Init - unpWspd350Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=negUwd350Init - unpUwd350Init,
                                        vVecVariable=negVwd350Init - unpVwd350Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[0])
# add a title
axs[0].set_title(dtInitStr + ' weakening: 350 hPa perturbation wind speed and wind')
# add contour labels to hgt
axs[0].clabel(cons[0],levels=hgtrng[::2])

#
# Bottom: intensifying experiment 350 hPa wind speed and wind perturbation (right)
#

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpHgt350Init, unpWspd350Init],
                                        contIntervalList=[hgtrng, spdrng], 
                                        contColorList=['black', '#35821b'],
                                        contLineThicknessList=[1.0, 2.0],
                                        shadVariable=posWspd350Init - unpWspd350Init,
                                        shadInterval=shdrng[mask],
                                        shadAlpha=0.4,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap='seismic',
                                        uVecVariable=posUwd350Init - unpUwd350Init,
                                        vVecVariable=posVwd350Init - unpVwd350Init,
                                        vectorThinning=10,
                                        vecColor='#35821b',
                                        vectorScale=30,
                                        figax=axs[1])

# add a title
axs[1].set_title(dtInitStr + ' intensifying: 350 hPa perturbation wind speed and wind')
# add contour labels to hgt
axs[1].clabel(cons[0],levels=hgtrng[::2])


# In[ ]:





# In[16]:


shdrng = np.arange(-200.,200.1,20.)
mask = np.ones((np.shape(shdrng)),dtype='bool')
mask[np.where(shdrng==0.)] = False
slprng = np.arange(950.,1050.1,4.)
thkrng = np.arange(3700.,4500.1,25.)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9), subplot_kw={'projection' : datProj})

ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdlInit,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=[unpSlpInit, unpThkInit_850_500],
                                        contIntervalList=[slprng, thkrng], 
                                        contColorList=['black', 'green'],
                                        contLineThicknessList=[1.0, 0.75],
                                        shadVariable=negThkInit_850_500 - unpThkInit_850_500,
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


# In[17]:


latBeg = 27.
lonBeg = -93.
latEnd = 50.
lonEnd = -55.
fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='green')
fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='green')
fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='green')
fig


# In[18]:


# cross-sections: perturbation potential temperature
wrfHDL = unpHdlInit
latBegList = [ 27.0]
lonBegList = [-93.0]
latEndList = [ 50.0]
lonEndList = [-53.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(get_wrf_th(unpHdlInit),unpHdlInit),
                             xSectShadVariable2=(get_wrf_th(posHdlInit),posHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(posHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.,
                             xSectTitleStr='intensifying case: pert. pot. temp'
                            )
plt.show(fig)

# cross-sections: perturbation potential temperature (weakening)
wrfHDL = unpHdlInit
latBegList = [ 27.0]
lonBegList = [-93.0]
latEndList = [ 50.0]
lonEndList = [-53.0]
u,v = get_uvmet(unpHdlInit)
unpWspd = np.sqrt(u**2. + v**2.)
spdrng = np.arange(25.,90.1,5.)
xSectShadInterval = np.arange(-6., 6.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                             latBegList=latBegList,
                             lonBegList=lonBegList,
                             latEndList=latEndList,
                             lonEndList=lonEndList,
                             xSectContVariableList=[(unpWspd,unpHdlInit), (get_wrf_th(unpHdlInit),unpHdlInit)],
                             xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                             xSectContColorList=['green', 'black'],
                             xSectContLineThicknessList=[2., 0.75],
                             xSectShadVariable1=(get_wrf_th(unpHdlInit),unpHdlInit),
                             xSectShadVariable2=(get_wrf_th(negHdlInit),negHdlInit),
                             xSectShadInterval=xSectShadInterval,
                             slp=get_wrf_slp(unpHdlInit),
                             slpInterval=np.arange(950., 1050.1, 4.),
                             thk=unpThkInit_850_500,
                             thkInterval=np.arange(3700., 4500.1, 50.),
                             slpPert=get_wrf_slp(negHdlInit)-get_wrf_slp(unpHdlInit),
                             slpPertInterval=slpPertInterval,
                             datProj=datProj,
                             plotProj=plotProj,
                             presLevMin=10000.,
                             xSectTitleStr='weakening case: pert. pot. temp'
                            )
plt.show(fig)


# In[19]:


fcstHrs = [6,12,18,24]

for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    vor = get_wrf_kinematic(unpHdl, 'vor')
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpVor500 = wrf.interplevel(field3d=vor,
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdVor500 = wrf.interplevel(field3d=vor,
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)


    # plan-section figure: perturbation 500 hPa vorticity and winds

    shdrng =  np.arange(-180.,180.01,20.)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(4900.,6550.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt500,unpHgt500-unpHgt850],
                                            contIntervalList=[hgtrng,np.arange(3600.,4500.,80.)], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.0,1.5],
                                            shadVariable=(ptdHgt500-unpHgt500),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 500 hPa perturbation heights'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])


# In[20]:


fcstHrs = [0,1,2,3,4,5,6,7,8,9,10,11,12]

for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    vor = get_wrf_kinematic(unpHdl, 'vor')
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpVor500 = wrf.interplevel(field3d=vor,
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdVor500 = wrf.interplevel(field3d=vor,
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)


    # plan-section figure: perturbation 500 hPa vorticity and winds

    shdrng =  np.arange(-24.,24.01,2.)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(4900.,6550.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt500,unpHgt500-unpHgt850],
                                            contIntervalList=[hgtrng,np.arange(3600.,4500.,80.)], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.0,1.5],
                                            shadVariable=(ptdHgt500-ptdHgt850) - (unpHgt500-unpHgt850),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 500 hPa perturbation 850-500 hPa thickness'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    
    latBeg = 32.5
    lonBeg = -80.
    latEnd = 50.
    lonEnd = -40.
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[21]:


latBegList = [32.25]
lonBegList = [-80.]
latEndList = [50.]
lonEndList = [-40.]
for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u,v = get_uvmet(unpHdl)
    unpWspd = np.sqrt(u**2. + v**2.)
    spdrng = np.arange(25.,90.1,5.)
    xSectShadInterval = np.arange(-18., 18.1, 2.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpWspd,unpHdl), (get_wrf_th(unpHdl),unpHdl)],
                                 xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadVariable2=(get_wrf_th(ptdHdl),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )
    xSectShadInterval = np.arange(-180., 180.1,20.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpWspd,unpHdl), (get_wrf_th(unpHdl),unpHdl)],
                                 xSectContIntervalList=[spdrng, np.arange(250., 450.1, 4.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(wrf.getvar(unpHdl,'z'),unpHdl),
                                 xSectShadVariable2=(wrf.getvar(ptdHdl,'z'),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. geop. hgt'
                                )
    plt.show(fig)


# In[22]:


latBegList = [32.25]
lonBegList = [-80.]
latEndList = [50.]
lonEndList = [-40.]
for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    pvorrng = np.arange(-2.,20.1,1.)
    xSectShadInterval = np.arange(-18., 18.1, 2.)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpPvor,unpHdl), (get_wrf_th(unpHdl),unpHdl)],
                                 xSectContIntervalList=[pvorrng, np.arange(250., 450.1, 4.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadVariable2=(get_wrf_th(ptdHdl),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )
    xSectShadInterval = np.arange(-8., 8.1, 0.5)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpPvor,unpHdl), (get_wrf_th(unpHdl),unpHdl)],
                                 xSectContIntervalList=[pvorrng, np.arange(250., 450.1, 4.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(wrf.getvar(unpHdl,'pvo'),unpHdl),
                                 xSectShadVariable2=(wrf.getvar(ptdHdl,'pvo'),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. PV'
                                )
    plt.show(fig)


# In[23]:


for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpPvor500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdPvor500 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)


    # plan-section figure: perturbation 500 hPa vorticity and winds

    shdrng =  np.arange(-4.,4.01,0.25)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(4900.,6550.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt500,unpHgt500-unpHgt850],
                                            contIntervalList=[hgtrng,np.arange(3600.,4500.,80.)], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.0,1.5],
                                            shadVariable=(ptdPvor500-unpPvor500),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 500 hPa perturbation PV'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[24]:


for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpPvor900 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt900 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdPvor900 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt900 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)


    # plan-section figure: perturbation 500 hPa vorticity and winds

    shdrng =  np.arange(-4.,4.01,0.25)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(600.,1200.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt900,unpHgt500-unpHgt850],
                                            contIntervalList=[hgtrng,np.arange(3600.,4500.,80.)], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.0,1.5],
                                            shadVariable=(ptdPvor900-unpPvor900),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 900 hPa perturbation PV'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[25]:


for fcstHr in fcstHrs:
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpPvor350 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt350 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdPvor350 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt350 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'z'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdHgt850 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'z'),
                                vert=prs,
                                desiredlev=85000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)


    # plan-section figure: perturbation 500 hPa vorticity and winds

    shdrng =  np.arange(-4.,4.01,0.25)
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(7400.,8800.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpHgt350,unpHgt500-unpHgt850],
                                            contIntervalList=[hgtrng,np.arange(3600.,4500.,80.)], 
                                            contColorList=['black','green'],
                                            contLineThicknessList=[1.0,1.5],
                                            shadVariable=(ptdPvor350-unpPvor350),
                                            shadInterval=shdrng[mask],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 350 hPa perturbation PV'.format(fcstHr))
    # add contour labels to hgt
    ax.clabel(cons[0],levels=hgtrng[::2])
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[ ]:





# In[ ]:





# In[26]:


for fcstHr in [24]:
    
    if fcstHr == 0:
        latBeg = 48.
        lonBeg = -100.
        latEnd = 22.
        lonEnd = -52.
    if fcstHr == 3:
        latBeg = 47.5
        lonBeg = -100.
        latEnd = 26.
        lonEnd = -52.
    if fcstHr == 6:
        latBeg = 47.5
        lonBeg = -100.
        latEnd = 28.
        lonEnd = -52.
    if fcstHr == 9:
        latBeg = 47.5
        lonBeg = -100.
        latEnd = 31.
        lonEnd = -52.
    if fcstHr == 12:
        latBeg = 47.
        lonBeg = -100.
        latEnd = 31.5
        lonEnd = -52.
    if fcstHr == 15:
        latBeg = 46.5
        lonBeg = -100.
        latEnd = 33.
        lonEnd = -52.
    if fcstHr == 18:
        latBeg = 42.
        lonBeg = -100.
        latEnd = 37.
        lonEnd = -52.
    if fcstHr == 21:
        latBeg = 40.
        lonBeg = -100.
        latEnd = 38.5
        lonEnd = -52.
    if fcstHr == 24:
        latBeg = 38.5
        lonBeg = -100.
        latEnd = 39.5
        lonEnd = -52.
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    u, v = get_uvmet(unpHdl)
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpPvor350 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False) 
    unpPvor500 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    unpPvor900 = wrf.interplevel(field3d=wrf.getvar(unpHdl,'pvo'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    u, v = get_uvmet(ptdHdl)
    vor = get_wrf_kinematic(ptdHdl, 'vor')
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdPvor350 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False) 
    ptdPvor500 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=50000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    ptdPvor900 = wrf.interplevel(field3d=wrf.getvar(ptdHdl,'pvo'),
                                vert=prs,
                                desiredlev=90000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})
    cntrng =  np.arange(-4.,4.01,0.25)
    mask = np.ones((np.shape(cntrng)),dtype='bool')
    mask[np.where(cntrng==0.)] = False
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[ptdPvor900-unpPvor900,ptdPvor500-unpPvor500,ptdPvor350-unpPvor350,get_wrf_slp(unpHdl)],
                                            contIntervalList=[cntrng[mask],cntrng[mask],cntrng[mask],np.arange(900.,1050.1,4.)], 
                                            contColorList=['blue','green','red','black'],
                                            contLineThicknessList=[1.0,1.0,1.0,1.0],
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 350 hPa perturbation PV'.format(fcstHr))
    #
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[27]:


# let's linearly interpolate between 3-hr segments to produce hourly cross-section settings
from scipy.interpolate import interp1d
t=np.arange(0.,24.1,3.)
latBeg3hr=np.asarray([48., 47.5, 47.5, 47.5, 47., 46.5, 42., 40., 38.5])
lonBeg3hr=np.asarray([-100., -100., -100., -100., -100., -100., -100., -100., -100.])
latEnd3hr=np.asarray([22., 26., 28., 31., 31.5, 33., 37., 38.5, 39.5])
lonEnd3hr=np.asarray([-52., -52., -52., -52., -52., -52., -52., -52., -52.])
f=interp1d(t,latBeg3hr)
latBeg1hr=f(np.arange(0.,24.1,1.))
f=interp1d(t,lonBeg3hr)
lonBeg1hr=f(np.arange(0.,24.1,1.))
f=interp1d(t,latEnd3hr)
latEnd1hr=f(np.arange(0.,24.1,1.))
f=interp1d(t,lonEnd3hr)
lonEnd1hr=f(np.arange(0.,24.1,1.))


# In[140]:


th_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (get_wrf_th))
p_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['p']))
for fcstHr in [12]:
    latBegList = [latBeg1hr[fcstHr].astype('float')]
    lonBegList = [lonBeg1hr[fcstHr].astype('float')]
    latEndList = [latEnd1hr[fcstHr].astype('float')]
    lonEndList = [lonEnd1hr[fcstHr].astype('float')]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdSpd_int = interpolate_sigma_levels(ptdSpd, p, ps, pt, s, unpHdl)
    ptdHgt_int = interpolate_sigma_levels(ptdHgt, p, ps, pt, s, unpHdl)
    hgtrng = np.arange(-60., 60.1, 8.)
    hgtrng = hgtrng[np.where(hgtrng != 0.)]
    pvorrng = [2.]
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    th_avg_int = interpolate_sigma_levels(th_avg, p_avg, ps, pt, s, unpHdl)
    xSectShadInterval = np.arange(-4., 4.1, 0.25)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpPvor,unpHdl), (ptdPvor,ptdHdl), (ptdHgt_int-unpHgt,unpHdl)],
                                 xSectContIntervalList=[pvorrng, pvorrng, hgtrng],
                                 xSectContColorList=['green', 'red','black'],
                                 xSectContLineThicknessList=[2., 2., 0.75],
                                 xSectShadVariable1=(wrf.getvar(unpHdl,'pvo'),unpHdl),  #(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadVariable2=(wrf.getvar(ptdHdl,'pvo'),ptdHdl),  #get_wrf_th(ptdHdl),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )
    #xSectShadInterval = np.arange(-5., 5.1, 0.25)
    #xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    #fig = cross_section_diffplot(
    #                             latBegList=latBegList,
    #                             lonBegList=lonBegList,
    #                             latEndList=latEndList,
    #                             lonEndList=lonEndList,
    #                             xSectContVariableList=[(unpPvor,unpHdl), (ptdPvor,ptdHdl), (get_wrf_th(unpHdl)-th_avg_int,unpHdl)],
    #                             xSectContIntervalList=[pvorrng, pvorrng, np.arange(-20., 20.1, 2.)],
    #                             xSectContColorList=['green', 'red', 'black'],
    #                             xSectContLineThicknessList=[2., 2., 0.75],
    #                             xSectShadVariable1=(wrf.getvar(unpHdl,'pvo'),unpHdl),
    #                             xSectShadVariable2=(wrf.getvar(ptdHdl,'pvo'),ptdHdl),
    #                             xSectShadInterval=xSectShadInterval,
    #                             slp=get_wrf_slp(unpHdl),
    #                             slpInterval=np.arange(950., 1050.1, 4.),
    #                             thk=unpHgt500-unpHgt850,
    #                             thkInterval=np.arange(3700., 4500.1, 50.),
    #                             slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
    #                             slpPertInterval=slpPertInterval,
    #                             datProj=datProj,
    #                             plotProj=plotProj,
    #                             presLevMin=10000.,
    #                             xSectTitleStr='pert. PV'
    #                            )
    print('hour {:d}'.format(fcstHr))
    plt.show(fig)


# In[176]:


th_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (get_wrf_th))
p_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['p']))
for fcstHr in [18]:
    latBegList = [latBeg1hr[fcstHr].astype('float') +1.]
    lonBegList = [lonBeg1hr[fcstHr].astype('float')]
    latEndList = [latEnd1hr[fcstHr].astype('float') +1.]
    lonEndList = [lonEnd1hr[fcstHr].astype('float')]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdSpd_int = interpolate_sigma_levels(ptdSpd, p, ps, pt, s, unpHdl)
    ptdHgt_int = interpolate_sigma_levels(ptdHgt, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    hgtrng = np.arange(-60., 60.1, 8.)
    hgtrng = hgtrng[np.where(hgtrng != 0.)]
    spdrng = np.arange(35.,100.,5.)
    pvorrng = [2.]
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    th_avg_int = interpolate_sigma_levels(th_avg, p_avg, ps, pt, s, unpHdl)
    xSectShadInterval = np.arange(-4., 4.1, 0.25)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpPvor, ptdPvor_int, ptdHgt_int-unpHgt],
                                 xSectContIntervalList=[pvorrng, pvorrng, hgtrng],
                                 xSectContColorList=['lime', 'gold','black'],
                                 xSectContLineThicknessList=[2., 2., 0.75],
                                 xSectShadVariable=ptdPvor_int-unpPvor,  #(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )
    xSectShadInterval = np.arange(-10., 10.1, 1.0)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpPvor, ptdPvor_int, unpSpd, ptdSpd_int],
                                 xSectContIntervalList=[[2.], [2.], spdrng, spdrng],
                                 xSectContColorList=['lime', 'gold', 'blue', 'red'],
                                 xSectContLineThicknessList=[2., 2., 1., 1.],
                                 xSectShadVariable=ptdPvor_int-unpPvor,  #(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. PV'
                                )
    print('hour {:d}'.format(fcstHr))
    plt.show(fig)


# In[177]:


th_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (get_wrf_th))
p_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['p']))
for fcstHr in [18]:
    latBegList = [latBeg1hr[fcstHr].astype('float') +1.]
    lonBegList = [lonBeg1hr[fcstHr].astype('float')]
    latEndList = [latEnd1hr[fcstHr].astype('float') +1.]
    lonEndList = [lonEnd1hr[fcstHr].astype('float')]
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    u,v = get_uvmet(unpHdl)
    unpSpd = np.sqrt(u**2. + v**2.)
    unpHgt = wrf.getvar(unpHdl,'z')
    u,v = get_uvmet(ptdHdl)
    ptdSpd = np.sqrt(u**2. + v**2.)
    ptdHgt = wrf.getvar(ptdHdl,'z')
    p = wrf.getvar(ptdHdl,'p')
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    ptdSpd_int = interpolate_sigma_levels(ptdSpd, p, ps, pt, s, unpHdl)
    ptdHgt_int = interpolate_sigma_levels(ptdHgt, p, ps, pt, s, unpHdl)
    ptdPvor_int = interpolate_sigma_levels(ptdPvor, p, ps, pt, s, unpHdl)
    hgtrng = np.arange(-60., 60.1, 8.)
    hgtrng = hgtrng[np.where(hgtrng != 0.)]
    spdrng = np.arange(35.,100.,5.)
    pvorrng = [2.]
    ps = np.asarray(unpHdl.variables['PSFC']).squeeze()
    pt = np.asarray(unpHdl.variables['P_TOP']) * 1.0
    s = np.asarray(unpHdl.variables['ZNU']).squeeze()
    th_avg_int = interpolate_sigma_levels(th_avg, p_avg, ps, pt, s, unpHdl)
    xSectShadInterval = np.arange(-4., 4.1, 0.25)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    slpPertInterval = np.arange(-30.,30.1, 2.)
    slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpPvor, ptdPvor_int, ptdHgt_int-unpHgt],
                                 xSectContIntervalList=[pvorrng, pvorrng, hgtrng],
                                 xSectContColorList=['lime', 'gold','black'],
                                 xSectContLineThicknessList=[2., 2., 0.75],
                                 xSectShadVariable=ptdPvor_int-unpPvor,  #(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )
    xSectShadInterval = np.arange(-10., 10.1, 1.0)
    xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
    fig = cross_section_plot(
                                 wrfHDL=unpHdl,
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[unpPvor, ptdPvor_int, unpSpd, ptdSpd_int],
                                 xSectContIntervalList=[[2.], [2.], spdrng, spdrng],
                                 xSectContColorList=['lime', 'gold', 'blue', 'red'],
                                 xSectContLineThicknessList=[2., 2., 1., 1.],
                                 xSectShadVariable=ptdPvor_int-unpPvor,  #(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. PV'
                                )
    print('hour {:d}'.format(fcstHr))
    plt.show(fig)


# In[ ]:


# First crack at an explanation for the March 2020 bomb cyclone case:
#
# The cross-sections are demonstrating that the majority of the influences on the cyclone's intensity are related
# to perturbations that evolve DOWNSTREAM rather than upstream of the cyclone, which modify the intensity and extent
# west-east of the SUBTROPICAL JET rather than the polar jet. In the weakening case, a dynamic tropopause fold
# downstream of the cyclone is pushed further downstream, which inhibits the westward extent of the subtrop. jet and
# reduces jet interaction. In the intensifying case, this dyn. trop. fold is eradicated entirely, allowing the
# subtrop. jet to extend upstream and increase jet interaction with the cyclone. These appear to be NONLINEAR
# results, as the intensifying case's dyn. trop. evolves very differently (not just reversed) from the weakening
# case in this key region.


# In[156]:


pv_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['pvo']))
p_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['p']))
for fcstHr in [18]:
    latBeg = latBeg1hr[fcstHr].astype('float') + 1.
    lonBeg = lonBeg1hr[fcstHr].astype('float')
    latEnd = latEnd1hr[fcstHr].astype('float') + 1.
    lonEnd = lonEnd1hr[fcstHr].astype('float')
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    unpFileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    ptdFileFcst = negDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFileFcst)
    ptdHdl = Dataset(ptdFileFcst)
    unpPvor = wrf.getvar(unpHdl,'pvo')
    ptdPvor = wrf.getvar(ptdHdl,'pvo')
    prs = np.asarray(wrf.getvar(unpHdl,'p')).squeeze()
    unpPvor350 = wrf.interplevel(field3d=unpPvor,
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    prs = np.asarray(wrf.getvar(ptdHdl,'p')).squeeze()
    ptdPvor350 = wrf.interplevel(field3d=ptdPvor,
                                vert=prs,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    avgPvor350 = wrf.interplevel(field3d=pv_avg,
                                vert=p_avg,
                                desiredlev=35000.,
                                missing=np.nan,
                                squeeze=True,
                                meta=False)
    # plan-section figure: 350 hpa PV spaghetti plots

    shdrng =  np.arange(-4.,4.01,0.25).astype('float')
    mask = np.ones((np.shape(shdrng)),dtype='bool')
    mask[np.where(shdrng==0.)] = False
    hgtrng = np.arange(7400.,8800.1,45.)


    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(12,7), subplot_kw={'projection' : datProj})

    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=unpHdl,
                                            lat=lat,
                                            lon=lon,
                                            contVariableList=[unpPvor350, get_wrf_slp(unpHdl), get_wrf_slp(ptdHdl)],
                                            contIntervalList=[np.arange(1.,8.1,2.),np.arange(900.,1008.1,6.),np.arange(900.,1008.1,6.)], 
                                            contColorList=['black','green','purple'],
                                            contLineThicknessList=[1.0,1.0,1.0],
                                            shadVariable=ptdPvor350-unpPvor350,
                                            shadInterval=shdrng[mask],
                                            shadAlpha=0.6,
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
    ax.set_title(dtFcstStr + ' ({:d} hrs) 350 hPa perturbation PV'.format(fcstHr))
    fig.gca().plot(lonBeg,latBeg,'o',transform=plotProj,color='magenta')
    fig.gca().plot(lonEnd,latEnd,'o',transform=plotProj,color='magenta')
    fig.gca().plot((lonBeg,lonEnd),(latBeg,latEnd),transform=plotProj,color='magenta')
    fig


# In[87]:


shdrng


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


th_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (get_wrf_th))
p_avg = gen_time_avg(d10Dir, avgFileNameBeg, avgTimeStamps, (wrf.getvar, ['p']))


# In[30]:


xSectShadInterval = np.arange(-12., 12.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpPvor,unpHdl), (get_wrf_th(unpHdl)-th_avg,unpHdl)],
                                 xSectContIntervalList=[pvorrng, np.arange(-30., 30.1, 4.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadVariable2=(get_wrf_th(ptdHdl),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )


# In[43]:


from analysis_dependencies import dim_coord_swap
# interpolate_sigma_levels: interpolate a 3D field from its own sigma-coordinate to a donor's
#                           sigma-coordinate.
#
# INPUTS:
#   field3D: field in native sigma-coordinate to be interpolated to donor's sigma-coordinate (*, [nz,ny,nx]-dimension)
#   pres3D: pressure values for field3D in native sigma-coordinate (float, [nz,ny,nx]-dimension)
#   sfcPresDonor: 2D field of surface-pressure values from donor (float, [ny,nx]-dimension)
#   topPresDonor: donor top pressure (float, single-value)
#   sigmaLevels: 1D vector of donor's sigma-levels to interpolate to (float, [nz]-dimension)
#   donorHdl: netcdf4.Dataset() file-handle for donor, to extract dimension/coordinate values
#
# OUTPUTS:
#   interp3D: field3D interpolated to donor's sigma-coordinate (*, [nz,ny,nx]-dimension)
#
# DEPENDENCIES:
#   numpy
#   xarray
#   netCDF4.Dataset()
#   wrf-python: wrf.interplevel(), wrf.getvar()
#   analysis_dependencies.dim_coord_swap()
#
def interpolate_sigma_levels(field3D, pres3D, sfcPresDonor, topPresDonor, sigmaLevels, donorHdl):
    # Some notes on the technique applied here:
    #
    # Sigma levels are a normalized pressure coordinate defined as:
    #
    # sig[k,j,i] = (p[k,j,i] - p_top) / (p_sfc[j,i] - p_top)
    #
    # where: p[k,j,i] is the pressure at a 3D point in [nz,ny,nx]-dimension
    #        p_sfc[j,i] is the surface pressure at the corresponding 2D surface point in [ny,nx]-dimension
    #        p_top is the model-top pressure, which is assumed to be a single fixed value at all [j,i] points
    #
    # Comparing two 3D fields on their own respective sigma coordinates does not work if the pressure field is
    # different between them (e.g., comparing the temperature fields between an unperturbed run and a perturbed
    # run which generate different pressure fields), because the normalization of the pressure that defines the
    # sigma surface is computed differently for both fields, i.e. "your sig=0.5 is not the same as my sig=0.5".
    #
    # To do this comparison in sigma-space, one of the fields has to be interpolated onto the other field's
    # sigma surface. This can be done by computing sig[k,j,i] for the interpolated field using it's own
    # p[k,j,i] values, but then using the donor's p_sfc[j,i] and p_top values to do the normalization. This way,
    # sig[k,j,i]=0.5 represents the *same* pressure value for both fields, and the two can be compared.
    #
    # The technique applied here is to compute the effective donor sigma-values on each level of the field
    # to be interpolated, then to interpolate from it's own sigma-values to the donor's sigma-values. The
    # interpolated field could intersect below-surface points (sigma>1) or above-top points (sigma<1), depending
    # on any underlying differences in the p_sfc or p_top values between the interpolated and donor model
    # states. These are given np.nan values.
    #
    # This can all be accomplished in xarray.DataArray() fields, but the interpolated field will lose its
    # vertical dimension and coordinate names/values in the process. This metadata can all be retrieved using
    # analysis_dependencies.dim_coord_swap().
    import numpy as np
    import xarray as xr
    from wrf import interplevel
    from wrf import getvar
    from netCDF4 import Dataset
    # compute sig3D as the interpolated field's pres3D normalized by sfcPresDonor and topPresDonor
    #   Assuming that pres3D is [nz,ny,nx]-dimension and sfcPresDonor is [ny,nx]-dimension, this operation
    #   should be broadcastable by numpy rules, with the denominator being left-padded as [1,ny,nx]-dimension
    #   and then applied across all nz-dimension levels. If this is not the case, you will probably get an
    #   error of the form "cannot be broadcast".
    sig3D = np.divide(pres3D - topPresDonor, sfcPresDonor - topPresDonor)
    # interpolate field3D on sig3D surfaces to donor's standard sigma-levels in sigmaLevels, with any
    # points that are not interperable assigned to np.nan
    interp3D = wrf.interplevel(field3d=field3D,
                               vert=sig3D,
                               desiredlev=sigmaLevels,
                               missing=np.nan,
                               meta=False) # meta=True will default to returning an xarray.DataArray() with
                                           # metadata (dimension and coordinate names/values) intact, rather
                                           # than a numpy array, if possible
    # if interp3D is an xarray.DataArray() object, the metadata is retained with the exception of the vertical
    # coordinate, in which case perform dim_coord_swap() on donor's pressure field to retain metadata
    presDonor = getvar(donorHdl, 'p')
    if type(interp3D) == xr.core.dataarray.DataArray:
        interp3D = dim_coord_swap(interp3D, presDonor)
    else:
        interp3D = xr.DataArray(interp3D)
        interp3D = dim_coord_swap(interp3D, presDonor)
    # return interp3D
    return interp3D
        


# In[44]:


th = th_avg
p = p_avg
ps = np.asarray(unpHdlFcst.variables['PSFC']).squeeze()
pt = np.asarray(unpHdlFcst.variables['P_TOP']) * 1.0
s = np.asarray(unpHdlFcst.variables['ZNU']).squeeze()
th_avg_int = interpolate_sigma_levels(th, p, ps, pt, s, unpHdlFcst)


# In[48]:


xSectShadInterval = np.arange(-12., 12.1, 0.5)
xSectShadInterval = xSectShadInterval[np.where(xSectShadInterval != 0.)]
slpPertInterval = np.arange(-30.,30.1, 2.)
slpPertInterval = slpPertInterval[np.where(slpPertInterval != 0.)]
fig = cross_section_diffplot(
                                 latBegList=latBegList,
                                 lonBegList=lonBegList,
                                 latEndList=latEndList,
                                 lonEndList=lonEndList,
                                 xSectContVariableList=[(unpPvor,unpHdl), (get_wrf_th(unpHdl)-th_avg_int,unpHdl)],
                                 xSectContIntervalList=[[2.], np.arange(-20., 20.1, 2.)],
                                 xSectContColorList=['green', 'black'],
                                 xSectContLineThicknessList=[2., 0.75],
                                 xSectShadVariable1=(get_wrf_th(unpHdl),unpHdl),
                                 xSectShadVariable2=(get_wrf_th(ptdHdl),ptdHdl),
                                 xSectShadInterval=xSectShadInterval,
                                 slp=get_wrf_slp(unpHdl),
                                 slpInterval=np.arange(950., 1050.1, 4.),
                                 thk=unpHgt500-unpHgt850,
                                 thkInterval=np.arange(3700., 4500.1, 50.),
                                 slpPert=get_wrf_slp(ptdHdl)-get_wrf_slp(unpHdl),
                                 slpPertInterval=slpPertInterval,
                                 datProj=datProj,
                                 plotProj=plotProj,
                                 presLevMin=10000.,
                                 xSectTitleStr='pert. pot. temp'
                                )


# In[35]:


th_avg


# In[39]:


th


# In[ ]:




