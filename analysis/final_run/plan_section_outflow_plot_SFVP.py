# UWAOS_WRF_QOIP-2 compliant (includes tcpipy)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import datetime
import wrf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import gen_wrf_proj
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_uvmet
from analysis_dependencies import compute_PI_vars
from analysis_dependencies import compute_vinterp_weights
from analysis_dependencies import compute_storm_relative_wind

# define unperturbed and perturbed forecast file-handles at low- and high-res
dtInit = datetime.datetime(2021, 8, 28, 18)

fcstHr = 18
print('processing forecast hour {:d}'.format(fcstHr))

dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
res='9km'
unpFileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/wrfout_d01_' + dtFcstStr
ptdFileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res +'_res/wrfout_d01_' + dtFcstStr
res='27km'
unpFileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/wrfout_d01_' + dtFcstStr
ptdFileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res +'_res/wrfout_d01_' + dtFcstStr
unpHdlHires = Dataset(unpFileHires)
ptdHdlHires = Dataset(ptdFileHires)
unpHdlLores = Dataset(unpFileLores)
ptdHdlLores = Dataset(ptdFileLores)

# load streamfunction and velocity potential fields
res='27km'
fileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/InverseLaplacian_' + res + '_' + dtFcstStr + '.nc4'
hdl = Dataset(fileLores)
unpSFLores = np.asarray(hdl.variables['SF']).squeeze()
unpVPLores = np.asarray(hdl.variables['VP']).squeeze()
fileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res + '_res/InverseLaplacian_' + res + '_' + dtFcstStr + '.nc4'
hdl = Dataset(fileLores)
ptdSFLores = np.asarray(hdl.variables['SF']).squeeze()
ptdVPLores = np.asarray(hdl.variables['VP']).squeeze()
res='9km'
fileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/InverseLaplacian_' + res + '_' + dtFcstStr + '.nc4'
hdl = Dataset(fileHires)
unpSFHires = np.asarray(hdl.variables['SF']).squeeze()
unpVPHires = np.asarray(hdl.variables['VP']).squeeze()
unpSFbHires = np.asarray(hdl.variables['SF_bounds']).squeeze()
unpVPbHires = np.asarray(hdl.variables['VP_bounds']).squeeze()
fileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res + '_res/InverseLaplacian_' + res + '_' + dtFcstStr + '.nc4'
hdl = Dataset(fileHires)
ptdSFHires = np.asarray(hdl.variables['SF']).squeeze()
ptdVPHires = np.asarray(hdl.variables['VP']).squeeze()
ptdSFbHires = np.asarray(hdl.variables['SF_bounds']).squeeze()
ptdVPbHires = np.asarray(hdl.variables['VP_bounds']).squeeze()

# compute lat and lon on hi-res and lo-res grids
latHires = np.asarray(unpHdlHires.variables['XLAT']).squeeze()
lonHires = np.asarray(unpHdlHires.variables['XLONG']).squeeze()
latLores = np.asarray(unpHdlLores.variables['XLAT']).squeeze()
lonLores = np.asarray(unpHdlLores.variables['XLONG']).squeeze()

# compute data and plot projections
plotProj = ccrs.PlateCarree()
datProj = gen_cartopy_proj(unpHdlHires)
#
# hi-res grids
#
# extract SLP and compute location of SLP minimum (hi-res grid only)
# unperturbed
unpSLP = np.asarray(get_wrf_slp(unpHdlHires)).squeeze()
unpJc, unpIc = np.unravel_index(np.argmin(unpSLP.flatten()),shape=np.shape(unpSLP))
# perturbed
ptdSLP = np.asarray(get_wrf_slp(ptdHdlHires)).squeeze()
ptdJc, ptdIc = np.unravel_index(np.argmin(ptdSLP.flatten()),shape=np.shape(ptdSLP))
# compute potential intensity variables
# unperturbed
x1,x2,x3,x4,x5 = compute_PI_vars(unpHdlHires)
unpOTL = x5.to_numpy()
# perturbed
x1,x2,x3,x4,x5 = compute_PI_vars(ptdHdlHires)
ptdOTL = x5.to_numpy()
#
# interpolate fields to outflow level
# unperturbed
pvec = np.asarray(wrf.getvar(unpHdlHires,'p')).squeeze()
nz,ny,nx = pvec.shape
pvec = pvec.reshape((nz,ny*nx))
ovec = 100. * unpOTL.copy()
ovec = ovec.reshape((ny*nx))
w = compute_vinterp_weights(pvec,ovec,isLog=True)
u,v = get_uvmet(unpHdlHires)
var = np.asarray(u).squeeze().reshape((nz,ny*nx))
unpUwdOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # zonal wind on outflow level surface
var = np.asarray(v).squeeze().reshape((nz,ny*nx))
unpVwdOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # merid wind on outflow level surface
var = np.asarray(unpSFHires + unpSFbHires).squeeze().reshape((nz,ny*nx))
unpSFOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # streamfunction on outflow level surface
var = np.asarray(unpVPHires + unpVPbHires).squeeze().reshape((nz,ny*nx))
unpVPOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # velocity potential on outflow level surface
##############################
# activate this block of lines to interpolate perturbed variables to perturbed outflow level surface, otherwise
# perturbed variables are put on unperturbed outflow level surface
##############################
pvec = np.asarray(wrf.getvar(ptdHdlHires,'p')).squeeze()
nz,ny,nx = pvec.shape
pvec = pvec.reshape((nz,ny*nx))
ovec = 100. * ptdOTL.copy()
ovec = ovec.reshape((ny*nx))
w = compute_vinterp_weights(pvec,ovec,isLog=True)
##############################
# perturbed
u,v = get_uvmet(ptdHdlHires)
var = np.asarray(u).squeeze().reshape((nz,ny*nx))
ptdUwdOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # zonal wind on outflow level surface
var = np.asarray(v).squeeze().reshape((nz,ny*nx))
ptdVwdOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # merid wind on outflow level surface
var = np.asarray(ptdSFHires + ptdSFbHires).squeeze().reshape((nz,ny*nx))
ptdSFOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # streamfunction on outflow level surface
var = np.asarray(ptdVPHires + ptdVPbHires).squeeze().reshape((nz,ny*nx))
ptdVPOTLHires = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # velocity potential on outflow level surface
#
# lo-res grids
#
# compute potential intensity variables
# unperturbed
x1,x2,x3,x4,x5 = compute_PI_vars(unpHdlLores)
unpOTL = x5.to_numpy()
# perturbed
x1,x2,x3,x4,x5 = compute_PI_vars(ptdHdlLores)
ptdOTL = x5.to_numpy()
#
# interpolate fields to outflow level
# unperturbed
pvec = np.asarray(wrf.getvar(unpHdlLores,'p')).squeeze()
nz,ny,nx = pvec.shape
pvec = pvec.reshape((nz,ny*nx))
ovec = 100. * unpOTL.copy()
ovec = ovec.reshape((ny*nx))
w = compute_vinterp_weights(pvec,ovec,isLog=True)
u,v = get_uvmet(unpHdlLores)
var = np.asarray(u).squeeze().reshape((nz,ny*nx))
unpUwdOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # zonal wind on outflow level surface
var = np.asarray(v).squeeze().reshape((nz,ny*nx))
unpVwdOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # merid wind on outflow level surface
var = np.asarray(unpSFLores).squeeze().reshape((nz,ny*nx))
unpSFOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # streamfunction on outflow level surface
var = np.asarray(unpVPLores).squeeze().reshape((nz,ny*nx))
unpVPOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # velocity potential on outflow level surface
##############################
# activate this block of lines to interpolate perturbed variables to perturbed outflow level surface, otherwise
# perturbed variables are put on unperturbed outflow level surface
##############################
pvec = np.asarray(wrf.getvar(ptdHdlLores,'p')).squeeze()
nz,ny,nx = pvec.shape
pvec = pvec.reshape((nz,ny*nx))
ovec = 100. * ptdOTL.copy()
ovec = ovec.reshape((ny*nx))
w = compute_vinterp_weights(pvec,ovec,isLog=True)
##############################
# perturbed
u,v = get_uvmet(ptdHdlLores)
var = np.asarray(u).squeeze().reshape((nz,ny*nx))
ptdUwdOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # zonal wind on outflow level surface
var = np.asarray(v).squeeze().reshape((nz,ny*nx))
ptdVwdOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # merid wind on outflow level surface
var = np.asarray(ptdSFLores).squeeze().reshape((nz,ny*nx))
ptdSFOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # streamfunction on outflow level surface
var = np.asarray(ptdVPLores).squeeze().reshape((nz,ny*nx))
ptdVPOTLLores = np.sum(np.multiply(w,var),axis=0).reshape((ny,nx))  # velocity potential on outflow level surface
# define hi-res subdomain
borderHires = np.zeros(np.shape(latHires))
borderHires[0,:] = 1.
borderHires[-1,:] = 1.
borderHires[:,0] = 1.
borderHires[:,-1] = 1.
# Define shading and contouring ranges
shdrng = 1.0E+05*np.arange(-16., 16.1, 2.).astype('float16')
mask = np.ones(np.shape(shdrng), dtype='bool')
mask[np.where(shdrng==0.)] = False
cntrng = 1.0E+05*np.arange(-90., 90.1, 9.).astype('float16')
cntrng = cntrng[np.where(cntrng!=0.)]
# Define figure and axis, set projection to datProjLores
fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(24,7),subplot_kw={'projection':datProj})
#
# Streamfunction on outflow-level
#
ax=axs
# 1. Plot lo-res shading and define as shd, zorder=0
shd=ax.contourf(lonLores, latLores, ptdSFOTLLores-unpSFOTLLores, shdrng[mask], cmap='bwr', transform=plotProj, zorder=0, extend='both')
oln=ax.contour(lonLores, latLores, ptdSFOTLLores-unpSFOTLLores, shdrng[mask], colors='black', linewidths=0.75, transform=plotProj,zorder=0)
# 2. Plot lo-res contour, zorder=1
ax.contour(lonLores, latLores, unpSFOTLLores, cntrng, colors='gray', transform=plotProj,zorder=1)
# 3. Plot borderHires as a fill pattern with levels of -9999. and 0., shading is white (blocking out underlying lo-res plotting), zorder=2
ax.contourf(lonHires, latHires, borderHires, levels=[-9999., 0.], colors='white', transform=plotProj,zorder=2)
# 4. Plot hi-res shading, zorder=3
ax.contourf(lonHires, latHires, ptdSFOTLHires-unpSFOTLHires, shdrng[mask], cmap='bwr', transform=plotProj,zorder=3)
ax.contour(lonHires, latHires, ptdSFOTLHires-unpSFOTLHires, shdrng[mask], colors='black', linewidths=0.75, transform=plotProj,zorder=3)
# 5. Plot contour of borderHires, levels of 0., 1., 9999., zorder=4
ax.contour(lonHires, latHires, borderHires, levels=[0., 1., 9999.], colors='green', transform=plotProj,zorder=4)
# 6. Plot hi-res contour, zorder=5
ax.contour(lonHires, latHires, unpSFOTLHires, cntrng, colors='black', transform=plotProj,zorder=5)
# 7. Plot hi-res unperturbed and perturbed SLP minimum
ax.plot(lonHires[unpJc,unpIc], latHires[unpJc, unpIc], 'o', markersize=8, markerfacecolor='orange', markeredgecolor='black', linewidth=0.75, transform=plotProj, zorder=6)
ax.plot(lonHires[ptdJc,ptdIc], latHires[ptdJc, ptdIc], 'o', markersize=8, markerfacecolor='green', markeredgecolor='black', linewidth=0.75, transform=plotProj, zorder=6)
# 8. Plot map
ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
# 9. Plot colorbar for shd
plt.colorbar(mappable=shd, ax=ax)
ax.set_title('streamfunction on outflow level')
plt.show()
# Define shading and contouring ranges
shdrng = 1.0E+05*np.arange(-16., 16.1, 2.).astype('float16')
mask = np.ones(np.shape(shdrng), dtype='bool')
mask[np.where(shdrng==0.)] = False
cntrng = 1.0E+05*np.arange(-90., 90.1, 9.)
# Define figure and axis, set projection to datProjLores
fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(24,7),subplot_kw={'projection':datProj})
#
# Velocity potential on outflow-level
#
ax=axs
# 1. Plot lo-res shading and define as shd, zorder=0
shd=ax.contourf(lonLores, latLores, ptdVPOTLLores-unpVPOTLLores, shdrng[mask], cmap='bwr', transform=plotProj, zorder=0, extend='both')
oln=ax.contour(lonLores, latLores, ptdVPOTLLores-unpVPOTLLores, shdrng[mask], colors='black', linewidths=0.75, transform=plotProj,zorder=0)
# 2. Plot lo-res contour, zorder=1
ax.contour(lonLores, latLores, unpVPOTLLores, cntrng, colors='gray', transform=plotProj,zorder=1)
# 3. Plot borderHires as a fill pattern with levels of -9999. and 0., shading is white (blocking out underlying lo-res plotting), zorder=2
ax.contourf(lonHires, latHires, borderHires, levels=[-9999., 0.], colors='white', transform=plotProj,zorder=2)
# 4. Plot hi-res shading, zorder=3
ax.contourf(lonHires, latHires, ptdVPOTLHires-unpVPOTLHires, shdrng[mask], cmap='bwr', transform=plotProj,zorder=3)
ax.contour(lonHires, latHires, ptdVPOTLHires-unpVPOTLHires, shdrng[mask], colors='black', linewidths=0.75, transform=plotProj,zorder=3)
# 5. Plot contour of borderHires, levels of 0., 1., 9999., zorder=4
ax.contour(lonHires, latHires, borderHires, levels=[0., 1., 9999.], colors='green', transform=plotProj,zorder=4)
# 6. Plot hi-res contour, zorder=5
ax.contour(lonHires, latHires, unpVPOTLHires, cntrng, colors='black', transform=plotProj,zorder=5)
# 7. Plot hi-res unperturbed and perturbed SLP minimum
ax.plot(lonHires[unpJc,unpIc], latHires[unpJc, unpIc], 'o', markersize=8, markerfacecolor='orange', markeredgecolor='black', linewidth=0.75, transform=plotProj, zorder=6)
ax.plot(lonHires[ptdJc,ptdIc], latHires[ptdJc, ptdIc], 'o', markersize=8, markerfacecolor='green', markeredgecolor='black', linewidth=0.75, transform=plotProj, zorder=6)
# 8. Plot map
ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
# 9. Plot colorbar for shd
plt.colorbar(mappable=shd, ax=ax)
ax.set_title('velocity potential on outflow level')
plt.show()
