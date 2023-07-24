##############################################################################################
#
# PYTHON 3 PROGRAM
#
# Computes perturbation kinematic fields: vorticity, divergence, stretching deformation, and
# shearing deformation, from perturbation u- and v-components of the wind.
#
# Outputs to netCDF file.
#
##############################################################################################
#
# Import necessary modules
#
from netCDF4 import Dataset #................................................................. netCDF i/o module
import numpy as np #.......................................................................... array module
import wrf #.................................................................................. wrf-python module
from adj_diagnostics import calc_vor  #....................................................... vorticity calculation
from adj_diagnostics import calc_div  #....................................................... divergence calculation 
from adj_diagnostics import calc_str  #....................................................... stretching deformation calculation 
from adj_diagnostics import calc_shr  #....................................................... shearing deformation calculation 
#
##############################################################################################
#
# Ingest user input
#
inp_file = input() #.......................................................................... full path to (unperturbed) wrfinput_d01 file
ptd_file = input() #.......................................................................... full path to perturbed wrfinput_d01 file
pert_outfile_name = input() #................................................................. full path+filename for perturbation output
#
##############################################################################################
#
# Extract sensitivity fields from adj_file and grid information from inp_file
#
inp_hdl = Dataset(inp_file) #................................................................. inp_file handle
ptd_hdl = Dataset(ptd_file) #................................................................. ptd_file handle
# Extract grid information
lat = wrf.getvar(inp_hdl,'lat') #............................................................. grid latitudes [nlat,nlon]
lon = wrf.getvar(inp_hdl,'lon') #............................................................. grid longitudes [nat,nlon]
cen_lon = inp_hdl.STAND_LON #................................................................. grid center longitude
true_lat1 = inp_hdl.TRUELAT1 #................................................................ grid true latitude 1
true_lat2 = inp_hdl.TRUELAT2 #................................................................ grid true latitude 2
radians_per_degree = np.pi/180.
mf = inp_hdl.variables['MAPFAC_M'] #.......................................................... grid map factors on mass points [nlat,nlon]
# Define dx, dy as [nlat,nlon] grids normalized by mf
dx = inp_hdl.DX*np.power(mf,-1.) #............................................................ grid dx
dy = inp_hdl.DY*np.power(mf,-1.) #............................................................ grid dy
# Define conic projection value based on true_lat1 and true_lat2
if((abs(true_lat1 - true_lat2) > 0.1) and
(abs(true_lat2 - 90.) > 0.1)):
    cone = (np.log(np.cos(true_lat1*radians_per_degree))
    - np.log(np.cos(true_lat2*radians_per_degree)))

    cone = (cone /
    (np.log(np.tan((45.-abs(true_lat1/2.))*radians_per_degree))
    - np.log(np.tan((45.-abs(true_lat2/2.))*radians_per_degree))))
else:
    cone = np.sin(abs(true_lat1)*radians_per_degree)
# Extract (un)perturbed u,v information
u0 = inp_hdl.variables['U'] #................................................................. unperturbed zonal flow [nlev,nlat,nlon]
v0 = inp_hdl.variables['V'] #................................................................. unperturbed merid flow [nlev,nlat,nlon]
u1 = ptd_hdl.variables['U'] #................................................................. perturbed zonal flow [nlev,nlat,nlon]
v1 = ptd_hdl.variables['V'] #................................................................. perturbed merid flow [nlev,nlat,nlon]
# Destagger wind grids to mass points
u0 = wrf.destagger(u0,stagger_dim=3)
v0 = wrf.destagger(v0,stagger_dim=2)
u1 = wrf.destagger(u1,stagger_dim=3)
v1 = wrf.destagger(v1,stagger_dim=2)
# Compute perturbation wind components
up = np.asarray(u1)-np.asarray(u0) #.......................................................... perturbation zonal flow [nlev,nlat,nlon]
vp = np.asarray(v1)-np.asarray(v0) #.......................................................... perturbation merid flow [nlev,nlat,nlon]
# Rotate perturbation wind grids to earth coordinates
rot = wrf.uvmet(up,vp,lat,lon,cen_lon,cone) #................................................. rotated (up,vp) fields
up_rot = rot[0,:,:,:,:] #..................................................................... rotated up
vp_rot = rot[1,:,:,:,:] #..................................................................... rotated vp
# Define grid dimensions
nz,ny,nx = np.shape(np.asarray(up).squeeze()) #............................................... grid dimensions
#
##############################################################################################
#
# Compute perturbation kinematic fields
#
pvor = calc_vor(up_rot,vp_rot,dx,dy) #........................................................ perturbation vor [nlev,nlat,nlon]
pdiv = calc_div(up_rot,vp_rot,dx,dy) #........................................................ perturbation div [nlev,nlat,nlon]
pstr = calc_str(up_rot,vp_rot,dx,dy) #........................................................ perturbation str [nlev,nlat,nlon]
pshr = calc_shr(up_rot,vp_rot,dx,dy) #........................................................ perturbation shr [nlev,nlat,nlon]
#
##############################################################################################
#
# Write to netCDF file (presumed to not exist, will be created, no
# clobbering but errors if file exists)
#
# perturbation-variables to pert_outfile_name
#
nc_out = Dataset( #...................................................... Dataset object for output
                  pert_outfile_name  , # Dataset input: Output file name
                  "w"              , # Dataset input: Make file write-able
                  format="NETCDF4" , # Dataset input: Set output format to netCDF4
                )
# Dimensions
nc_lat  = nc_out.createDimension( #......................................... Output dimension
                                  "lat" , # nc_out.createDimension input: Dimension name 
                                  None   # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                )
nc_lon  = nc_out.createDimension( #......................................... Output dimension
                                  "lon" , # nc_out.createDimension input: Dimension name 
                                  None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                )
nc_lev = nc_out.createDimension( #.......................................... Output dimension
                                 "lev" , # nc_out.createDimension input: Dimension name 
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                               )
# Variables
P_VOR = nc_out.createVariable( #.................................... Output variable
                               "P_VOR"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
P_DIV = nc_out.createVariable( #.................................... Output variable
                               "P_DIV"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
P_STR = nc_out.createVariable( #.................................... Output variable
                               "P_STR"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
P_SHR = nc_out.createVariable( #.................................... Output variable
                               "P_SHR"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
XLAT_M = nc_out.createVariable( #.................................... Output variable
                               "XLAT_M"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"         # nc_out.createVariable input: Variable dimension
                               )
                             )
XLONG_M = nc_out.createVariable( #.................................... Output variable
                                 "XLONG_M"  , # nc_out.createVariable input: Variable name 
                                 "f8"          , # nc_out.createVariable input: Variable format 
                                 ( 
                                   "lat"       , # nc_out.createVariable input: Variable dimension
                                   "lon"         # nc_out.createVariable input: Variable dimension
                                 )
                               )
# Fill netCDF arrays via slicing
P_VOR[:,:,:] = pvor
P_DIV[:,:,:] = pdiv
P_STR[:,:,:] = pstr
P_SHR[:,:,:] = pshr
XLAT_M[:,:] = lat
XLONG_M[:,:] = lon
# Close netCDF file
nc_out.close()
#
#########################################################################
#
# END
#
