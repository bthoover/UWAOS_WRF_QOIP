##############################################################################################
#
# PYTHON 3 PROGRAM
#
# Computes sensitivity to kinematic fields: vorticity, divergence, stretching deformation, and
# shearing deformation, from sensitivity to u- and v-components of the wind.
#
# This is achieved by computing the equivalent kinematic component from the sensitivity with
# respect to u- and v-components of the wind, and then computing minus the inverse-laplacian
# of that quantity to recover the sensitivity. This requires both the adjoint output and the
# wrfinput file, since there is necessary information about the grid projection in the input
# file that is needed to properly rotate the sensitivity to u- and v-components into earth-
# relative coordinates.
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
from adj_diagnostics import sensitivity_to_kinematic_by_level #............................... sensitivity solver
#
##############################################################################################
#
# Ingest user input
#
adj_file = input() #.......................................................................... full path to gradient_wrfplus_d01 file
inp_file = input() #.......................................................................... full path to wrfinput_d01 file
adj_outfile_name = input() #.................................................................. full path+filename for sensitivity output
#
##############################################################################################
#
# Extract sensitivity fields from adj_file and grid information from inp_file
#
adj_hdl = Dataset(adj_file) #................................................................. adj_file handle
inp_hdl = Dataset(inp_file) #................................................................. inp_file handle
# Extract grid information
lat = wrf.getvar(inp_hdl,'lat') #............................................................. grid latitudes [nlat,nlon]
lon = wrf.getvar(inp_hdl,'lon') #............................................................. grid longitudes [nat,nlon]
cen_lon = inp_hdl.CEN_LON #................................................................... grid center longitude
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
# Extract sensitivity information
au = adj_hdl.variables['A_U'] #............................................................... zonal flow sensitivity [nlev,nlat,nlon]
av = adj_hdl.variables['A_V'] #............................................................... merid flow sensitivity [nlev,nlat,nlon]
# Destagger wind grids to mass points
au = wrf.destagger(au,stagger_dim=3)
av = wrf.destagger(av,stagger_dim=2)
# Rotate wind grids to earth coordinates
rot = wrf.uvmet(au,av,lat,lon,cen_lon,cone) #................................................. rotated (au,av) fields
au_rot = rot[0,:,:,:,:] #..................................................................... rotated au
av_rot = rot[1,:,:,:,:] #..................................................................... rotated av
# Define grid dimensions
nz,ny,nx = np.shape(np.asarray(au).squeeze()) #............................................... grid dimensions
#
##############################################################################################
#
# Compute forcing-terms for sensitivity to kinematic fields. Each of these is the equivalent
# kinematic field computed from (au,av) instead of (u,v). The sensitivity to the kinematic
# field (kin) is computed as -<inverse-laplacian>(kin).
#
avor_forcing = calc_vor(au_rot,av_rot,dx,dy) #................................................ forcing for avor [nlev,nlat,nlon]
adiv_forcing = calc_div(au_rot,av_rot,dx,dy) #................................................ forcing for adiv [nlev,nlat,nlon]
astr_forcing = calc_str(au_rot,av_rot,dx,dy) #................................................ forcing for astr [nlev,nlat,nlon]
ashr_forcing = calc_shr(au_rot,av_rot,dx,dy) #................................................ forcing for ashr [nlev,nlat,nlon]
#
# Compute sensitivity to vor, div, str, and shr
#
avor = np.nan * np.ones((nz,ny,nx)) #......................................................... sensitivity with respect to vor (initialized to nan) [nlev,nlat,nlon]
adiv = np.nan * np.ones((nz,ny,nx)) #......................................................... sensitivity with respect to div (initialized to nan) [nlev,nlat,nlon]
astr = np.nan * np.ones((nz,ny,nx)) #......................................................... sensitivity with respect to str (initialized to nan) [nlev,nlat,nlon]
ashr = np.nan * np.ones((nz,ny,nx)) #......................................................... sensitivity with respect to shr (initialized to nan) [nlev,nlat,nlon]
# Loop through levels, compute sensitivity on level (psi), assign to 3D field
for k in range(nz):
    print(k,nz-1)
    psi = sensitivity_to_kinematic_by_level(avor_forcing,k,nx,ny,nz,inp_hdl.DX,inp_hdl.DY,np.asarray(mf).squeeze())
    avor[k,:,:]=psi
    psi = sensitivity_to_kinematic_by_level(adiv_forcing,k,nx,ny,nz,inp_hdl.DX,inp_hdl.DY,np.asarray(mf).squeeze())
    adiv[k,:,:]=psi
    psi = sensitivity_to_kinematic_by_level(astr_forcing,k,nx,ny,nz,inp_hdl.DX,inp_hdl.DY,np.asarray(mf).squeeze())
    astr[k,:,:]=psi
    psi = sensitivity_to_kinematic_by_level(ashr_forcing,k,nx,ny,nz,inp_hdl.DX,inp_hdl.DY,np.asarray(mf).squeeze())
    ashr[k,:,:]=psi
#
##############################################################################################
#
# Write to netCDF file (presumed to not exist, will be created, no
# clobbering but errors if file exists)
#
# Adjoint-variables to adj_outfile_name
#
nc_out = Dataset( #...................................................... Dataset object for output
                  adj_outfile_name  , # Dataset input: Output file name
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
A_VOR = nc_out.createVariable( #.................................... Output variable
                               "A_VOR"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
A_DIV = nc_out.createVariable( #.................................... Output variable
                               "A_DIV"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
A_STR = nc_out.createVariable( #.................................... Output variable
                               "A_STR"  , # nc_out.createVariable input: Variable name 
                               "f8"          , # nc_out.createVariable input: Variable format 
                               ( 
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
A_SHR = nc_out.createVariable( #.................................... Output variable
                               "A_SHR"  , # nc_out.createVariable input: Variable name 
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
A_VOR[:,:,:] = avor
A_DIV[:,:,:] = adiv
A_STR[:,:,:] = astr
A_SHR[:,:,:] = ashr
XLAT_M[:,:] = lat
XLONG_M[:,:] = lon
# Close netCDF file
nc_out.close()
#
#########################################################################
#
# END
#
