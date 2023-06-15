# dim_coord_swap: assign dimension names and coordinate values from one xarray.Dataset() variable
#                 to another xarray.Dataset() variable
#
# INPUTS:
#
# recDat: xarray.Dataset() object receiving dimension names and coordinates from donor
# donDat: xarray.Dataset() object donating dimension names and coordinates to receiver
#
# OUTPUTS:
#
# recDat input xarray.Dataset(), with dimension names and coordinate values from donDat
#
# DEPENDENCIES:
#
# wrf-python
# xarray (implicit, also dependency of wrf-python)
#
# NOTE: This function presumes recDat and donDat *should* have identical dimension names
#       and coordinate values. If the number of dimensions does not match, will error and
#       return without swapping dimensions/coordinates
def dim_coord_swap(recDat,donDat):
    import wrf
    # check if number of dimensions of recDat and donDat match
    # this is the only sanity check applied, if it passes we are
    # assuming that the dimension names should be identical AND
    # both datasets should have the same coordinate values
    if len(recDat.dims) == len(donDat.dims):
        # create dictionary for swapping recDat dimensions with
        # donDat dimensions
        dimDict = {}
        for i in range(len(recDat.dims)):
            dimDict[recDat.dims[i]] = donDat.dims[i]
        # swap dimensions
        recDat=recDat.rename(dimDict)
        # swap coordinates
        recDat=recDat.assign_coords(donDat.coords)
        return recDat
    else:
        print('number of receiever dimensions ({:d})'.format(len(recDat.dims)) +
              ' does not match number of donor dimensions ({:d})'.format(len(donDat.dims)))
        print('no dimension/coordinate swapping applied')
        return recDat

# get_wrf_slp: given the netCDF4 Dataset() handle for a WRF file, produces the sea-level pressure
#
# INPUTS:
#
# wrfHdl: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# slp: sea-level pressure in (ny,nx) dimension (hPa)
#
# DEPENDENCIES:
#
# numpy
# netCDF4.Dataset()
# wrf(-python)
# analysis_dependiencies.dim_coord_swap()
#
# NOTE: Draws slp from wrf.getvar() only to provide a donor-variable for dim_coord_swap()
#       to provide appropriate dimensions
# and coordinates 
def get_wrf_slp(wrfHdl):
    import numpy as np
    from netCDF4 import Dataset
    import wrf
    # Define base-state pot. temperature as 300 K
    # NOTE: This is different from the 'T00' variable in the WRF file, which is specific
    #       to an internal parameter used for a hypothetical WRF profile. See dev notes
    #       under 'derived variables for wrf-python'/'Computing temperature correctly'
    #       for details
    baseTp = 300.  # (K)
    # Define other constants
    R = 287.053  # dry gas constant (J/K*kg)
    cp = 1005.  # heat capacity of air at constant pressure (J/K*kg)
    kappa = R/cp  # (dimensionless)
    # Draw necessary fields to produce slp:
    #  Z: geopotential height (m)
    #  dynaTp: 'dynamic', or perturbation pot. temperature added to baseTp to derive full Tp (K)
    #  P: total pressure (baseP + dynaP) (Pa)
    #  baseP: base-state pressure - this is directly from 'P00' stored in WRF file (Pa)
    #  qVap: water vapor mixing ratio (kg/kg)
    Z = np.asarray(wrf.getvar(wrfHdl,'z')).squeeze()
    dynaTp = np.asarray(wrfHdl.variables['T']).squeeze()
    P = np.asarray(wrf.getvar(wrfHdl,'p')).squeeze()
    baseP = np.asarray(wrfHdl.variables['P00']).squeeze()
    qVap = np.asarray(wrfHdl.variables['QVAPOR']).squeeze()
    # Compute Tp (K)
    Tp = dynaTp + baseTp
    # Compute temperature T (K)
    T = Tp * np.power(P / baseP, kappa)
    # Derive slp from input (Z,T,P,qVap)
    slp = wrf.slp(Z, T, P, qVap)  # hPa
    # Draw (incorrect) slp from wrf.getvar(), which has appropriate dimension names and
    # coordinates
    dimCoordDonor = wrf.getvar(wrfHdl,'slp')
    # Swap dimension names and coordinates from dimCoordDonor
    slp = dim_coord_swap(slp,dimCoordDonor)
    # Return slp
    return slp


# get_wrf_rh: given the netCDF4 Dataset() handle for a WRF file, produces the relative humidity
#
# INPUTS:
#
# wrfHdl: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# rh: relative humidity in (nz,ny,nx) dimension (%)
#
# DEPENDENCIES:
#
# numpy
# netCDF4.Dataset()
# wrf(-python)
# analysis_dependencies.dim_coord_swap()
def get_wrf_rh(wrfHdl):
    import numpy as np
    from netCDF4 import Dataset
    import wrf
    # Define base-state pot. temperature as 300 K
    # NOTE: This is different from the 'T00' variable in the WRF file, which is specific
    #       to an internal parameter used for a hypothetical WRF profile. See dev notes
    #       under 'derived variables for wrf-python'/'Computing temperature correctly'
    #       for details
    baseTp = 300.  # (K)
    # Define other constants
    R = 287.053  # dry gas constant (J/K*kg)
    cp = 1005.  # heat capacity of air at constant pressure (J/K*kg)
    kappa = R/cp  # (dimensionless)
    # Draw necessary fields to produce slp:
    #  dynaTp: 'dynamic', or perturbation pot. temperature added to baseTp to derive full Tp (K)
    #  P: total pressure (baseP + dynaP) (Pa)
    #  baseP: base-state pressure - this is directly from 'P00' stored in WRF file (Pa)
    #  qVap: water vapor mixing ratio (kg/kg)
    dynaTp = np.asarray(wrfHdl.variables['T']).squeeze()
    P = np.asarray(wrf.getvar(wrfHdl,'p')).squeeze()
    baseP = np.asarray(wrfHdl.variables['P00']).squeeze()
    qVap = np.asarray(wrfHdl.variables['QVAPOR']).squeeze()
    # Compute Tp (K)
    Tp = dynaTp + baseTp
    # Compute temperature T (K)
    T = Tp * np.power(P / baseP, kappa)
    # Derive rh from input (qVap,P,T)
    rh = wrf.rh(qVap, P, T)  # (%)
    # Swap dimension names and coordinates from wrf.getvar(wrfHdl,'p'), which has
    # same dimension and coordinates that rh should have
    dimCoordDonor = wrf.getvar(wrfHdl,'p')
    rh = dim_coord_swap(rh,dimCoordDonor)
    # Return rh
    return rh
# gen_wrf_proj: generate a wrf.WrfProj() object from metadata contained in a WRF file
#
# INPUTS:
#
# wrfHDL: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# wrfProjObj: wrf.WrfProj() object containing WRF grid's map projection properties.
#
# DEPENDENCIES:
#
# netCDF4.Dataset()
# wrf.WrfProj()
def gen_wrf_proj(wrfHdl):
    from netCDF4 import Dataset
    from wrf import WrfProj
    
    return WrfProj(map_proj=wrfHdl.MAP_PROJ,
                   truelat1=wrfHdl.TRUELAT1,
                   truelat2=wrfHdl.TRUELAT2,
                   moad_cen_lat=wrfHdl.MOAD_CEN_LAT,
                   stand_lon=wrfHdl.STAND_LON,
                   pole_lat=wrfHdl.POLE_LAT,
                   pole_lon=wrfHdl.POLE_LON,
                   dx=wrfHdl.DX,
                   dy=wrfHdl.DY
                  )
# gen_cartopy_proj: generate a cartopy.crs() object from metadata contained in a WRF file
#
# INPUTS:
#
# wrfHDL: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# crsProjObj: cartopy.crs() object containing WRF grid's map projection properties
#
# DEPENDENCIES:
#
# netCDF4.Dataset()
# cartopy.crs()
#
# NOTES:
# I am following the advice given in this post to explicitly define an Earth SPHERE for
# the projection by defining the cartopy.crs.Globe() with a semimajor_axis and
# semiminor_axis of 6370000. m to match WRF's definition of the Earth as a sphere
# rather than as an ellipsoid.
# https://fabienmaussion.info/2018/01/06/wrf-projection/
# I don't see any significant difference with/without the explicit definition of an
# Earth sphere (lat/lon pass back and forth between either projection and
# cartopy.crs.PlateCarree() with differences on the order of 1.0e-9 degrees), but
# since this is apparently the correct way to do it and it doesn't generate large
# differences, I'm going to go with it in-case my proj <--> PlateCarree tests are
# being done incorrectly. I'm doing no harm with it, as far as I can tell.
def gen_cartopy_proj(wrfHDL):
    from netCDF4 import Dataset
    from cartopy import crs as ccrs
    # WRF is assumed on a Lambert Conformal projection, if it's not the routine
    # will report an error and return None. You can add more projection options
    # as you run into them, but the most common projection is Lambert Conformal
    # so I'm only coding that one here.
    if (wrfHDL.MAP_PROJ == 1) & (wrfHDL.MAP_PROJ_CHAR == "Lambert Conformal"):
        return ccrs.LambertConformal(
                                     central_longitude=wrfHDL.CEN_LON,
                                     central_latitude=wrfHDL.CEN_LAT,
                                     standard_parallels=(wrfHDL.TRUELAT1,wrfHDL.TRUELAT2),
                                     globe=ccrs.Globe(
                                                      ellipse='sphere',
                                                      semimajor_axis=6370000.,
                                                      semiminor_axis=6370000.
                                                     )
                                    )
    else:
        print('FAILED: UNKNOWN PROJECTION ENCOUNTERED:')
        print('.MAP_PROJ={:d}'.format(wrfHDL.MAP_PROJ))
        print('.MAP_PROJ_CHAR=' + wrfHDL.MAP_PROJ_CHAR)
        return None

