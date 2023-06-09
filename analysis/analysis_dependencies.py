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

# get_wrf_tk: given the netCDF4 Dataset() handle for a WRF file, produces the temperature (K)
#
# INPUTS:
#
# wrfHdl: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# tk: temperature in (nz,ny,nx) dimension (K)
#
# DEPENDENCIES:
#
# numpy
# xarray
# netCDF4.Dataset()
# wrf(-python)
# analysis_dependencies.dim_coord_swap()
def get_wrf_tk(wrfHdl):
    import numpy as np
    import xarray as xr
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
    tk = Tp * np.power(P / baseP, kappa)
    # Assert tk as xarray.DataArray() object
    tk = xr.DataArray(tk)
    # Swap dimension names and coordinates from wrf.getvar(wrfHdl,'p'), which has
    # same dimension and coordinates that tk should have
    dimCoordDonor = wrf.getvar(wrfHdl,'p')
    tk = dim_coord_swap(tk,dimCoordDonor)
    # Return tk
    return tk

# get_wrf_th: given the netCDF4 Dataset() handle for a WRF file, produces the potential
# temperature (K)
#
# INPUTS:
#
# wrfHdl: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# th: potential temperature in (nz,ny,nx) dimension (K)
#
# DEPENDENCIES:
#
# numpy
# xarray
# netCDF4.Dataset()
# wrf(-python)
# analysis_dependencies.dim_coord_swap()
#
# NOTE: This function may not be necessary, since wrf-python's wrf.getvar() seems
#       to produce a reasonable potential temperature field
def get_wrf_th(wrfHdl):
    import numpy as np
    import xarray as xr
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
    # Compute th (K)
    th = dynaTp + baseTp
    # Assert th as xarray.DataArray() object
    th = xr.DataArray(th)
    # Swap dimension names and coordinates from wrf.getvar(wrfHdl,'p'), which has
    # same dimension and coordinates that th should have
    dimCoordDonor = wrf.getvar(wrfHdl,'p')
    th = dim_coord_swap(th,dimCoordDonor)
    # Return th
    return th


# get_wrf_ss: given the netCDF4.Dataset() file handle of a WRF file, compute the static stability
#             as per: 
#                 https://www.ncl.ucar.edu/Document/Functions/Contributed/static_stability.shtml
#             see reference:
#                 Bluestein, H.B. (1992): Synoptic-Dynamic Meteorology in Midlatitudes
#                                         Volume 1: Principles of Kinematics and Dynamics
#             equation:
#                 s = -T*d[log(theta)]/dp = -(T/theta)*d(theta)/dp
#
# INPUTS:
#
# wrfHDL: netCDF4.Dataset() file handle for WRF file
#
# OUTPUTS:
#
# statStab: 3D static stability (NaN on upper and lower bounds)
#
# DEPENDENCIES:
#
# numpy
# xarray
# wrf-python
# analysis_dependencies.dim_coord_swap()
#
# NOTES:
#
# Presumes 3D variables are in (nz,ny,nx) dimension format with dimension names as
# file handle attributes of (bottom_top, south_north, west_east)
def get_wrf_ss(wrfHDL):
    import numpy as np
    import xarray as xr
    import wrf
    # Define dimension sizes (presumed dimension names)
    nz = wrfHDL.dimensions['bottom_top'].size
    ny = wrfHDL.dimensions['south_north'].size
    nx = wrfHDL.dimensions['west_east'].size
    # Define static stability as np.nan array
    statStab = np.nan * np.ones((nz,ny,nx))
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
    # Extract dynamic potential temperature from wrfHDL and compute total potential
    # temperature as dynTp + baseTp
    Tp = np.asarray(wrfHDL.variables['T']).squeeze() + baseTp
    # Extract full pressure and base pressure (Pa)
    P = np.asarray(wrf.getvar(wrfHDL,'p')).squeeze()
    baseP = np.asarray(wrfHDL.variables['P00']).squeeze()
    # Compute temperature (K)
    T = Tp * np.power(P / baseP, kappa)
    # Loop through vertical levels
    for k in range(1, nz-1):
        # Compute vertical gradient d(Tp)/dP on level k
        dTp_dP = np.divide(Tp[k+1,:,:].squeeze() - Tp[k-1,:,:].squeeze(),
                           P[k+1,:,:].squeeze() - P[k-1,:,:].squeeze())
        # Compute static stability on level k
        statStab[k,:,:] = np.multiply(np.divide(-T[k,:,:].squeeze(), Tp[k,:,:].squeeze()),
                                      dTp_dP)
    # Assert statStab as xarray.DataArray() object
    statStab = xr.DataArray(statStab)
    # Perform dimension/coordinate swap with pressure, which has appropriate dim/coord values
    statStab = dim_coord_swap(statStab,wrf.getvar(wrfHDL,'p'))
    # Return statStab
    return statStab


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
# get_xsect: return a cross-section between a chosen starting/ending latitude/longitude for
#            a provided xarray.DataArray() variable, and lists of the latitude/longigude
#            points along the cross-section. Presumes pressure-based vertical coordinate.
#
# INPUTS:
#
# wrfHDL: netCDF4.Dataset() file handle for WRF file
# var3D:  3D xarray.DataArray() of variable for cross-section, presumed to have the
#         same dimensions and coordinates as pressure. Will swap dimensions/coordinates
#         to pressure dimensions/coordinates automatically
# latBeg: beginning latitude of cross-section (deg)
# lonBeg: beginning longitude of cross-section (deg)
# latEnd: ending latitude of cross-section (deg)
# lonEnd: ending longitude of cross-section (deg)
#
# OUTPUTS:
#
# xSect: 2D cross-section of var3D from (latBeg,lonBeg) to (latEnd,lonEnd)
# latList: list of latitudes at each cross-section point
# lonList: list of longitudes at each cross-section point
#
# DEPENDENCIES:
#
# wrf-python
# numpy
# xarray (implicit, also a dependency of wrf-python)
# analysis_dependencies.gen_wrf_proj()
# analysis_dependencies.dim_coord_swap()
def get_xsect(wrfHDL, var3D, latBeg, lonBeg, latEnd, lonEnd):
    import wrf
    import numpy as np
    # draw pressure from wrf.getvar() for comparison of dimensions/coordinates
    p = wrf.getvar(wrfHDL,'p')
    # assert dimCoordEqual as True
    dimCoordEqual = True
    # check dimension names: if names are not equivalent, set dimCoordEqual to
    # False
    if p.dims != var3D.dims: dimCoordEqual = False
    # check coordinate names: if names are not equivalent, set dimCoordEqual to
    # False, else check equivalence of each coordinate's values
    # NOTE: coordinates may be equivalent but in a different order, so we can't
    #       just do a straightforward equivalence test, we have to make sure
    #       all coordinate names exist in both lists regardless of order, so
    #       list of keys is sorted() first.
    if sorted(list(p.coords.keys())) == sorted(list(var3D.coords.keys())):
        for coord in sorted(list(var3D.coords.keys())):
            if not p.coords[coord].identical(var3D.coords[coord]): dimCoordEqual = False
    else:
        dimCoordEqual = False
    # if dimCoordEqual is False, perform dim_coord_swap() to borrow dimensions
    # and coordinates from pressure
    if not dimCoordEqual:
        var3D = dim_coord_swap(var3D,p)
    # define inputs to cross-section
    plevs = np.arange(10000., 102000.1, 5000.)  # levels
    stag = 'm'  # stagger
    proj = gen_wrf_proj(wrfHDL)  # projection
    ptBeg = wrf.CoordPair(lat=latBeg, lon=lonBeg)  # start_point
    ptEnd = wrf.CoordPair(lat=latEnd, lon=lonEnd)  # end_point
    # generate cross-section with lat/lon points
    xSect = wrf.vertcross(
                          field3d     = var3D,
                          vert        = p,
                          levels      = plevs,
                          missing     = np.nan,
                          wrfin       = wrfHDL,
                          stagger     = stag,
                          projection  = proj,
                          start_point = ptBeg,
                          end_point   = ptEnd,
                          latlon = True)
    # extract latitude and longitude along cross-section from
    # xSect.xy_loc strings
    latList=[]
    lonList=[]
    for point in xSect.xy_loc.values:
        pointLatLonStr = point.latlon_str().split(', ')
        latList.append(float(pointLatLonStr[0]))
        lonList.append(float(pointLatLonStr[1]))
    # return xSect, latList, lonList
    return xSect, latList, lonList


# cross_section_plot: Generates a series of 2-panel plots of
#                         left: cross-section from (latBeg,lonBeg) to (latEnd,lonEnd) of xSectCont contours and xSectShad shading
#                         right: sea-level pressure and thickness contours with sea-level pressure perturbation shaded, and the cross-section line
#                     Figure is 2 columns by X rows, with a row for each cross-section, up to 3 cross-sections
#
# REQUIRED INPUTS:
#
# wrfHDL: netCDF.Dataset() file-handle for WRF file for dimension/coordinate data
# latBegList: list of beginning latitude for each cross-section (max: 3)
# lonBegList: list of beginning longitudes for each cross-section (max: 3)
# latEndList: list of ending latitude for each cross-section (max: 3)
# lonEndList: list of ending longitudes for each cross-section (max: 3)
# xSectContVariable: 3D variable for producing cross-section contours (xarray.DataArray() with dimension names and coordinate variables)
# xSectContInterval: interval values for xSectCont contours
# xSectShadVariable: 3D variable for producing cross-section shading (xarray.DataArray() with dimension names and coordinate variables)
# xSectShadInterval: interval values for xSectCont shading
# slp: 2D sea-level pressure (probably unperturbed SLP)
# slpInterval: interval values for slp contours
# thk: 2D thickness (probably unperturbed thk)
# thkInterval: interval values for thickness contours
# slpPert: 2D sea-level pressure perturbation
# slpPertInterval: interval values for slpPert shading
# datProj: cartopy.crs() projection of data
# plotProj: cartopy.crs() projection of 2D plots
#
# OPTIONAL INPUTS:
#
# xSectShadCmap: name of colormap for xSectShad (default: 'seismic')
# xSectContColor: name of color for xSectCont (default: 'black')
# slpPertCmap: name of colormap for slpPert (default: 'seismic')
# presLevMin: minimum pressure level of cross-section (default: 10000. Pa)
# presLevMax: maximum pressure level of cross-section (default: 100000. Pa)
#
# OUTPUTS:
#
# fig: figure handle containing all panels
#
# DEPENDENCIES:
#
# numpy
# wrf-python
# xarray (implicit, also dependency of wrf-python)
# matplotlib.pyplot
# matplotlib.ticker
# cartopy.crs
# cartopy.feature
# cartopy.mpl.gridliner functions: LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# analysis_dependencies.get_xsect()
def cross_section_plot(wrfHDL, latBegList, lonBegList, latEndList, lonEndList,
                       xSectContVariable, xSectContInterval, xSectShadVariable,
                       xSectShadInterval, slp, slpInterval, thk, thkInterval,
                       slpPert, slpPertInterval, datProj, plotProj,
                       xSectShadCmap='seismic', xSectContColor='black',
                       slpPertCmap='seismic', presLevMin=10000., presLevMax=100000.):
    import numpy as np
    import wrf
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from cartopy import crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    # sanity test some inputs: all of the lat/lon lists should be same length, and no longer
    # than 3, and no shorter than 1
    listLengths = [len(latBegList), len(lonBegList), len(latEndList), len(lonEndList)]
    if len(set(listLengths)) > 1:
        print('ERROR: not all lat/lon lists equal length, aborting')
        return None
    elif any(listLengths) > 3:
        print('ERROR: at least one lat/lon list too long, aborting')
        return None
    elif any(listLengths) < 1:
        print('ERROR: at least one lat/lon list size zero, aborting')
    else:
        # define number of cross-sections from listLengths (all should be equal, just pick one)
        numxSect = len(latBegList)
        # define figure panel dimensions based on numxSect:
        # 1 cross-section
        if numxSect == 1:
            leftPanels =  [
                           [0.0, 0.0, 0.5, 1.0]
                          ]
            rightPanels = [
                           [0.5, 0.0, 0.5, 1.0]
                          ]
        # 2 cross-sections
        elif numxSect == 2:
            leftPanels =  [
                           [0.0, 0.5, 0.4, 0.4],
                           [0.0, 0.0, 0.4, 0.4]
                          ]
            rightPanels = [
                           [0.5, 0.5, 0.5, 0.4],
                           [0.5, 0.0, 0.5, 0.4]
                          ]
        # 3 cross-sections
        elif numxSect == 3:
            leftPanels =  [
                           [0.0, 0.7, 0.4, 0.3],
                           [0.0, 0.35, 0.4, 0.3],
                           [0.0, 0.0, 0.4, 0.3]
                          ]
            rightPanels = [
                           [0.5, 0.7, 0.5, 0.3],
                           [0.5, 0.35, 0.5, 0.3],
                           [0.5, 0.0, 0.5, 0.3]
                          ]
    # loop through cross-sections
    for i in range(numxSect):
        # generate 2-panel plots with (left)  cross-section of initial geopotential height perturbations, and
        #                             (right) cross-section line on SLP/thickness plot with SLP perturbation
        # define latBeg, lonBeg, latEnd, lonEnd from lists
        latBeg = latBegList[i]
        lonBeg = lonBegList[i]
        latEnd = latEndList[i]
        lonEnd = lonEndList[i]
        # define lat/lon lines for SLP/thickness plot
        latLines = np.arange(-90., 90., 5.)
        lonLines = np.arange(-180., 180. ,5.)
        # define cross-section line-color based on value of i
        if i == 0:
            xSectLineColor = 'green'
        elif i == 1:
            xSectLineColor = 'orange'
        elif i == 2:
            xSectLineColor = 'magenta'
        # define figure for all panels (allow longer y-dimension for more cross-sections)
        fig = plt.figure(figsize=(12,5*numxSect))
        # define cross-section and lat/lon values
        # contours
        xSectCont, latList, lonList = get_xsect(wrfHDL,
                                                xSectContVariable,
                                                latBeg, lonBeg, latEnd, lonEnd)
        # shading (only pulling first return-value since latList and lonList should be the same)
        xSectShad                   = get_xsect(wrfHDL,
                                                xSectShadVariable,
                                                latBeg, lonBeg, latEnd, lonEnd)[0]
        # plot cross-section (left panel)
        ax = fig.add_axes(rect=leftPanels[i])
        shd = ax.contourf(xSectShad, levels=xSectShadInterval, cmap=xSectShadCmap, extend='both')
        con = ax.contour(xSectCont, levels=xSectContInterval, colors=xSectContColor)
        yTickIndex = np.where((xSectShad.coords['vertical'].values >= presLevMin) &
                              (xSectShad.coords['vertical'].values <= presLevMax))[0]
        yTickVals = xSectShad.coords['vertical'].values[yTickIndex]
        ax.set_yticks(ticks=yTickIndex, labels=yTickVals * 0.01)  # tick values in hPa
        ax.set_ylim((np.min(yTickIndex),np.max(yTickIndex)))
        ax.invert_yaxis()
        ax.set_title('cross section {:d}'.format(i) + ' (' + xSectLineColor + ')')
        plt.colorbar(ax=ax, mappable=shd)
        # plot SLP and thickness with SLP perturbation and cross-section line (right panel)
        ax = fig.add_axes(rect=rightPanels[i], projection=datProj)
        # define 2D latitude and longitude arrays from wrfHDL
        lat2D = np.asarray(wrfHDL.variables['XLAT']).squeeze()
        lon2D = np.asarray(wrfHDL.variables['XLONG']).squeeze()
        # assert lon2D as 0 to 360 format
        fix = np.where(lon2D < 0.)
        lon2D[fix] = lon2D[fix] + 360.
        # plot slp perturbation as shading at 60% transparancy
        shd = ax.contourf(lon2D, lat2D, slpPert, slpPertInterval, cmap='seismic', extend='both',
                          alpha=0.4, transform=plotProj)
        # plot slp as black contours
        con1 = ax.contour(lon2D, lat2D, slp, slpInterval, colors='black', linewidths=1.0,
                          transform=plotProj)
        # label every other slp contour
        ax.clabel(con1, levels=slpInterval[::2])
        # plot thk as green contours
        con2 = ax.contour(lon2D, lat2D, thk, thkInterval, colors='green', linewidths=0.75,
                          transform=plotProj)
        # add coastline in brown
        ax.add_feature(cfeature.COASTLINE, color='brown', linewidth=1.5)
        ax.set_title('cross section {:d}'.format(i))
        # add gridlines
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
        # plot cross-section line
        ax.plot(lonBeg, latBeg, 'o', transform=plotProj, color=xSectLineColor)
        ax.plot(lonEnd, latEnd, 'o', transform=plotProj, color=xSectLineColor)
        ax.plot((lonBeg, lonEnd), (latBeg, latEnd), transform=plotProj, color=xSectLineColor)
    # return figure handle
    return fig


