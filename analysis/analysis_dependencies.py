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


# get_wrf_eth: given the netCDF4 Dataset() handle for a WRF file, produces the 
# equivalent potential temperature (K)
#
# INPUTS:
#
# wrfHdl: WRF file netCDF4.Dataset() handle
#
# OUTPUTS:
#
# eth: equivalent potential temperature in (nz,ny,nx) dimension (K)
#
# DEPENDENCIES:
#
# numpy
# xarray
# netCDF4.Dataset()
# wrf(-python)
# analysis_dependencies.get_wrf_tk()
# analysis_dependencies.dim_coord_swap()
def get_wrf_eth(wrfHdl):
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset
    import wrf
    # define input variables to wrf.eth():
    # P: Pressure
    P = np.asarray(wrf.getvar(wrfHdl,'p')).squeeze()
    # qVap: Vapor Pressure
    qVap = np.asarray(wrfHdl.variables['QVAPOR']).squeeze()
    # T: Temperature (K)
    T = np.asarray(get_wrf_tk(wrfHdl)).squeeze()
    # derive eth from input (qVap,T,P)
    eth = wrf.eth(qVap, T, P, meta=False, units='K')  # (K)
    # assert eth as xarray.DataArray() object
    eth = xr.DataArray(eth)
    # Swap dimension names and coordinates from wrf.getvar(wrfHdl,'p'), which has
    # same dimension and coordinates that eth should have
    dimCoordDonor = wrf.getvar(wrfHdl,'p')
    eth = dim_coord_swap(eth,dimCoordDonor)
    # Return eth
    return eth


# get_uvmet: given the netCDF4.Dataset() file handle of a WRF file, extract the (u,v) wind
#            components and perform two operations on them before returning: (1) destagger
#            to mass-points, and (2) rotate to Earth-relative coordinates. See:
#
#            https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
#
#            Returns met(u,v) as xarray.DataArray() with dimensions and coordinates for
#            producing cross-sections (borrowed from pressure variable)
#
# INPUTS:
#
# wrfHDL: netCDF4.Dataset() file handle for WRF file
#
# OUTPUTS:
#
# uMet: u-component on mass-points and Earth-relative coordinates
# vMet: v-component on mass-points and Earth-relative coordinates
#
# DEPENDENCIES:
#
# numpy
# xarray
# wrf-python
# analysis_dependencies.dim_coord_swap()
# NOTES:
#
# Presumes 3D variables are in (nz,ny,nx) dimension format with dimension names as
# file handle attributes of (bottom_top, south_north, west_east)
def get_uvmet(wrfHDL):
    import numpy as np
    import xarray as xr
    import wrf
    # define latitude and longitude
    lat = np.asarray(wrfHDL.variables['XLAT']).squeeze()
    lon = np.asarray(wrfHDL.variables['XLONG']).squeeze()
    # assert longitude as 0-to-360 format
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define central longitude from STAND_LON
    cen_lon = wrfHDL.STAND_LON
    # define standard parallels from TRUELAT1, TRUELAT2
    true_lat1 = wrfHDL.TRUELAT1
    true_lat2 = wrfHDL.TRUELAT2
    # define radians-per-degree
    radians_per_degree = np.pi/180.
    # define map-factors on mass-points
    mf = wrfHDL.variables['MAPFAC_M']
    # define dx, dy as [nlat,nlon] grids normalized by mf
    dx = wrfHDL.DX*np.power(mf, -1.)
    dy = wrfHDL.DY*np.power(mf, -1.)
    # define conic projection value based on true_lat1 and true_lat2
    if((abs(true_lat1 - true_lat2) > 0.1) and
       (abs(true_lat2 - 90.) > 0.1)):
        cone = (np.log(np.cos(true_lat1 * radians_per_degree))
              - np.log(np.cos(true_lat2 * radians_per_degree)))

        cone = (cone /
                (np.log(np.tan((45. - abs(true_lat1 / 2.)) * radians_per_degree))
                - np.log(np.tan((45. - abs(true_lat2 / 2.)) * radians_per_degree))))
    else:
        cone = np.sin(abs(true_lat1) * radians_per_degree)
    # define WRF u- and v-components (staggered, grid-relative)
    uGrd = wrfHDL.variables['U']
    vGrd = wrfHDL.variables['V']
    # destagger to mass-points, fields in [time,lev,lon,lat]-dimension
    uGrdM = wrf.destagger(uGrd, stagger_dim=3)
    vGrdM = wrf.destagger(vGrd, stagger_dim=2)
    # rotate to Earth-relative coordinate, fields in [var,time,lev,lon,lat]-dimension
    # asserted as squeezed numpy array
    rot = wrf.uvmet(uGrdM, vGrdM, lat, lon, cen_lon, cone)
    uMet = np.asarray(rot[0,:,:,:,:]).squeeze()
    vMet = np.asarray(rot[1,:,:,:,:]).squeeze()
    # assert uMet, vMet as xarray.DataArray() object with blank dimension-names and coordinate values
    uMet = xr.DataArray(uMet)
    vMet = xr.DataArray(vMet)
    # perform dimension/coordinate swap with pressure, which has appropriate dim/coord values
    # .squeeze() reduces fields to [lev,lon,lat]-dimension
    uMet = dim_coord_swap(uMet.squeeze(), wrf.getvar(wrfHDL,'p'))
    vMet = dim_coord_swap(vMet.squeeze(), wrf.getvar(wrfHDL,'p'))
    # return uMet, vMat
    return uMet, vMet


# get_wrf_kinematic: given the netCDF4.Dataset() file handle of a WRF file, return a chosen kinematic
#                    quantity: 'vor', 'div', 'str', or 'shr'. Returns an xarray.DataArray() object with
#                    appropriate dimension-names and coordinate varibles borrowed from pressure.
#
# INPUTS:
#
# wrfHDL: netCDF4.Dataset() file handle of WRF file
# kinName: name of kinematic to compute ('vor', 'div', 'str', 'shr')
#
# OUTPUTS:
#
# kin: chosen kinematic field
#
# DEPENDENCIES:
#
# numpy
# xarray
# wrf-python
# analysis_dependencies.dim_coord_swap()
# analysis_dependencies.get_uvmet()
def get_wrf_kinematic(wrfHDL, kinName):
    import numpy as np
    import xarray as xr
    import wrf
    # define internal functions to calculate kinematic terms:
    #     each function takes the u- and v- grids on mass-points and in Earth-
    #     relative coordinates, and dx and dy normalized by map-factors as
    #     inputs, and returns kinemeatic field.
    def calc_div(u,v,dx,dy):
        ug  = np.asarray(u).squeeze()
        vg  = np.asarray(v).squeeze()
        dxg = np.asarray(dx).squeeze()
        dyg = np.asarray(dy).squeeze()
        nz,ny,nx = np.shape(ug)
        kin = np.nan * np.ones((nz,ny,nx))
        for k in range(nz):
            for j in range(1,ny-1,1):
                for i in range(1,nx-1,1):
                    dudx = (ug[k,j,i+1]-ug[k,j,i-1])/(2.*dxg[j,i])
                    dvdy = (vg[k,j+1,i]-vg[k,j-1,i])/(2.*dyg[j,i])
                    kin[k,j,i] = dudx + dvdy
        return kin
    def calc_vor(u,v,dx,dy):
        ug  = np.asarray(u).squeeze()
        vg  = np.asarray(v).squeeze()
        dxg = np.asarray(dx).squeeze()
        dyg = np.asarray(dy).squeeze()
        nz,ny,nx = np.shape(ug)
        kin = np.nan * np.ones((nz,ny,nx))
        for k in range(nz):
            for j in range(1,ny-1,1):
                for i in range(1,nx-1,1):
                    dvdx = (vg[k,j,i+1]-vg[k,j,i-1])/(2.*dxg[j,i])
                    dudy = (ug[k,j+1,i]-ug[k,j-1,i])/(2.*dyg[j,i])
                    kin[k,j,i] = dvdx - dudy
        return kin
    def calc_str(u,v,dx,dy):
        ug  = np.asarray(u).squeeze()
        vg  = np.asarray(v).squeeze()
        dxg = np.asarray(dx).squeeze()
        dyg = np.asarray(dy).squeeze()
        nz,ny,nx = np.shape(ug)
        kin = np.nan * np.ones((nz,ny,nx))
        for k in range(nz):
            for j in range(1,ny-1,1):
                for i in range(1,nx-1,1):
                    dudx = (ug[k,j,i+1]-ug[k,j,i-1])/(2.*dxg[j,i])
                    dvdy = (vg[k,j+1,i]-vg[k,j-1,i])/(2.*dyg[j,i])
                    kin[k,j,i] = dudx - dvdy
        return kin
    def calc_shr(u,v,dx,dy):
        ug  = np.asarray(u).squeeze()
        vg  = np.asarray(v).squeeze()
        dxg = np.asarray(dx).squeeze()
        dyg = np.asarray(dy).squeeze()
        nz,ny,nx = np.shape(ug)
        kin = np.nan * np.ones((nz,ny,nx))
        for k in range(nz):
            for j in range(1,ny-1,1):
                for i in range(1,nx-1,1):
                    dvdx = (vg[k,j,i+1]-vg[k,j,i-1])/(2.*dxg[j,i])
                    dudy = (ug[k,j+1,i]-ug[k,j-1,i])/(2.*dyg[j,i])
                    kin[k,j,i] = dvdx + dudy
        return kin
    # sanity-check kinName, fail reports error and returns None
    if kinName in ['vor', 'div', 'str', 'shr']:
        # define u- and v- components on mass-points and Earth-relative coordinates
        u, v = get_uvmet(wrfHDL)
        # define map-factors on mass-points
        mf = wrfHDL.variables['MAPFAC_M']
        # define dx, dy as [nlat,nlon] grids normalized by mf
        dx = wrfHDL.DX*np.power(mf, -1.)
        dy = wrfHDL.DY*np.power(mf, -1.)
        # define kinematic term
        if kinName == 'vor':
            kin = calc_vor(u, v, dx, dy)
        elif kinName == 'div':
            kin = calc_div(u, v, dx, dy)
        elif kinName == 'str':
            kin = calc_str(u, v, dx, dy)
        elif kinName == 'shr':
            kin = calc_shr(u, v, dx, dy)
        # assert kin as xarray.DataArray() object
        kin = xr.DataArray(kin)
        # perform dimension/coordinate swap with pressure, which has appropriate dim/coord values
        kin = dim_coord_swap(kin, wrf.getvar(wrfHDL,'p'))
        # return kin
        return kin
    else:
        print(kinName + ' not in kinematics list: vor, div, str, shr')
        return None

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
def gen_cartopy_proj(wrfHDL, cenLat=None, cenLon=None):
    from netCDF4 import Dataset
    from cartopy import crs as ccrs
    # WRF is assumed on a Lambert Conformal projection, if it's not the routine
    # will report an error and return None. You can add more projection options
    # as you run into them, but the most common projection is Lambert Conformal
    # so I'm only coding that one here.
    if (wrfHDL.MAP_PROJ == 1) & (wrfHDL.MAP_PROJ_CHAR == "Lambert Conformal"):
        # if cenLat or cenLon are None, use wrfHDL attributes, otherwise override
        # with selected values (for when attributes are defaulting to missing values)
        #
        # NOTE: Better to use MOAD_CEN_LAT and STAND_LON for cenLat and cenLon than
        #       the subdomain-specific CEN_LAT and CEN_LON values. See this note
        #       as it pertains to rotating from WRF-relative to Earth-relative wind
        #       coordinates: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
        if cenLat is None:
            cenLat = wrfHDL.MOAD_CEN_LAT
        if cenLon is None:
            cenLon = wrfHDL.STAND_LON
        return ccrs.LambertConformal(
                                     central_longitude=cenLon,
                                     central_latitude=cenLat,
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


# plan_section_plot: Generates a horizontal plan-section plot of shading, contours, and vectors as desired
#                    Figure is a single panel
#
# REQUIRED INPUTS:
#
# wrfHDL: netCDF.Dataset() file-handle for WRF file for dimension/coordinate data
# lat: [ny,nx] grid of latitudes
# lon: [ny,nx] grid of longitudes
# contVariableList: list of [ny,nx] variables for producing contours
# contIntervalList: list of interval values for each contour variable in contourVariableList
# contColorList: list of colors for each contour variable in contourVariableList
# shadVariable: [ny,nx] variable for producing shading
# shadInterval: interval values for shading
# datProj: cartopy.crs() projection of data
# plotProj: cartopy.crs() projection of 2D plots
#
# OPTIONAL INPUTS:
#
# shadCmap: name of colormap for shading (default: 'seismic')
# contLineThicknessList: list of line thicknesses for each contour variable in contourVariableList (default: 1.0)
# shadAlpha: alpha of shading (default: 1.0)
# vecColor: name of color for vectors (default: 'black')
# uVecVariable: [ny,nx] variable for u-component of vectors
# vVecVariable: [ny,nx] variable for v-component of vectors
# vectorThinning: skip-ratio for plotting vectors (e.g. vectorThinning=2 only plots [::2])
# figax: figure axis to plot to 
#
# OUTPUTS:
#
# fig: figure handle
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
def plan_section_plot(wrfHDL, lat, lon, contVariableList, contIntervalList, contColorList,
                      shadVariable, shadInterval, datProj, plotProj, shadCmap='seismic',
                      contLineThicknessList=None, shadAlpha=1.0, vecColor='black',
                      uVecVariable=None, vVecVariable=None, vectorThinning=1, figax=None):
    import numpy as np
    import wrf
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from cartopy import crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # define lat/lon lines for SLP/thickness plot
    latLines = np.arange(-90., 90., 5.)
    lonLines = np.arange(-180., 180. ,5.)
    # define figure for a single panel, if no figax provided
    if figax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': datProj}, figsize=(12,9))
    else:
        ax = figax
    # plot shading, if any
    if shadVariable is not None:
        shd = ax.contourf(lon,
                          lat,
                          shadVariable,
                          levels=shadInterval,
                          cmap=shadCmap,
                          vmin=np.min(shadInterval),
                          vmax=np.max(shadInterval),
                          extend='both',
                          transform=plotProj,
                          alpha=shadAlpha)
    else:
        shd = None
    # assert contour inputs as list if they are not lists (i.e. if a single value was passed without
    # encapsulating in a list)
    contVariableList = contVariableList if type(contVariableList)==list else [contVariableList]
    contIntervalList = contIntervalList if type(contIntervalList)==list else [contIntervalList]
    contColorList = contColorList if type(contColorList)==list else [contColorList]
    # contLineThicknessList is optional, if not set (is None) then replace with list of None
    if contLineThicknessList is not None:
        contLineThicknessList = contLineThicknessList if type(contLineThicknessList)==list else [contLineThicknessList]
    else:
        contLineThicknessList = [None] * len(contVariableList)
    # check for same length among all contour lists, if not same length, report error and do not plot contours
    contourLists=[len(contVariableList), len(contIntervalList), len(contColorList), len(contLineThicknessList)]
    if len(set(contourLists)) == 1:
        print('generating {:d} contours'.format(len(contVariableList)))
        # plot all contours with provided interval, color, and line thickness (if any, default 1.0)
        # put each contour into a list for returning to user for any further modification
        cons = []
        for i in range(len(contVariableList)):
            contVariable = contVariableList[i]
            contInterval = contIntervalList[i]
            contColor = contColorList[i]
            contLineThickness = contLineThicknessList[i] if contLineThicknessList[i] is not None else 1.
            if contVariable is not None:
                con = ax.contour(lon,
                                 lat,
                                 contVariable,
                                 levels=contInterval,
                                 colors=contColor,
                                 linewidths=contLineThickness,
                                 transform=plotProj)
            else:
                con = None
            cons.append(con)
    else:
        print('ERROR: Contour lists (Variable, Interval, Color) not same length, no contours plotted')
    # plot a colorbar for the shading, if any
    if shadVariable is not None:
        plt.colorbar(ax=ax, mappable=shd)
    # plot vectors, if any
    if (uVecVariable is not None) & (vVecVariable is not None):
        vec=ax.quiver(x=lon[::vectorThinning, ::vectorThinning],
                      y=lat[::vectorThinning, ::vectorThinning],
                      u=uVecVariable[::vectorThinning, ::vectorThinning],
                      v=vVecVariable[::vectorThinning, ::vectorThinning],
                      color=vecColor,
                      transform=plotProj)
    else:
        vec=None
    # add coastline in brown
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown', linewidth=1.5)
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
    # return figure axis handle along with shd, cons, and vec as a tuple
    return ax, (shd, cons, vec)


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

# compute_inverse_laplacian(): A function that will compute the quantity:
#
#   <inverse-laplacian>(frc)
#
# for some 2D forcing field frc in [ny,nx] space
#
# INPUTS:
#
#    wrfHDL ........................................... netCDF4.Dataset() file-handle of WRF file
#    frc .............................................. 2D [ny,nx] forcing field for inverse-laplacian
#
# OUTPUTS:
#
#    full_psi ......................................... <inverse-laplacian>(frc)
#
# NOTES:
#
#    uses numPy, wrf(-python), sciPy routines
#    uses convention of i-points are meridional and j-points are zonal
#
def compute_inverse_laplacian(wrfHDL, frc):
    #########################################################################################
    #
    # Import necessary modules
    #
    import numpy as np #..................................................................... array module
    from numpy import reshape #.............................................................. numPy array reshape routine
    import time #............................................................................ clock module
    from wrf import to_np #.................................................................. xarray-to-numPy-array conversion routine 
    import scipy as sp #..................................................................... scientific function module
    import scipy.optimize #.................................................................. sciPy function optimization routine
    import scipy.linalg #.................................................................... sciPy linear algebra routine
    import scipy.sparse #.................................................................... sciPy sparse array routine
    #
    #########################################################################################
    #
    # Initialize forcing (-kin) field, and interior and boundary conditions of solved (psi)
    # field. psi will become full_psi once it has converged.
    #
    # Start clock
    start = time.time() #.................................................................... beginning time
    #
    # FIELDS DEFINED
    #
    # Define 2D grid-dimensions
    ny,nx = np.shape(frc) #.................................................................. 2D grid-dimensions, from forcing grid
    # Define grid map-factors
    msfm = np.asarray(wrfHDL.variables['MAPFAC_M']).squeeze() #.............................. grid map factors on mass points
    # Define grid-spacing in x- and y-directions
    ds_x = wrfHDL.DX #....................................................................... grid dx
    ds_y = wrfHDL.DY #....................................................................... grid dy
    # Define inverted field: psi
    psi = np.zeros((ny, nx)) #............................................................... initialized psi (Dirichlet boundary conditions)
    psib = 0.0 #............................................................................. prescribed psi boundary value (Dirichlet boundary conditions)
    # Define full_psi, the psi array that is operated on in relaxation
    full_psi=np.zeros((ny,nx)) #............................................................. initialized full_psi
    # Define dimensions of interior-points to grid
    nxm2 = nx - 2 #.......................................................................... x-direction inner points number
    nym2 = ny - 2 #.......................................................................... y-direction inner points number
    # Define forcing field and forcing field as a flattened vector
    forcegrid=np.ndarray(shape=(nym2,nxm2)) #................................................ forcing grid (interior points only, initialized empty)
    forcevec=np.ndarray(forcegrid.size) #.................................................... forcing vector of forcegrid
    #
    # FIELDS INITIALIZED
    #
    # Initialize forcegrid, forcevec, and psi
    psi=full_psi[:,:] #...................................................................... psi (initialized to full_psi initial values)
    forcegrid = frc[1:ny-1,1:nx-1]/msfm[1:ny-1,1:nx-1]**2.0 #................................ forcegrid (initialized to frc, divided by square of map-factors)
    forcevec[:]=to_np(forcegrid).flatten() #................................................. forcevec (initialized to forcegrid as flattened array)
    # Initialize boundary conditions of forcevec
    for i in range(nym2):
        for j in range(nxm2):
            index=i*nxm2+j #................................................................. TEMPORARY VARIABLE: index of current [i,j] point in forcevec
            # Adjust values in forcevec based on prescribed boundary conditions
            if(i==0) : forcevec[index] = forcevec[index] - psib/ds_x**2.
            if(i==nxm2-1): forcevec[index] = forcevec[index] - psib/ds_x**2.
            if(j==0): forcevec[index] = forcevec[index] - psib/ds_y**2
            if(j==nym2-1): forcevec[index] = forcevec[index] - psib/ds_y**2.
    #
    #########################################################################################
    #
    # Compute inverse-laplacian of forcevec to recover psi
    #
    # Define laplacian operators
    lap = np.zeros((nxm2*nym2,nxm2*nym2),dtype=np.float64) #................................. laplacian operators field
    # Initialize laplacian operator values
    for i in range(nym2):
        for j in range(nxm2):
            index=i*nxm2 + j #............................................................... TEMPORARY VARIABLE: index of current [i,j] point in forcevec
            # Construct indices of 5-point molecule around index
            A1 = (i-1)*nxm2 + j #............................................................ TEMPORARY VARIABLE: south point
            A2 = i*nxm2 + (j-1) #............................................................ TEMPORARY VARIABLE: west point
            A3 = index #..................................................................... TEMPORARY VARIABLE: center [i,j] point
            A4 = i*nxm2 + (j+1) #............................................................ TEMPORARY VARIABLE: east point
            A5 = (i+1)*nxm2 + j #............................................................ TEMPORARY VARIABLE: north point
            # Compute laplacian for points in molecule
            if((i>0)):lap[index,A1] = 1./ds_y**2;
            if(j<nxm2-1): lap[index,A4] = 1./ds_y**2;
            if(j>0) :lap[index,A2] = 1./ds_x**2;
            if(i<nym2-1):lap[index,A5] = 1./ds_x**2;
            if((i>=0) and (i<=nym2) and (j>=0) and (j<=nxm2)): lap[index,index]=-2./ds_x**2.-2./ds_y**2.
    # Apply block-sparse-row function to lap
    lap = scipy.sparse.bsr_matrix(lap)
    # Compute transpose (adjoint) of laplacian operator
    laps = lap.T #........................................................................... transpose of lap
    # Define initial guess of psi
    psi0 = np.ones((nym2, nxm2), dtype=np.float64) #......................................... initial guess of psi (all ones)
    # Define laplacian and transpose of laplacian operators
    LAP = lap
    LAPS = laps    
    #
    # Define functions to solve
    #
    # L-BFGS method
    #
    # Define a function to compute error term
    def fun(psi0):
        error = LAP.dot(psi0) - forcevec
        return 0.5*(error**2).sum()
    # Define a function to compute gradient term
    def grad(psi0):
        return LAPS.dot(LAP.dot(psi0) - forcevec)
    # Optimize to solve
    field, value, info = scipy.optimize.fmin_l_bfgs_b(fun, psi0.flatten(), grad, pgtol=1e-14, factr=.0001,
                                                  maxiter=150000,maxfun=15000*50)
    # Stop clock
    end = time.time() #...................................................................... Ending time
    #
    #########################################################################################
    #
    # Report timing statistics for solving on level-k
    print('time for solve = ', end - start, ' seconds.' )
    print(info)
    # Define full_psi as reshaped matrix of converged psi
    full_psi[:,:] = psi.copy()
    full_psi[1:-1,1:-1] = reshape(field, (nym2, nxm2)).copy()    
    # Return full_psi
    return full_psi[:,:]
    #
    # END
    #
