# Ventilation index is defined in the following publications:
#
# Tang, B., and K. Emanuel, 2012: A ventilation index for tropical cyclones. Bulletin of the American Meteorological
#     Society, 93, 1901-1912, doi: 10.1175/BAMS-D-11-00165.1
# Tang, B. and K. Emanuel, 2012: Supplement to “A ventilation index for tropical cyclones”
#
# The index is composed of several terms multiplied or divided:
#
# vi = A * B / C
#    A = environmental shear
#    B = nondimensional entropy deficit
#    C = potential intensity
#
# A: Environmental Shear
#    This term is computed from the difference vector of the winds at 850 and 200 hPa, but it is not computed from the full wind field.
#    Instead, the "environmental" component is computed from removal of the "differential" vorticity and divergence within r=900km of
#    the TC center, as per pg. 2472 of:
#
#    Davis, C., C. Snyder, and A. C. Didlake Jr., 2008: A vortex-based perspective of eastern Pacific tropical cyclone
#        formation. Monthly Weather Review, 136, 2461-2477, doi: 10.1175/2007MWR2317.1
#
#    This is accomplished by inverting the differential vorticity and divergence to compute a differential streamfunction and
#    velocity potential, which are then used to compute the nondivergent and irrotational components of the differential wind field
#    presumed to be associated with the TC. This vector is removed from the differential wind vector before computing the shear.
#
# B: Nondimensional Entropy Deficit
#    This term is composed of three sub-terms:
#
#    B = (B1 - B2) / B3
#
#    B1 is the pseudoadiabatic saturated entropy associated with the TC. In the index, this is computed following Equation 11 of:
#
#    Bryan, G. H., 2008: On the Computation of Pseudoadiabatic Entropy and Equivalent Potential Temperature. Mon. Wea.
#        Rev., 136, 5239–5245, https://doi.org/10.1175/2008MWR2593.1
#
#    Notably, Equation 12 establishes an equivalency between the pseudoadiabatic entropy and cp*ln(thta_e). tcpyPI has a function
#    that *should* be usable for computing this term, but I am running into problems with it and will instead rely on the Eqn 12
#    equivalency instead.
#
#    B1 is calculated as a mean value in a disc of r=100km centered on the TC at 600 hPa, representing midlevel entropy of the TC
#
#    B2 is the pseudoadiabatic entropy associated with the environment. It is the same as B1 in all respects other than that it is
#    computed using the real moisture rather than assuming saturation, and in an annulus between r1=100km and r2=300km at 600 hPa,
#    representing the midlevel entropy of the environment
#
#    B3 is described as the "thermodynamic disequilibrium", and is not explicitly defined in the Tang and Emanuel papers, but
#    instead the reader is referred to:
#
#    Bister, M., and K. A. Emanuel, 2002: Low frequency variability of tropical cyclone potential intensity. I:
#        Interannual to interdecadal variability. Journal of Geophysical Research, 107, doi: 10.1029/2001JD000776
#
#    where I likewise to not see a derivation. However, tcpyPI has a function to compute this disequilibrium term as a residual
#    of other easily computable terms.
#
# C: Potential intensity
#    This is the same potential intensity computed by tcpyPI, in m/s format, under pseudoadiabatic assumptions.
#
#
# I will look at these three terms, or pseudo-equivalencies of these terms, separately:
#
# Environmental Shear: We can compute the streamfunction and velocity potential of the differential vorticity in the same manner as
# is done normally, first computing on the 27-km grid and then establishing boundary conditions for the computation on the 9km grid.
# Further, let's assume that perturbations to the vorticity and divergence, even within the r=900km radius, are perturbations to the
# environment rather than to the TC, and can affect the environmental shear.
#
# Nondimensional Entropy Deficit: Rather than compute this directly, we will look at the numerator and denominator indirectly. The
# numerator will be approached with the pseudo-equivalent entropy values of cp*ln(thta_e_sat) and cp*ln(thta_e) for the tc and
# environmental entropy terms respectively. We will use the full-field at 600 hPa, but we can put the 100km and 300km radii on the plot
# for reference. The denominator is computed from the residual as per tcpyPI calculation.
#
# Potential Intensity: This is computed from tcpyPI.
#
# We can likewise pick out the TC efficiency term of the thermodynamic disequilibrium fairly easily:
#
# e = (SST - TO) / TO
#
# where SST is the sea-surface temperature and TO is the outflow temperature computed from tcpyPI. We will investigate this as well.
import numpy as np
from netCDF4 import Dataset
import wrf
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import compute_inverse_laplacian_with_boundaries
from analysis_dependencies import compute_hires_border
from analysis_dependencies import haversine
from analysis_dependencies import get_wrf_slp
import datetime
import argparse
#
# begin
#
if __name__ == "__main__":
    # define parser for input argument(s)
    parser = argparse.ArgumentParser(description='Define inputs')
    parser.add_argument('baseDataDir', metavar='BASEDATADIR', type=str, help='full path to directory containing ' +
                        'high- and low-res file directories (ends in /)')
    parser.add_argument('Lores', metavar='LORES', type=str, help='name of low-res directory, probably 9km')
    parser.add_argument('Hires', metavar='HIRES', type=str, help='name of high-res directory, probably 27km')
    parser.add_argument('YYYY', metavar='YYYY', type=int, help='initialization year (YYYY)')
    parser.add_argument('MM', metavar='MM', type=int, help='initialization month (M, or MM)')
    parser.add_argument('DD', metavar='DD', type=int, help='initialization day (D, or DD)')
    parser.add_argument('HH', metavar='HH', type=int, help='initialization hour (H, or HH)')
    parser.add_argument('fcstHr', metavar='FCSTHR', type=int, help='forecast hour (F, or FF)')
    parser.add_argument('start_i', metavar='STARTI', type=int, help='lower-left corner starting i of high-res grid')
    parser.add_argument('start_j', metavar='STARTJ', type=int, help='lower-left corner starting j of high-res grid')
    parser.add_argument('resRatio', metavar='RESRAT', type=float, help='ratio of high- to low-res grid resolutions, probably 3.')
    commandInputs = parser.parse_args()
    # define inputs form commandInputs
    baseDataDir = commandInputs.baseDataDir
    Lores = commandInputs.Lores
    Hires = commandInputs.Hires
    YYYY = int(commandInputs.YYYY)
    MM = int(commandInputs.MM)
    DD = int(commandInputs.DD)
    HH = int(commandInputs.HH)
    fcstHr = float(commandInputs.fcstHr)
    start_i = int(commandInputs.start_i)
    start_j = int(commandInputs.start_j)
    resRatio = float(commandInputs.resRatio)
    # quality-control directory name inputs, if they do not end in '/', add it
    if baseDataDir[-1] != '/':
        baseDataDir = baseDataDir + '/'
    # define dtFcstStr based on initializaiton and fcstHr
    dtInit = datetime.datetime(YYYY, MM, DD, HH)
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst, '%Y-%m-%d_%H:00:00')
    # define wrf forecast files based on baseDataDir, Lo/Hires, and dtFcstStr
    wrfFileLores = baseDataDir + Lores + '_res/wrfout_d01_' + dtFcstStr
    wrfFileHires = baseDataDir + Hires + '_res/wrfout_d01_' + dtFcstStr
    # generate wrf file-handles for forecast files
    wrfHdlLores = Dataset(wrfFileLores)
    wrfHdlHires = Dataset(wrfFileHires)
    # compute vorticity and divergence on lo-/hires grids (native vertical coordinate)
    vor3DLores = get_wrf_kinematic(wrfHdlLores, 'vor')
    vor3DHires = get_wrf_kinematic(wrfHdlHires, 'vor')
    div3DLores = get_wrf_kinematic(wrfHdlLores, 'div')
    div3DHires = get_wrf_kinematic(wrfHdlHires, 'div')
    # extract latitude and longitude on lo-res and hi-res grids
    latLores = np.asarray(wrfHdlLores.variables['XLAT']).squeeze()
    lonLores = np.asarray(wrfHdlLores.variables['XLONG']).squeeze()
    latHires = np.asarray(wrfHdlHires.variables['XLAT']).squeeze()
    lonHires = np.asarray(wrfHdlHires.variables['XLONG']).squeeze()
    # extract SLP on hi-res grid for finding lat/lon of TC center
    slpHires = np.asarray(get_wrf_slp(wrfHdlHires)).squeeze()
    # compute ji location of SLP minimum to define TC center on hi-res grid
    cenjiHires = np.argmin(slpHires.flatten())
    # compute distance of each grid-point from TC center on hi-res grid
    d = haversine(latHires.flatten()[cenjiHires],lonHires.flatten()[cenjiHires],latHires.flatten(),lonHires.flatten())
    distHires = d.reshape(np.shape(latHires))
    # compute distance of each grid-point from TC center on lo-res grid (TC center defined on hi-res grid)
    d = haversine(latHires.flatten()[cenjiHires],lonHires.flatten()[cenjiHires],latLores.flatten(),lonLores.flatten())
    distLores = d.reshape(np.shape(latLores))
    # generate zero-arrays to store sf and vp on lo-hires grids
    sf3DLores = np.zeros(np.shape(vor3DLores))
    sf3DHires = np.zeros(np.shape(vor3DHires))
    vp3DLores = np.zeros(np.shape(div3DLores))
    vp3DHires = np.zeros(np.shape(div3DHires))
    # generate zero-arrays to store boundary information (and potentially first-guess information on interior)
    sf3DHires_bounds = np.zeros(np.shape(vor3DHires))
    vp3DHires_bounds = np.zeros(np.shape(div3DHires))
    #
    # compute streamfunction on low-res grid, with zero boundaries
    #
    # define forcing grid as low-res vorticity grid
    forc3DLores = vor3DLores.copy()
    # define number of vertical levels
    nz = np.shape(forc3DLores)[0]
    # loop through all vertical levels
    for k in range(nz):
        print(Lores + ': computing inverse-laplacian on level {:d} of {:d}'.format(k, nz))
        # compute inverse-laplacian on lo-res grid with zero-boundaries
        forc2D = np.asarray(forc3DLores[k,:,:]).squeeze()
        # zero out vorticity and divergence outside 900 km of TC center to define vor/div forcing
        forc2D[distLores>900.] = 0.
        lapl2D = compute_inverse_laplacian_with_boundaries(wrfHdlLores, forc2D, boundaries=None)
        sf3DLores[k,:,:] = lapl2D
    #
    # compute velocity potential on low-res grid, with zero boundaries
    #
    # define forcing grid as low-res divergence field
    forc3DLores = div3DLores
    # define number of vertical levels
    nz = np.shape(forc3DLores)[0]
    # loop through all vertical levels
    for k in range(nz):
        print(Lores + ': computing inverse-laplacian on level {:d} of {:d}'.format(k, nz))
        # compute inverse-laplacian on lo-res grid with zero-boundaries
        forc2D = np.asarray(forc3DLores[k,:,:]).squeeze()
        # zero out vorticity and divergence outside 900 km of TC center to define vor/div forcing
        forc2D[distLores>900.] = 0.
        lapl2D = compute_inverse_laplacian_with_boundaries(wrfHdlLores,forc2D,boundaries=None)
        vp3DLores[k,:,:] = lapl2D
    #
    # compute streamfunction on high-res grid, with boundaries informed from low-res grid and using
    # the prior level as a first-guess (to speed up convergence)
    #
    # define forcing grid as high-res vorticity field
    forc3DHires = vor3DHires
    # define number of vertical levels
    nz = np.shape(forc3DHires)[0]
    # loop through all vertical levels
    for k in range(nz):
        print(Hires + ': computing inverse-laplacian on level {:d} of {:d}'.format(k, nz))
        # compute boundaries of hi-res grid
        forc2D = np.asarray(forc3DHires[k,:,:]).squeeze()
        # zero out vorticity and divergence outside 900 km of TC center to define vor/div forcing
        forc2D[distHires>900.] = 0.
        lores2D = sf3DLores[k,:,:].squeeze()
        bound2D = compute_hires_border(lores2D, forc2D, start_i, start_j, resRatio)
        # if there is a prior level already solved, use it as a first-guess to solve the current level
        # NOTE: this step presumes that the streamfunction does not drastically change from one vertical
        #       level to the next, thus the first-guess from the prior level will speed up convergence
        if k > 0:
            bound2D = bound2D + sf3DHires[k-1,:,:].squeeze()
        # compute inverse-laplacian on hi-res grid with boundaries
        lapl2D = compute_inverse_laplacian_with_boundaries(wrfHdlHires,forc2D,boundaries=bound2D)
        sf3DHires[k,:,:] = lapl2D
        sf3DHires_bounds[k,:,:] = bound2D
    #
    # compute velocity potential on high-res grid, with boundaries informed from low-res grid and using
    # the prior level as a first-guess (to speed up convergence)
    #
    # define forcing grid as high-res divergence field
    forc3DHires = div3DHires
    # define number of vertical levels
    nz = np.shape(forc3DHires)[0]
    # loop through all vertical levels
    for k in range(nz):
        print(Hires + ': computing inverse-laplacian on level {:d} of {:d}'.format(k, nz))
        # compute boundaries of hi-res grid
        forc2D = np.asarray(forc3DHires[k,:,:]).squeeze()
        # zero out vorticity and divergence outside 900 km of TC center to define vor/div forcing
        forc2D[distHires>900.] = 0.
        lores2D = vp3DLores[k,:,:].squeeze()
        bound2D = compute_hires_border(lores2D, forc2D, start_i, start_j, resRatio)
        # if there is a prior level already solved, use it as a first-guess to solve the current level
        # NOTE: this step presumes that the velocity potential does not drastically change from one vertical
        #       level to the next, thus the first-guess from the prior level will speed up convergence
        if k > 0:
            bound2D = bound2D + vp3DHires[k-1,:,:].squeeze()
        # compute inverse-laplacian on hi-res grid with boundaries
        lapl2D = compute_inverse_laplacian_with_boundaries(wrfHdlHires,forc2D,boundaries=bound2D)
        vp3DHires[k,:,:] = lapl2D
        vp3DHires_bounds[k,:,:] = bound2D
    # write low-res grid SF and VP to netCDF file
    nc_out = Dataset( #...................................................... Dataset object for output
                      'InverseLaplacian_' + Lores + '_' + dtFcstStr + '.nc4'  , # Dataset input: Output file name
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
    SF = nc_out.createVariable( #.................................... Output variable
                                "SF"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    VP = nc_out.createVariable( #.................................... Output variable
                               "VP"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    # Fill netCDF arrays via slicing
    SF[:,:,:] = sf3DLores
    VP[:,:,:] = vp3DLores
    # Close netCDF file
    nc_out.close()
    #
    # write high-res SF, VP, and boundaries to netCDF file
    # NOTE: make sure to zero out interior of boundaries before sending to file, since the interior of these fields
    #       may contain first-guess information rather than zeros
    nc_out = Dataset( #...................................................... Dataset object for output
                      'InverseLaplacian_' + Hires + '_' + dtFcstStr + '.nc4'  , # Dataset input: Output file name
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
    SF = nc_out.createVariable( #.................................... Output variable
                               "SF"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    VP = nc_out.createVariable( #.................................... Output variable
                               "VP"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "lev"       , # nc_out.createVariable input: Variable dimension
                                 "lat"       , # nc_out.createVariable input: Variable dimension
                                 "lon"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    SF_bounds = nc_out.createVariable( #.................................... Output variable
                                      "SF_bounds"  , # nc_out.createVariable input: Variable name
                                      "f8"          , # nc_out.createVariable input: Variable format
                                      (
                                       "lev"       , # nc_out.createVariable input: Variable dimension
                                       "lat"       , # nc_out.createVariable input: Variable dimension
                                       "lon"   # nc_out.createVariable input: Variable dimension
                                     )
                                    )
    VP_bounds = nc_out.createVariable( #.................................... Output variable
                                  "VP_bounds"  , # nc_out.createVariable input: Variable name
                                  "f8"          , # nc_out.createVariable input: Variable format
                                  (
                                   "lev"       , # nc_out.createVariable input: Variable dimension
                                   "lat"       , # nc_out.createVariable input: Variable dimension
                                   "lon"   # nc_out.createVariable input: Variable dimension
                                  )
                                )
    # Fill netCDF arrays via slicing
    SF[:,:,:] = sf3DHires
    VP[:,:,:] = vp3DHires
    # *_bounds may contain first-guess information in interior, zero out interior before writing to file
    sf3DHires_bounds[:,1:-1,1:-1] = 0.
    vp3DHires_bounds[:,1:-1,1:-1] = 0.
    VP_bounds[:,:,:] = vp3DHires_bounds
    SF_bounds[:,:,:] = sf3DHires_bounds
    # Close netCDF file
    nc_out.close()
#
# end
#
