# load modules
import numpy as np
from netCDF4 import Dataset
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import compute_inverse_laplacian_with_boundaries
from analysis_dependencies import compute_hires_border
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
    forc3DLores = vor3DLores
    # define number of vertical levels
    nz = np.shape(forc3DLores)[0]
    # loop through all vertical levels
    for k in range(nz):
        print(Lores + ': computing inverse-laplacian on level {:d} of {:d}'.format(k, nz))
        # compute inverse-laplacian on lo-res grid with zero-boundaries
        forc2D = forc3DLores[k,:,:].squeeze()
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
        forc2D = forc3DLores[k,:,:].squeeze()
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
        forc2D = forc3DHires[k,:,:].squeeze()
        lores2D = sf3DLores[k,:,:].squeeze()
        bound2D = compute_hires_border(lores2D, forc2D, i_start, j_start, resRatio)
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
        forc2D = forc3DHires[k,:,:].squeeze()
        lores2D = vp3DLores[k,:,:].squeeze()
        bound2D = compute_hires_border(lores2D, forc2D, i_start, j_start, resRatio)
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
