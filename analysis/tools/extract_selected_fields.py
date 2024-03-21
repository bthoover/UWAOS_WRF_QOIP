# import modules (bth-wrf_qoip/UWAOS_WRF_QOIP compliant)
import numpy as np
from netCDF4 import Dataset
from analysis_dependencies import get_wrf_slp
import datetime
import wrf
import argparse
#
# begin
#
if __name__ == "__main__":
    # define parser for input argument(s)
    parser = argparse.ArgumentParser(description='Define inputs')
    parser.add_argument('baseDataDir', metavar='BASEDATADIR', type=str, help='full path to directory containing ' +
                        'high- and low-res file directories (ends in /)')
    parser.add_argument('Hires', metavar='HIRES', type=str, help='name of high-res directory, probably 9km or 3km')
    parser.add_argument('YYYY', metavar='YYYY', type=int, help='initialization year (YYYY)')
    parser.add_argument('MM', metavar='MM', type=int, help='initialization month (M, or MM)')
    parser.add_argument('DD', metavar='DD', type=int, help='initialization day (D, or DD)')
    parser.add_argument('HH', metavar='HH', type=int, help='initialization hour (H, or HH)')
    parser.add_argument('fcstHr', metavar='FCSTHR', type=int, help='forecast hour (F, or FF)')
    commandInputs = parser.parse_args()
    # parse argument inputs into preferred formats
    Hires = commandInputs.Hires
    YYYY = int(commandInputs.YYYY)
    MM = int(commandInputs.MM)
    DD = int(commandInputs.DD)
    HH = int(commandInputs.HH)
    fcstHr = float(commandInputs.fcstHr)
    # quality control baseDataDir
    if commandInputs.baseDataDir[-1] != '/':
        baseDataDir = commandInputs.baseDataDir + '/'
    else:
        baseDataDir = commandInputs.baseDataDir
    # define dtFcstStr based on initializaiton and fcstHr
    dtInit = datetime.datetime(YYYY, MM, DD, HH)
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst, '%Y-%m-%d_%H:00:00')
    # define hires wrf forecast file based on baseDataDir, Hires, and dtFcstStr
    wrfFileHires = baseDataDir + commandInputs.Hires + '_res/wrfout_d01_' + dtFcstStr
    # generate wrf file-handle
    wrfHdlHires = Dataset(wrfFileHires)
    # extract necessary fields from wrf file
    lat = np.asarray(wrfHdlHires.variables['XLAT']).squeeze()
    lon = np.asarray(wrfHdlHires.variables['XLONG']).squeeze()
    slp = np.asarray(get_wrf_slp(wrfHdlHires)).squeeze()
    P = np.asarray(wrfHdlHires.variables['PSFC']).squeeze()
    T = np.asarray(wrfHdlHires.variables['T2']).squeeze()
    Q = np.asarray(wrfHdlHires.variables['Q2']).squeeze()
    rainc = np.asarray(wrfHdlHires.variables['RAINC']).squeeze()   # convective
    rainnc = np.asarray(wrfHdlHires.variables['RAINNC']).squeeze() # non-convective
    rainsh = np.asarray(wrfHdlHires.variables['RAINSH']).squeeze() # shallow convective
    # compute surface equivalent potential temperature
    eth = np.asarray(wrf.eth(Q[None,:,:], T[None,:,:], P[None,:,:], meta=False, units='K')).squeeze()
    # extract necessary attributes for generating grid projection
    MAP_PROJ = wrfHdlHires.MAP_PROJ
    TRUELAT1 = wrfHdlHires.TRUELAT1
    TRUELAT2 = wrfHdlHires.TRUELAT2
    MOAD_CEN_LAT = wrfHdlHires.MOAD_CEN_LAT
    STAND_LON = wrfHdlHires.STAND_LON
    POLE_LAT = wrfHdlHires.POLE_LAT
    POLE_LON = wrfHdlHires.POLE_LON
    DX = wrfHdlHires.DX
    DY = wrfHdlHires.DY
    # write variables to output file
    nc_out = Dataset( #...................................................... Dataset object for output
                      'wrfout_d01_selectFields_' + Hires + '_' + dtFcstStr + '.nc4'  , # Dataset input: Output file name
                      "w"              , # Dataset input: Make file write-able
                      format="NETCDF4" , # Dataset input: Set output format to netCDF4
                    )
    # Dimensions
    nc_ny  = nc_out.createDimension( #......................................... Output dimension
                                      "ny" , # nc_out.createDimension input: Dimension name
                                      None   # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                    )
    nc_nx  = nc_out.createDimension( #......................................... Output dimension
                                      "nx" , # nc_out.createDimension input: Dimension name
                                      None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                    )
    # Variables
    nc_lat = nc_out.createVariable( #.................................... Output variable
                               "XLAT"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_lon = nc_out.createVariable( #.................................... Output variable
                               "XLONG"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_slp = nc_out.createVariable( #.................................... Output variable
                               "SLP"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_eth = nc_out.createVariable( #.................................... Output variable
                               "ETH"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_rc = nc_out.createVariable( #.................................... Output variable
                               "RAINC"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_rnc = nc_out.createVariable( #.................................... Output variable
                               "RAINNC"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    nc_rsh = nc_out.createVariable( #.................................... Output variable
                               "RAINSH"  , # nc_out.createVariable input: Variable name
                               "f8"          , # nc_out.createVariable input: Variable format
                               (
                                 "ny"       , # nc_out.createVariable input: Variable dimension
                                 "nx"   # nc_out.createVariable input: Variable dimension
                               )
                             )
    # Fill netCDF arrays via slicing
    nc_lat[:,:] = lat
    nc_lon[:,:] = lon
    nc_slp[:,:] = slp
    nc_eth[:,:] = eth
    nc_rc[:,:] = rainc
    nc_rnc[:,:] = rainnc
    nc_rsh[:,:] = rainsh
    # Assign attributes
    nc_out.MAP_PROJ = MAP_PROJ
    nc_out.TRUELAT1 = TRUELAT1
    nc_out.TRUELAT2 = TRUELAT2
    nc_out.MOAD_CEN_LAT = MOAD_CEN_LAT
    nc_out.STAND_LON = STAND_LON
    nc_out.POLE_LAT = POLE_LAT
    nc_out.POLE_LON = POLE_LON
    nc_out.DX = DX
    nc_out.DY = DY
    # Close netCDF file
    nc_out.close()
#
# end
#
