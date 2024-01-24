# Figure D:
# All panels: Plot the local projection operator for each iteration from the unperturbed simulation to
# the most intense simulation, from black to blue respectively, with the LPO of the unperturbed
# simulation in black.
# a) 24-hr forecast sea level pressure of unperturbed simulation
# b) 24-hr forecast sea level pressure of most-intense simulation
# NOTE: the sea-level pressure minimum can stray slightly from the center of the LPO in the perturbed
#       simulations, because we are comparing the LPO derived from the *unperturbed* sea-level pressure from
#       that iteration to the *perturbed* sea-level pressure from the resulting optimal perturbation
# load modules (UWAOS_WRF_QOIP compliant)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import wrf
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import plan_section_plot
#
# define internal functions
#
def generate_figure_panel(wrfHdl, sensIndex, figureName):
    # extract latitude and longitude, set longitude to 0 to 360 deg format
    lat = np.asarray(wrfHdl.variables['XLAT']).squeeze()
    lon = np.asarray(wrfHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # define data and plot projection
    datProj = gen_cartopy_proj(wrfHdl)
    plotProj = ccrs.PlateCarree()
    # extract sea-level pressure
    slp = np.asarray(get_wrf_slp(wrfHdl)).squeeze()
    # generate figure panel
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(14,7), subplot_kw={'projection' : datProj})
    # define contour levels of SLP and thickness
    slpContours = np.arange(900., 1040.1, 4.)
    # assemble list of final-time sensitivity gradients with respect to MU from unperturbed
    # to most-intense (largest positive iteration)
    #   NOTE: in operation the response function is multiplied by -1, so here we are mapping the
    #         most-intense to blue to match the simplified language in the paper where the response
    #         function is shown only as the average MU in the LPO. These are merely bookkeeping
    #         differences.
    # assemble list of positive (intensifying, mapped to blue) iterations
    sensList = glob('/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/positive/uvTq/final_sens_d01_unpi*')
    sensList.sort()
    # generate empty lists to store contour values, colors, line-thicknesses, and contour-intervals
    # initialize each list with data for plotting the sea level pressure for wrfHdl
    contVarList = [slp]
    contColList = ['black']
    contThkList = [1.0]
    contIntList = [slpContours]
    # loop through sensList and append each component in-order to the lists
    for i in range(len(sensList)):
        sensFile = sensList[i]
        hdl = Dataset(sensFile)
        contVarList.append(-1.*np.asarray(hdl.variables['G_MU']).squeeze())  # -1 included here to force contours into positive values and be plotted as solid contours
        contIntList.append([-1.,0.,1.])
        if i == sensIndex:
            if 'final_sens_d01_unpi00' in sensFile:
                contColList.append('black')
            else:
                contColList.append(matplotlib.colors.to_hex((1.-i/len(sensList), 0., i/len(sensList))))
            contThkList.append(3.0)
        else:
            contColList.append('gray')
            contThkList.append(1.0)
    # generate plan-section plot for figure panel axis
    ax, (shd, cons, vec) = plan_section_plot(wrfHDL=wrfHdl,
                                        lat=lat,
                                        lon=lon,
                                        contVariableList=contVarList,
                                        contIntervalList=contIntList, 
                                        contColorList=contColList,
                                        contLineThicknessList=contThkList,
                                        shadVariable=None,
                                        shadInterval=None,
                                        shadAlpha=None,
                                        datProj=datProj,
                                        plotProj=plotProj,
                                        shadCmap=None,
                                        uVecVariable=None,
                                        vVecVariable=None,
                                        vectorThinning=None,
                                        vecColor=None,
                                        figax=ax)
    # add contour labels to slp
    ax.clabel(cons[0], levels=slpContours[::2])
    # save file
    fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return
#
# begin
#
if __name__ == "__main__":
    # define directory of unperturbed, most-intense, and least-intense WRF forecast
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/unperturbed/'
    posDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/march2020/R_mu/positive/uvTq/ptdi22/'
    # define datetime of WRF initialization
    dtInit = datetime.datetime(2020, 3, 6, 12)

    # FIG Da: 24-hr unperturbed WRF SLP and all local projection operators
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = unpDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 0, 'FIGD_panel_A')
    # FIG Db: 24-hr most-intense WRF SLP and all local projection operators
    fcstHr = 24
    # define forecast datetime stamp
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # define WRF forecast file and open netCDF4 file-handle
    fileFcst = posDir + 'wrfout_d01_' + dtFcstStr
    wrfHdl = Dataset(fileFcst)
    generate_figure_panel(wrfHdl, 22, 'FIGD_panel_B')
#
# end
#
