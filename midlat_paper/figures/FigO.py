# Figure O
# a) initial and final-forecast perturbation energy norm (u', v', T') profiles of least-intense simulation
# b) as panel-a, but for most-intense simulation
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from analysis_dependencies import get_wrf_tk
#
# define internal functions
#
def generate_figure_panel(unpInitHdl, ptdInitHdl, unpFcstHdl, ptdFcstHdl, figureName):
    # constants
    cp = 1004.  # specific heat at constant pressure
    tr = 270.   # reference temperature
    L = 2.5104E+06  # latent heat of condensation
    eps = 1.0  # latent heat coefficient
    # extract eta-levels from unpInitHdl (should be identical for all files)
    eta = np.asarray(unpInitHdl.variables['ZNU']).squeeze()  # d(eta) on half levels
    # extract (u,v,T,q) from each WRF file handle
    unpInitU = np.asarray(unpInitHdl.variables['U']).squeeze()
    ptdInitU = np.asarray(ptdInitHdl.variables['U']).squeeze()
    unpInitV = np.asarray(unpInitHdl.variables['V']).squeeze()
    ptdInitV = np.asarray(ptdInitHdl.variables['V']).squeeze()
    unpInitT = np.asarray(get_wrf_tk(unpInitHdl)).squeeze()
    ptdInitT = np.asarray(get_wrf_tk(ptdInitHdl)).squeeze()
    unpInitQ = np.asarray(unpInitHdl.variables['QVAPOR']).squeeze()
    ptdInitQ = np.asarray(ptdInitHdl.variables['QVAPOR']).squeeze()
    unpFcstU = np.asarray(unpFcstHdl.variables['U']).squeeze()
    ptdFcstU = np.asarray(ptdFcstHdl.variables['U']).squeeze()
    unpFcstV = np.asarray(unpFcstHdl.variables['V']).squeeze()
    ptdFcstV = np.asarray(ptdFcstHdl.variables['V']).squeeze()
    unpFcstT = np.asarray(get_wrf_tk(unpFcstHdl)).squeeze()
    ptdFcstT = np.asarray(get_wrf_tk(ptdFcstHdl)).squeeze()
    unpFcstQ = np.asarray(unpFcstHdl.variables['QVAPOR']).squeeze()
    ptdFcstQ = np.asarray(ptdFcstHdl.variables['QVAPOR']).squeeze()
    # compute perturbation quantities
    pInitU = ptdInitU - unpInitU
    pInitV = ptdInitV - unpInitV
    pInitT = ptdInitT - unpInitT
    pInitQ = ptdInitQ - unpInitQ
    pFcstU = ptdFcstU - unpFcstU
    pFcstV = ptdFcstV - unpFcstV
    pFcstT = ptdFcstT - unpFcstT
    pFcstQ = ptdFcstQ - unpFcstQ
    # compute initial and final energy profiles
    pInitKE = np.nan * np.ones(np.shape(eta))
    pInitAPE = np.nan * np.ones(np.shape(eta))
    pInitQE = np.nan * np.ones(np.shape(eta))
    pFcstKE = np.nan * np.ones(np.shape(eta))
    pFcstAPE = np.nan * np.ones(np.shape(eta))
    pFcstQE = np.nan * np.ones(np.shape(eta))
    for k in range(np.size(eta)):
        # initial-time profiles
        up = pInitU[k,:,:].squeeze()
        vp = pInitV[k,:,:].squeeze()
        tp = pInitT[k,:,:].squeeze()
        qp = pInitQ[k,:,:].squeeze()
        pInitKE[k] = 0.
        pInitKE[k] = pInitKE[k] + 0.5 * np.sum(up**2.)
        pInitKE[k] = pInitKE[k] + 0.5 * np.sum(vp**2.)
        pInitAPE[k] = 0.
        pInitAPE[k] = pInitAPE[k] + 0.5 * (cp/tr) * np.sum(tp**2.)
        pInitQE[k] = 0.
        pInitQE[k] = pInitKE[k] + 0.5 * eps * L**2./(cp*tr) * np.sum(qp**2.)
        # final-time profiles
        up = pFcstU[k,:,:].squeeze()
        vp = pFcstV[k,:,:].squeeze()
        tp = pFcstT[k,:,:].squeeze()
        qp = pFcstQ[k,:,:].squeeze()
        pFcstKE[k] = 0.
        pFcstKE[k] = pFcstKE[k] + 0.5 * np.sum(up**2.)
        pFcstKE[k] = pFcstKE[k] + 0.5 * np.sum(vp**2.)
        pFcstAPE[k] = 0.
        pFcstAPE[k] = pFcstAPE[k] + 0.5 * (cp/tr) * np.sum(tp**2.)
        pFcstQE[k] = 0.
        pFcstQE[k] = pFcstKE[k] + 0.5 * eps * L**2./(cp*tr) * np.sum(qp**2.)
    # compute initial/forecast total energy (norm) profiles
    pInitTOT = pInitKE + pInitAPE
    pFcstTOT = pFcstKE + pFcstAPE
    # plot figure panel: initial/forecast energy norm profile, with and without QE term
    fig = plt.figure(figsize=(4,8))
    plt.plot(5. * pInitTOT, eta, color='black', linewidth=2.0)
    plt.plot(5. * (pInitTOT + pInitQE), eta, color='black', linewidth=2.0, linestyle='dotted')
    plt.plot(pFcstTOT, eta, color='orange', linewidth=2.0)
    plt.legend(['norm init (mul. 5)', 'norm init + QE (mul. 5)', 'norm final'])
    plt.gca().invert_yaxis()
    # save figure
    fig.savefig(figureName + '.png', bbox_inches='tight', facecolor='white')
    return
#
# begin
#
if __name__ == "__main__":
    # Panel A: inital/final energy norm profiles for least-intense simulaiton
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/negative/uvTq'
    unpInitHdl = Dataset(dataDir + '/wrfinput_d01_unpi00')
    ptdInitHdl = Dataset(dataDir + '/wrfinput_d01_ptdi19')
    unpFcstHdl = Dataset(dataDir + '/wrfout_d01_unpi00')
    ptdFcstHdl = Dataset(dataDir + '/wrfout_d01_ptdi19')
    generate_figure_panel(unpInitHdl, ptdInitHdl, unpFcstHdl, ptdFcstHdl, 'FIGO_panel_A')
    # Panel A: inital/final energy norm profiles for most-intense simulaiton
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/positive/uvTq'
    unpInitHdl = Dataset(dataDir + '/wrfinput_d01_unpi00')
    ptdInitHdl = Dataset(dataDir + '/wrfinput_d01_ptdi24')
    unpFcstHdl = Dataset(dataDir + '/wrfout_d01_unpi00')
    ptdFcstHdl = Dataset(dataDir + '/wrfout_d01_ptdi24')
    generate_figure_panel(unpInitHdl, ptdInitHdl, unpFcstHdl, ptdFcstHdl, 'FIGO_panel_B')
#
# end
#
