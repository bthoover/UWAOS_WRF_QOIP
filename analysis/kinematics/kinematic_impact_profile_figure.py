# import dependencies
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import matplotlib.cm as cm
# define local functions
#
# plot_normalized_iterprofiles: generates a plot of normalized profiles (sums to 1.0 for each profile)
# for each iteration on a given colormap / plot axis
#
# INPUTS
#
# profVar (float): profile variable in (nLev,nIter) dimension
# profLev (float): profile levels in (nLev,) dimension
# titleStr (string): title for plot
# colMap (string): name of colormap
# ax (plt.axes): matplotlib.pyplot axis
#
# OUTPUTS
#
# no explicit outputs, but places figure on axis=ax
#
# DEPENDENCIES
#
# matplotlib.pyplot
# matplotlib.cm
# numpy
def plot_normalized_iterprofiles(profVar, profLev, titleStr, colMap, ax):
    # define number of levels (nl) and number of iterations (ni) based on
    # implicit dimensions of profVar [nl,ni]
    nl,ni = np.shape(profVar)
    # define colormap for each profile based on a colormap of colMap with
    # a length of [0,ni-1]
    scalarMap = cm.ScalarMappable(cmap=colMap)
    scalarMapList = scalarMap.to_rgba(range(ni))
    # loop through iterations
    for i in range(ni):
        # define profile color as ith element of scalarMapList
        profColor = list(scalarMapList[i][0:3])
        # define profile as ith iteration's profile [nl,]
        prof = profVar[:,i].squeeze()
        # plot normalized profile
        ax.plot(prof/np.abs(np.nansum(prof)),profLev,color=profColor,linewidth=2)
    # plot a dashed zero-line profile for reference
    ax.plot(np.zeros((nl,)),profLev,color='black',linestyle='dashed',linewidth=2)
    # add title
    ax.set_title(titleStr)
    # plotting on eta-levels: reverse y-axis
    ax.invert_yaxis()
    return
#
# begin
#
if __name__ == "__main__":
    # Note: will need to replace this hard-wired directory with an input-option at some point
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/positive/uvTq/'
    # list of kinematic sensitivity files for each iteration
    sensFileList = glob.glob(dataDir + 'gradient_kinematic_d01_unpi*')
    sensFileList.sort()
    # list of initial condition kinematic perturbation files for each iteration
    pertFileList = glob.glob(dataDir + 'wrfinput_kinematic_d01_perti*')
    pertFileList.sort()
    # number of files in lists (should all be equal, let's check)
    if len(sensFileList) != len(pertFileList):
        print('LIST MISMATCH: sensFileList size={:d} pertFileList={:d}'.format(len(sensFileList), len(pertFileList)))
        nFiles = None
    else:
        print('{:d} files discovered'.format(len(sensFileList)))
        nFiles = len(sensFileList)
    # use wrfinput_d01_unpi00 to pick eta values on each level
    # eta value at each level is computed as etaMidLev, computed as the average between
    # the eta values at each half-level between [1.,0.]
    dn = np.asarray(Dataset(dataDir+'wrfinput_d01_unpi00').variables['DN']).squeeze()  # d(eta) on half levels
    eta = 1.0+np.cumsum(dn)
    # add 0-top level
    eta = np.append(eta, 0.)
    # compute eta mid-levels
    etaMidLev = 0.5*(eta[0:-1] + eta[1:])
    # loop through files, generating impacts (sens*pert) at each grid-point for each
    # kinematic variable: VOR, DIV, STR, SHR
    # lat and lon are picked on the first sens-file, should be identical for all files
    # impact arrays are defined on first sens-file (dimensions copied from sens data)
    nIter = nFiles
    for i in range(nIter):
        print('processing iteration {:d} of {:d}'.format(i + 1, nFiles))
        sens_hdl = Dataset(sensFileList[i])
        pert_hdl = Dataset(pertFileList[i])
        if i == 0:
            lat = np.asarray(sens_hdl.variables['XLAT_M']).squeeze()
            lon = np.asarray(sens_hdl.variables['XLONG_M']).squeeze()
            nz, ny, nx = np.shape(np.asarray(sens_hdl.variables['A_VOR']).squeeze())
            impVOR = np.nan * np.ones((nz, ny, nx, nFiles))
            impDIV = np.nan * np.ones((nz, ny, nx, nFiles))
            impSTR = np.nan * np.ones((nz, ny, nx, nFiles))
            impSHR = np.nan * np.ones((nz, ny, nx, nFiles))
        # vorticity impact for iteration i
        impVOR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_VOR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_VOR']).squeeze())
        # divergence impact for iteration i
        impDIV[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_DIV']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_DIV']).squeeze())
        # stretching deformation impact for iteration i
        impSTR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_STR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_STR']).squeeze())
        # shearing deformation impact for iteration i
        impSHR[:, :, :, i] = np.multiply(np.asarray(sens_hdl.variables['A_SHR']).squeeze(),
                                         np.asarray(pert_hdl.variables['P_SHR']).squeeze())
    # fix longitudes
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # generate vertical-profiles of impact: (nz,nIter) dimensions
    impVORVertProf = np.nan * np.ones((nz, nIter))
    impDIVVertProf = np.nan * np.ones((nz, nIter))
    impSTRVertProf = np.nan * np.ones((nz, nIter))
    impSHRVertProf = np.nan * np.ones((nz, nIter))
    for i in range(nIter):
        for z in range(nz):
            impVORVertProf[z, i] = np.nansum(impVOR[z, :, :, i].squeeze().flatten())
            impDIVVertProf[z, i] = np.nansum(impDIV[z, :, :, i].squeeze().flatten())
            impSTRVertProf[z, i] = np.nansum(impSTR[z, :, :, i].squeeze().flatten())
            impSHRVertProf[z, i] = np.nansum(impSHR[z, :, :, i].squeeze().flatten())
     # 4-panel plot of each iteration's normalized profile for each kinematic term
    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    plot_normalized_iterprofiles(impVORVertProf, etaMidLev, 'Vorticity', 'gist_rainbow_r', axs.flatten()[0])
    plot_normalized_iterprofiles(impDIVVertProf, etaMidLev, 'Divergence', 'gist_rainbow_r', axs.flatten()[1])
    plot_normalized_iterprofiles(impSTRVertProf, etaMidLev, 'Stretching Def.', 'gist_rainbow_r', axs.flatten()[2])
    plot_normalized_iterprofiles(impSHRVertProf, etaMidLev, 'Shearing Def.', 'gist_rainbow_r', axs.flatten()[3])
    # save figure
    fig.savefig('iterprof_figure.png', bbox_inches='tight', facecolor='white')
    print('figure completed')
    #
    # end
    #
