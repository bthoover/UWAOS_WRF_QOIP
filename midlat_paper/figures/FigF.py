# Figure F
# a) normalized initial condition perturbation kinetic energy (u', v') profiles on each iteration of least-intense simulation
# b) normalized initial condition perturbation available potential (t') energy profiles on each iteration of least-intense simulation
# c) normalized initial condition perturbation latent (q') energy profiles on each iteration of least-intense simulation
# d) total-column initial condition perturbation energy of each type (KE, APE, QE) on each iteration of least-intense simulation
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import matplotlib.cm as cm
from analysis_dependencies import get_wrf_tk
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
def plot_normalized_iterprofiles(profVar, profLev, titleStr, colMap, figureName):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,8))
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
    ax.set_xlabel('normalized energy (J/kg)')
    ax.set_ylabel('sigma level')
    # plotting on eta-levels: reverse y-axis
    ax.invert_yaxis()
    # save figure
    fig.savefig(figureName+'.png', bbox_inches='tight', facecolor='white')
    return
#
# begin
#
if __name__ == "__main__":
    # define constants
    cp = 1004.  # specific heat at constant pressure
    tr = 270.   # reference temperature
    L = 2.5104E+06  # latent heat of condensation
    eps = 1.0  # latent heat coefficient
    # define data directory
    dataDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/march2020/R_mu/negative/uvTq'
    # check to make sure dataDir ends in '/', if not, add it to end of string
    if dataDir[-1] != '/':
        dataDir = dataDir + '/'
    # list of initial condition unperturbed files for each iteration
    unpFileList = glob.glob(dataDir + 'wrfinput_d01_unpi*')
    unpFileList.sort()
    # list of initial condition perturbed files for each iteration
    ptdFileList = glob.glob(dataDir + 'wrfinput_d01_ptdi*')
    ptdFileList.sort()
    # number of files in lists (should all be equal, let's check)
    if len(unpFileList) != len(ptdFileList):
        print('LIST MISMATCH: unpFileList size={:d} ptdFileList={:d}'.format(len(unpFileList), len(ptdFileList)))
        nFiles = None
    else:
        print('{:d} files discovered'.format(len(unpFileList)))
        nFiles = len(unpFileList)
    # use wrfinput_d01_unpi00 to pick eta values on each level
    # eta value at each level is computed as etaMidLev, computed as the average between
    # the eta values at each half-level between [1.,0.]
    dn = np.asarray(Dataset(dataDir+'wrfinput_d01_unpi00').variables['DN']).squeeze()  # d(eta) on half levels
    eta = 1.0+np.cumsum(dn)
    # add 0-top level
    eta = np.append(eta, 0.)
    # compute eta mid-levels
    etaMidLev = 0.5*(eta[0:-1] + eta[1:])
    # loop through files, generating ke and ape on each level
    nIter = nFiles
    for i in range(nIter):
        print('processing iteration {:d} of {:d}'.format(i + 1, nFiles))
        unpHdl = Dataset(unpFileList[i])
        ptdHdl = Dataset(ptdFileList[i])
        if i == 0:
            lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
            lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
            nz = np.shape(np.asarray(unpHdl.variables['P']).squeeze())[0]
            ke = np.nan * np.ones((nz, nFiles))
            ape = np.nan * np.ones((nz, nFiles))
            qe = np.nan * np.ones((nz, nFiles))
        # vorticity impact for iteration i
        u0 = np.asarray(unpHdl.variables['U']).squeeze()
        u1 = np.asarray(ptdHdl.variables['U']).squeeze()
        v0 = np.asarray(unpHdl.variables['V']).squeeze()
        v1 = np.asarray(ptdHdl.variables['V']).squeeze()
        t0 = np.asarray(get_wrf_tk(unpHdl)).squeeze()
        t1 = np.asarray(get_wrf_tk(ptdHdl)).squeeze()
        q0 = np.asarray(unpHdl.variables['QVAPOR']).squeeze()
        q1 = np.asarray(ptdHdl.variables['QVAPOR']).squeeze()
        for k in range(nz):
            up = u1[k,:,:].squeeze() - u0[k,:,:].squeeze()
            vp = v1[k,:,:].squeeze() - v0[k,:,:].squeeze()
            tp = t1[k,:,:].squeeze() - t0[k,:,:].squeeze()
            qp = q1[k,:,:].squeeze() - q0[k,:,:].squeeze()
            ke[k,i] = 0.
            ke[k,i] = ke[k,i] + 0.5*np.sum(up**2.)
            ke[k,i] = ke[k,i] + 0.5*np.sum(vp**2.)
            ape[k,i] = 0.
            ape[k,i] = 0.5*(cp/tr)*np.sum(tp**2.)
            qe[k,i] = 0.
            qe[k,i] = 0.5*eps*L**2./(cp*tr)*np.sum(qp**2.)
     # 4-panel plot of each iteration's normalized profile for each energy term
    plot_normalized_iterprofiles(ke, etaMidLev, 'KE', 'jet', 'FigF_panel_A')
    plot_normalized_iterprofiles(ape, etaMidLev, 'APE', 'jet', 'FigF_panel_B')
    plot_normalized_iterprofiles(qe, etaMidLev, 'QE', 'jet', 'FigF_panel_C')
    # generate bar-plot of total-column energy on each iteration
    totKE = np.sum(ke, axis=0)
    totAPE = np.sum(ape, axis=0)
    totQE = np.sum(qe, axis=0)
    # define x-axis values (iterations)
    x=np.arange(np.size(totKE))
    # define bar-width
    barWidth=0.3
    fig=plt.figure(figsize=(8,4))
    plt.bar(x=x-barWidth, height=totKE, width=barWidth, color='blue', edgecolor='white')
    plt.bar(x=x, height=totAPE, width=barWidth, color='red', edgecolor='white')
    plt.bar(x=x+barWidth, height=totQE, width=barWidth, color='green', edgecolor='white')
    plt.xticks(x)
    plt.xlabel('iteration')
    plt.ylabel('total-column perturbation energy (J/kg)')
    plt.legend(['KE', 'APE', 'QE'])
    plt.savefig('FigF_panel_D.png', bbox_inches='tight', facecolor='white')
    #
    # end
    #
