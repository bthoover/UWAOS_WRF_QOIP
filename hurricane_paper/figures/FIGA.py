# FIG A: 24-hr forecast profiles for (a) Ida 2021, and (b) Michael 2018: net divergence in 400 km of storm center
#        (blue bars) and vertical weighting profile of sigma-levels (orange)
#
# import modules
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import wrf
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_wrf_kinematic
import datetime
from scipy.stats import beta
# define internal functions
# internal functions:
#
# haversine_np: compute the distance between 2 (lat,lon) points, or between
# a vector of (lat,lon) points and a single point. Uses great-circle distance
# estimated from Haversine function using the WRF's approximation of the Earth
# sphere to define radius.
#
# INPUTS:
#
# lat1: starting latitude (deg)
# lon1: starting longitude (deg)
# lat2: ending latitude (deg)
# lon2: ending longitude (deg)
#
# NOTE: either (lat1,lon1) can be a vector or array of points, or (lat2,lon2),
#       but not both. One of these sets has to be a scalar to compute the distance
#       to a single point for all points in the vector/array set.
#
# OUTPUTS:
#
# km: distance (km, assuming Earth is a sphere of 6370 km radius)
#
# DEPENDENCIES:
#
# numpy
def haversine_np(lat1, lon1, lat2, lon2):
    import numpy as np
    # assert all (lat,lon) values as radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # compute delta-(lat,lon) between points 1 and 2
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine equation for unit sphere
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # scale to Earth approx. radius
    km = 6370 * c  # using same km radius as WRF earth-approximation
    return km


# compute_outflow_profile: Generates a vertical profile in native (sigma) coordinates of
# the summed divergence within a chosen radius of a chosen grid-point.
#
# INPUTS:
#
# wrfHDL: netCDF4.Dataset() file-handle of WRF file
# jCen: j value in 2D [j,i]-space to center the radius
# iCen: i value in 2D [j,i]-space to center the radius
# radMin: minimum radius from [j,i] point to sum divergence (km)
# radMax: maximum radius from [j,i] point to sum divergence (km)
#
# OUTPUTS:
#
# divProf: profile of summed divergence between radMin and radMax radii of point [jCen,iCen]
#
# DEPENDENCIES:
#
# numpy
# netCDF4.Dataset()
# analysis_dependencies.get_wrf_kinematic()
# haversine_np
def compute_outflow_profile(wrfHDL, jCen, iCen, radMin=0., radMax=400.):
    import numpy as np
    from netCDF4 import Dataset
    from analysis_dependencies import get_wrf_kinematic
    # pull latitude and longitude from wrfHDL
    lat = np.asarray(unpHdl.variables['XLAT']).squeeze()
    lon = np.asarray(unpHdl.variables['XLONG']).squeeze()
    fix = np.where(lon < 0.)
    lon[fix] = lon[fix] + 360.
    # pull divergence from wrfHDL
    div = np.asarray(get_wrf_kinematic(wrfHDL,'div')).squeeze()
    # compute distance of all 2D [j,i]-space points from jCen, iCen
    dist = haversine_np(lat,lon,lat[jCen,iCen],lon[jCen,iCen])
    # find all 2D [j,i]-space points within radMin and radMax of center-point 
    inRad = np.where((dist.flatten() >= radMin) & (dist.flatten() <= radMax))
    # define divProf as NaN vector and fill with summed divergence on each level
    divProf = np.nan * np.ones((np.shape(div)[0],))
    for k in range(np.size(divProf)):
        dk = np.asarray(div[k,:,:].squeeze())
        divProf[k] = np.nansum(dk.flatten()[inRad])
    # return
    return divProf

# plot_profile: given profiles of net divergence and weighting coefficients on sigma-levels, as well as the
#               sigma values on each level, return a profile plot.
#
# INPUTS:
#
# netDivergence: (nz,) profile of net divergence
# weightCoeff: (nz,) profile of weighting coefficients
# sigVals: (nz,) profile of sigma-values for each level
#
# OUTPUTS:
#
# figure-panel of vertical profiles
#
def plot_profile(netDivergence, weightCoeff, sigVals):
    # define figure-handle and bottom-x axis
    fig, ax1 = plt.subplots(figsize=(6,9))
    # bottom-x axis: plot net divergence as blue bars and a zero-line
    ax1.barh(y=sig, width=unpNetDiv, height=0.015, color='blue')
    ax1.plot(0.*unpNetDiv, sig, color='gray', linewidth=0.75, linestyle='dashed')
    # create a twin top-x axis
    ax2 = plt.twiny(ax1)
    # top-x axis: plot weighting coefficients as orange line and a zero-line
    ax2.plot(w, sig, color='#de980b', linewidth=2.)
    ax2.plot(0.*w, sig, color='gray', linewidth=0.75, linestyle='dashed')
    # invert y-axis
    plt.gca().invert_yaxis()
    # set x-axis colors
    plt.gca().spines['bottom'].set_color('blue')
    plt.gca().spines['top'].set_color('#de980b')
    # set x-axis tick colors
    ax1.tick_params(axis='x', colors='blue')
    ax2.tick_params(axis='x', colors='#de980b')
    # set x-axis labels
    ax1.set_xlabel('net divergence in 400 km of center')
    ax2.set_xlabel('weighting applied to sigma-level')
    # set y-axis label
    ax1.set_ylabel('sigma value')
    # return figure-handle
    return fig

#
# begin
#
if __name__=="__main__":
    # FIG Aa: Ida (2021) profiles
    res = '9km_res'
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '/'
    dtInit = datetime.datetime(2021, 8, 28, 18)
    fcstHr = 24
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst, '%Y-%m-%d_%H:00:00')
    unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFile)
    sig = np.asarray(unpHdl.variables['ZNU']).squeeze()
    unpSLP = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    ji_min = np.unravel_index(np.argmin(unpSLP), np.shape(unpSLP))
    ji_min = np.asarray(ji_min)
    ju,iu = ji_min
    unpNetDiv = compute_outflow_profile(unpHdl, ju, iu, 0., 400.)
    nz=np.size(unpNetDiv)
    kmin=29.
    kmode=54.
    kmax=59.
    kmean = (kmin + 4.*kmode + kmax)/6.
    PERT_a = 6.*((kmean - kmin)/(kmax - kmin))
    PERT_b = 6.*((kmax - kmean)/(kmax - kmin))
    pd = beta(PERT_a,PERT_b,loc=kmin,scale=kmax-kmin)
    p = pd.pdf(np.arange(nz))
    w = p/np.max(p)
    fig = plot_profile(unpNetDiv, w, sig)
    fig.savefig('FIGAa.png', bbox_inches='tight', facecolor='white')
    
    # FIG Ab: Michael (2018) profiles
    res = '9km_res'
    unpDir = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Michael2018/R_mu/unperturbed/' + res + '/'
    dtInit = datetime.datetime(2018, 10, 9, 12)
    fcstHr = 24
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst, '%Y-%m-%d_%H:00:00')
    unpFile = unpDir + 'wrfout_d01_' + dtFcstStr
    unpHdl = Dataset(unpFile)
    sig = np.asarray(unpHdl.variables['ZNU']).squeeze()
    unpSLP = np.asarray(get_wrf_slp(unpHdl)).squeeze()
    ji_min = np.unravel_index(np.argmin(unpSLP), np.shape(unpSLP))
    ji_min = np.asarray(ji_min)
    ju,iu = ji_min
    unpNetDiv = compute_outflow_profile(unpHdl, ju, iu, 0., 400.)
    nz=np.size(unpNetDiv)
    kmin=29.
    kmode=54.
    kmax=59.
    kmean = (kmin + 4.*kmode + kmax)/6.
    PERT_a = 6.*((kmean - kmin)/(kmax - kmin))
    PERT_b = 6.*((kmax - kmean)/(kmax - kmin))
    pd = beta(PERT_a,PERT_b,loc=kmin,scale=kmax-kmin)
    p = pd.pdf(np.arange(nz))
    w = p/np.max(p)
    fig = plot_profile(unpNetDiv, w, sig)
    fig.savefig('FIGAb.png', bbox_inches='tight', facecolor='white')
#
# end
#
