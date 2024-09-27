# FIGURE: Compute panels representing state of TC ventilation and its perturbations in the weakening experiment:
#    Panel-A: (contoured) unperturbed 850-200 hPa environmental shear, (shaded) perturbation to shear, with shear perturbations witin 900-km exclusion zone included as "environmental" shear
#    Panel-B: (contoured) unperturbed 600 hPa entropy deficit, (shaded) perturbation to entropy deficit
#    Panel-C: (contoured) unperturbed near-surface equivalent potential temperature, (shaded) perturbation to eth
#    Panel-D: (contoured) unperturbed TC outflow temperature, (shaded) perturbation to outflow-T
#
###################################### Some Notes ######################################
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
###################################### End Notes ######################################
#
# load modules
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import wrf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from analysis_dependencies import gen_cartopy_proj
from analysis_dependencies import get_uvmet
from analysis_dependencies import get_wrf_kinematic
from analysis_dependencies import compute_inverse_laplacian_with_boundaries
from analysis_dependencies import compute_hires_border
from analysis_dependencies import haversine
from analysis_dependencies import get_wrf_grad
from analysis_dependencies import get_wrf_rh
from analysis_dependencies import get_wrf_slp
from analysis_dependencies import get_wrf_eth
from analysis_dependencies import get_wrf_ethsat
from analysis_dependencies import interpolate_sigma_levels
from analysis_dependencies import compute_PI_vars
import datetime
#
# internal functions
#
#
# compute_TC_shear: The TC shear is computed as the combination of the nondivergent (psi) and irrotational (kai) flow
#                   from the 850-200 hPa differential vorticity and divergence within 900 km of the TC center. This is
#                   computed through inversion to produce the streamfunction (psi) and velocity potential (kai) and then
#                   the gradients are used to produce the vector components. This is computed on both the lo-res and hi-
#                   res grids, because the hi-res grid requires boundary information from the lo-res grid to solve psi
#                   and kai.
#
# INPUTS:
#    hdlLores: netCDF4.Dataset() handle of WRF forecast on lo-res grid
#    hdlHires: netCDF4.Dataset() handle of WRF forecast on hi-res grid
#    start_i: beginning i-location of hi-res grid on lo-res grid
#    start_j: beginning j-location of hi-res grid on lo-res grid
#    resRatio: ratio of hi-res to lo-res grid resolution (probably 3)
#
# OUTPUTS:
#    psiULores + kaiULores: total zonal TC shear on lo-res grid
#    psiVLores + kaiVLores: total merid TC shear on lo-res grid
#    psiUHires + kaiUHires: total zonal TC shear on hi-res grid
#    psiVHires + kaiVHires: total merid TC shear on hi-res grid
#
# DEPENDENCIES:
#    numpy (np)
#    netCDF4.Dataset()
#    wrf-python (wrf)
#    analysis_dependencies.get_wrf_kinematic()
#    analysis_dependencies.get_uvmet()
#    analysis_dependencies.get_wrf_slp()
#    analysis_dependencies.haversine()
#    analysis_dependencies.compute_inverse_laplacian_with_boundaries()
#    analysis_dependencies.compute_hires_border()
#    analysis_dependencies.get_wrf_grad()
def compute_TC_shear(hdlLores, hdlHires, start_i, start_j, resRatio):
    # compute 3D vorticity and divergence
    vorHires = get_wrf_kinematic(hdlHires,'vor')
    divHires = get_wrf_kinematic(hdlHires,'div')
    vorLores = get_wrf_kinematic(hdlLores,'vor')
    divLores = get_wrf_kinematic(hdlLores,'div')
    # interpolate vor, div, u, v to 850 and 200 hPa
    # hi-res grid
    u,v = get_uvmet(hdlHires)
    p = wrf.getvar(hdlHires,'p')
    vorHires850 = wrf.interplevel(field3d=vorHires,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vorHires200 = wrf.interplevel(field3d=vorHires,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    divHires850 = wrf.interplevel(field3d=divHires,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    divHires200 = wrf.interplevel(field3d=divHires,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    uwdHires850 = wrf.interplevel(field3d=u,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    uwdHires200 = wrf.interplevel(field3d=u,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vwdHires850 = wrf.interplevel(field3d=v,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vwdHires200 = wrf.interplevel(field3d=v,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    # lo-res grid
    u,v = get_uvmet(hdlLores)
    p = wrf.getvar(hdlLores,'p')
    vorLores850 = wrf.interplevel(field3d=vorLores,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vorLores200 = wrf.interplevel(field3d=vorLores,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    divLores850 = wrf.interplevel(field3d=divLores,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    divLores200 = wrf.interplevel(field3d=divLores,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    uwdLores850 = wrf.interplevel(field3d=u,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    uwdLores200 = wrf.interplevel(field3d=u,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vwdLores850 = wrf.interplevel(field3d=v,
                                  vert=p,
                                  desiredlev=85000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    vwdLores200 = wrf.interplevel(field3d=v,
                                  vert=p,
                                  desiredlev=20000.,
                                  missing=np.nan,
                                  squeeze=True,
                                  meta=False)
    # compute minimum SLP on hi-res grid to define storm-center
    slp = np.asarray(get_wrf_slp(hdlHires)).squeeze()
    cenjiHires = np.argmin(slp.flatten())
    # compute distance of each hi-res grid-point from storm-center
    latHires = np.asarray(hdlHires.variables['XLAT']).squeeze()
    lonHires = np.asarray(hdlHires.variables['XLONG']).squeeze()
    d = haversine(latHires.flatten()[cenjiHires],lonHires.flatten()[cenjiHires],latHires.flatten(),lonHires.flatten())
    distHires = d.reshape(np.shape(latHires))
    # compute distance of each lo-res grid-point from storm-center (defined on hi-res grid)
    latLores = np.asarray(hdlLores.variables['XLAT']).squeeze()
    lonLores = np.asarray(hdlLores.variables['XLONG']).squeeze()
    # NOTE: latHires/lonHires are invoked here for the center-point because we are computing the distance of all
    #       latLores/lonLores points from the center defined on the hi-res grid
    d = haversine(latHires.flatten()[cenjiHires],lonHires.flatten()[cenjiHires],latLores.flatten(),lonLores.flatten())
    distLores = d.reshape(np.shape(latLores))
    # define differentical vorticity and divergence as 200-hPa minus 850-hPa values, zeroed out outside of 900 km radius
    # hi-res grid
    diffVorHires = vorHires200 - vorHires850
    diffDivHires = divHires200 - divHires850
    diffVorHires[np.where(distHires>900.)] = 0.
    diffDivHires[np.where(distHires>900.)] = 0.
    # lo-res grid
    diffVorLores = vorLores200 - vorLores850
    diffDivLores = divLores200 - divLores850
    diffVorLores[np.where(distLores>900.)] = 0.
    diffDivLores[np.where(distLores>900.)] = 0.
    # compute streamfunction and velocity potential of differential vorticity/divergence on lo-res grid with zero boundaries
    # define forcing grid as differential vorticity grid
    forc2D = diffVorLores
    print('computing inverse-laplacian of differential vorticty: lo-res grid')
    # compute inverse-laplacian with zero-boundaries
    diffSFLores = compute_inverse_laplacian_with_boundaries(hdlLores, forc2D, boundaries=None)
    # define forcing grid as differential divergence grid
    forc2D = diffDivLores
    print('computing inverse-laplacian of differential divergence: lo-res grid')
    # compute inverse-laplacian with zero-boundaries
    diffVPLores = compute_inverse_laplacian_with_boundaries(hdlLores, forc2D, boundaries=None)
    # compute streamfunction on high-res grid, with boundaries informed from low-res grid
    # define forcing grid as high-res differential vorticity field
    forc2D = diffVorHires
    print('computing inverse-laplacian of differential vorticity: hi-res grid')
    # compute boundaries of hi-res grid
    lores2D = diffSFLores
    bound2D = compute_hires_border(lores2D, forc2D, start_i, start_j, resRatio)
    # compute inverse-laplacian on hi-res grid with boundaries
    lapl2D = compute_inverse_laplacian_with_boundaries(hdlHires, forc2D, boundaries=bound2D)
    diffSFHires = lapl2D
    diffSFbHires = bound2D
    # define forcing grid as high-res differential divergence field
    forc2D = diffDivHires
    print('computing inverse-laplacian of differential divergence: hi-res grid')
    # compute boundaries of hi-res grid
    lores2D = diffVPLores
    bound2D = compute_hires_border(lores2D, forc2D, start_i, start_j, resRatio)
    # compute inverse-laplacian on hi-res grid with boundaries
    lapl2D = compute_inverse_laplacian_with_boundaries(hdlHires, forc2D, boundaries=bound2D)
    diffVPHires = lapl2D
    diffVPbHires = bound2D
    # compute nondivergent and irrotational components of differential flow
    # hi-res grid (incl. boundary values)
    dPdx, dPdy = get_wrf_grad(hdlHires, diffSFHires + diffSFbHires)
    psiUHires = -dPdy
    psiVHires = dPdx
    dPdx, dPdy = get_wrf_grad(hdlHires, diffVPHires + diffVPbHires)
    kaiUHires = dPdx
    kaiVHires = dPdy
    # lo-res grid
    dPdx, dPdy = get_wrf_grad(hdlLores, diffSFLores)
    psiULores = -dPdy
    psiVLores = dPdx
    dPdx, dPdy = get_wrf_grad(hdlLores, diffVPLores)
    kaiULores = dPdx
    kaiVLores = dPdy
    # return total (psi+kai) differential TC flow
    return psiULores + kaiULores, psiVLores + kaiVLores, psiUHires + kaiUHires, psiVHires + kaiVHires
#
# begin
#
if __name__ == "__main__":
    # define unperturbed and perturbed forecast file-handles at low- and high-res
    dtInit = datetime.datetime(2021, 8, 28, 18)
    fcstHr = 0
    dtFcst = dtInit + datetime.timedelta(hours=fcstHr)
    dtFcstStr = datetime.datetime.strftime(dtFcst,'%Y-%m-%d_%H:00:00')
    # load base forecast files
    res='9km'
    unpFileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/wrfout_d01_' + dtFcstStr
    ptdFileHires = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res +'_res/wrfout_d01_' + dtFcstStr
    res='27km'
    unpFileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/' + res + '_res/wrfout_d01_' + dtFcstStr
    ptdFileLores = '/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/negative/uvTq/upper_troposphere/ptdi13/' + res +'_res/wrfout_d01_' + dtFcstStr
    unpHdlHires = Dataset(unpFileHires)
    ptdHdlHires = Dataset(ptdFileHires)
    unpHdlLores = Dataset(unpFileLores)
    ptdHdlLores = Dataset(ptdFileLores)
    # define lo- vs hi-res grid geometries
    start_i = 20
    start_j = 15
    resRatio = 3
    #
    ############## NO MODIFICATIONS BELOW THIS LINE ##############
    #
    # compute data and plot projections
    plotProj = ccrs.PlateCarree()
    datProj = gen_cartopy_proj(unpHdlHires)
    # compute lo-res and hi-res lat/lon grids
    latLores = np.asarray(unpHdlLores.variables['XLAT']).squeeze()
    lonLores = np.asarray(unpHdlLores.variables['XLONG']).squeeze()
    latHires = np.asarray(unpHdlHires.variables['XLAT']).squeeze()
    lonHires = np.asarray(unpHdlHires.variables['XLONG']).squeeze()
    # define hi-res subdomain
    borderHires = np.zeros(np.shape(latHires))
    borderHires[0,:] = 1.
    borderHires[-1,:] = 1.
    borderHires[:,0] = 1.
    borderHires[:,-1] = 1.
    # compute TC differential 850-200 hPa shear of unperturbed forecast (this takes a long time)
    # this is done on both lo-res and hi-res grids because the hi-res TC-flow is computed from boundary conditions borrowed from lo-res TC-flow
    unpTCULores, unpTCVLores, unpTCUHires, unpTCVHires = compute_TC_shear(unpHdlLores, unpHdlHires, start_i=start_i, start_j=start_j, resRatio=resRatio)
    # compute environmental differential flow as total differential 850-200 hPa flow minus TC flow
    # hi-res grid only
    u,v = get_uvmet(unpHdlHires)
    p = np.asarray(wrf.getvar(unpHdlHires,'p'))
    u850 = wrf.interplevel(field3d=u,
                           vert=p,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    v850 = wrf.interplevel(field3d=v,
                           vert=p,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    u200 = wrf.interplevel(field3d=u,
                           vert=p,
                           desiredlev=20000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    v200 = wrf.interplevel(field3d=v,
                           vert=p,
                           desiredlev=20000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    unpEnvUwdHires = u200 - u850 - unpTCUHires
    unpEnvVwdHires = v200 - v850 - unpTCVHires
    # compute perturbed environmental shear using unperturbed TC shear flow (wind perturbations within 900 km of TC center count as environment)
    # hi-res grid only
    u,v = get_uvmet(ptdHdlHires)
    p = np.asarray(wrf.getvar(ptdHdlHires,'p'))
    u850 = wrf.interplevel(field3d=u,
                           vert=p,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    v850 = wrf.interplevel(field3d=v,
                           vert=p,
                           desiredlev=85000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    u200 = wrf.interplevel(field3d=u,
                           vert=p,
                           desiredlev=20000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    v200 = wrf.interplevel(field3d=v,
                           vert=p,
                           desiredlev=20000.,
                           missing=np.nan,
                           squeeze=True,
                           meta=False)
    ptdEnvUwdHires = u200 - u850 - unpTCUHires
    ptdEnvVwdHires = v200 - v850 - unpTCVHires
    # compute entropy deficit at 600 hPa
    # unperturbed
    p = np.asarray(wrf.getvar(unpHdlHires,'p')).squeeze()
    unpEthSatHires600 = wrf.interplevel(field3d=np.asarray(get_wrf_ethsat(unpHdlHires)).squeeze(),
                                         vert=p,
                                         desiredlev=60000.,
                                         missing=np.nan,
                                         squeeze=True,
                                         meta=False)
    unpEthHires600 = wrf.interplevel(field3d=np.asarray(get_wrf_eth(unpHdlHires)).squeeze(),
                                     vert=p,
                                     desiredlev=60000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    # perturbed
    p = np.asarray(wrf.getvar(ptdHdlHires,'p')).squeeze()
    ptdEthSatHires600 = wrf.interplevel(field3d=np.asarray(get_wrf_ethsat(ptdHdlHires)).squeeze(),
                                         vert=p,
                                         desiredlev=60000.,
                                         missing=np.nan,
                                         squeeze=True,
                                         meta=False)
    ptdEthHires600 = wrf.interplevel(field3d=np.asarray(get_wrf_eth(ptdHdlHires)).squeeze(),
                                     vert=p,
                                     desiredlev=60000.,
                                     missing=np.nan,
                                     squeeze=True,
                                     meta=False)
    # compute pseudo-entropy as cp * log(Eth)
    # hi-res grid only
    cp = 1005.
    unpEntHires600 = cp * np.log(unpEthHires600)
    unpEntSatHires600 = cp * np.log(unpEthSatHires600)
    ptdEntHires600 = cp * np.log(ptdEthHires600)
    ptdEntSatHires600 = cp * np.log(ptdEthSatHires600)
    # compute potential intensity variables and landmask
    # hi-res grid only
    # unperturbed
    unpLMASKHires = np.asarray(unpHdlHires.variables['LANDMASK']).squeeze()
    x1,x2,x3,x4,x5 = compute_PI_vars(unpHdlHires)
    unpVMAXHires = x1.to_numpy()
    unpPMINHires = x2.to_numpy()
    unpIFLHires = x3.to_numpy()
    unpTOHires = x4.to_numpy()
    unpOTLHires = x5.to_numpy()
    # perturbed
    ptdLMASKHires = np.asarray(ptdHdlHires.variables['LANDMASK']).squeeze()
    x1,x2,x3,x4,x5 = compute_PI_vars(ptdHdlHires)
    ptdVMAXHires = x1.to_numpy()
    ptdPMINHires = x2.to_numpy()
    ptdIFLHires = x3.to_numpy()
    ptdTOHires = x4.to_numpy()
    ptdOTLHires = x5.to_numpy()
    # compute near-surface equivalent potential temperature
    # hi-res grid only
    # unperturbed
    qv = np.asarray(unpHdlHires.variables['Q2']).squeeze()
    tk = np.asarray(unpHdlHires.variables['T2']).squeeze()
    p = np.asarray(unpHdlHires.variables['PSFC']).squeeze()
    unpEthSfcHires = wrf.eth(qv[None,:,:],tk[None,:,:],p[None,:,:],meta=False,units='K').squeeze()
    # perturbed
    qv = np.asarray(ptdHdlHires.variables['Q2']).squeeze()
    tk = np.asarray(ptdHdlHires.variables['T2']).squeeze()
    p = np.asarray(ptdHdlHires.variables['PSFC']).squeeze()
    ptdEthSfcHires = wrf.eth(qv[None,:,:],tk[None,:,:],p[None,:,:],meta=False,units='K').squeeze()
    #
    # PLOTS
    #
    # PANEL-A: 850-200 environmental shear perturbation (treating all wind perts. as environmental here), and SLP
    #
    # Define shading and contouring ranges
    shdrng = np.arange(-12., 12.1, 2.).astype('float16')
    mask = np.ones(np.shape(shdrng), dtype='bool')
    mask[np.where(shdrng==0.)] = False
    cntrng = np.arange(900., 10401, 4.)
    colmap='bwr'
    # Define figure and axis, set projection to datProjLores
    fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(16,12),subplot_kw={'projection':datProj})
    ax=axs
    # 1. Plot hi-res shading, zorder=3
    shd=ax.contourf(lonHires, latHires, np.sqrt(ptdEnvUwdHires**2. + ptdEnvVwdHires**2.)-np.sqrt(unpEnvUwdHires**2. + unpEnvVwdHires**2.), shdrng[mask], cmap=colmap, transform=plotProj,zorder=3, extend='both')
    ax.contour(lonHires, latHires, get_wrf_slp(unpHdlHires), cntrng, colors='black', transform=plotProj, zorder=3)
    # 2. Plot map
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
    # 3. Plot colorbar for shd
    plt.colorbar(mappable=shd, ax=ax)
    ax.set_title('environmental 850-200 hPa shear')
    plt.savefig('ventilation_a.png', bbox_inches='tight', facecolor='white')
    #
    # PANEL-B: 600 hPa entropy deficit perturbation and SLP
    #
    # Define shading and contouring ranges
    shdrng = np.arange(-18., 18.1, 2.).astype('float16')
    mask = np.ones(np.shape(shdrng), dtype='bool')
    mask[np.where(shdrng==0.)] = False
    cntrng = np.arange(900., 1040.1, 4.)
    colmap = 'bwr'
    # Define figure and axis, set projection to datProjLores
    fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(16,12),subplot_kw={'projection':datProj})
    ax=axs
    # 1. Plot hi-res shading, zorder=3
    shd=ax.contourf(lonHires, latHires, (ptdEntSatHires600-ptdEntHires600)-(unpEntSatHires600-unpEntHires600), shdrng[mask], cmap=colmap, transform=plotProj,zorder=3, extend='both')
    ax.contour(lonHires, latHires, get_wrf_slp(unpHdlHires), cntrng, colors='black', transform=plotProj, zorder=3)
    # 2. Plot map
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
    # 3. Plot colorbar for shd
    plt.colorbar(mappable=shd, ax=ax)
    ax.set_title('600 hPa entropy deficit (pseudo)')
    plt.savefig('ventilation_b.png', bbox_inches='tight', facecolor='white')
    #
    # PANEL-C: near-surface equivalent potential temperature perturbation, and SLP
    #
    # Define shading and contouring ranges
    shdrng = np.arange(-3., 3.1, 0.25).astype('float16')
    mask = np.ones(np.shape(shdrng), dtype='bool')
    mask[np.where(shdrng==0.)] = False
    cntrng = np.arange(900., 1040.1, 4.)
    colmap = 'bwr'
    # Define figure and axis, set projection to datProjLores
    fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(16,12),subplot_kw={'projection':datProj})
    ax=axs
    # 1. Plot hi-res shading, zorder=3
    shd=ax.contourf(lonHires, latHires, np.ma.array(ptdEthSfcHires, mask = ptdLMASKHires==1)-np.ma.array(unpEthSfcHires, mask = unpLMASKHires==1), shdrng[mask], cmap=colmap, transform=plotProj,zorder=3, extend='both')
    ax.contour(lonHires, latHires, get_wrf_slp(unpHdlHires), cntrng, colors='black', transform=plotProj, zorder=3)
    # 2. Plot map
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
    # 3. Plot colorbar for shd
    plt.colorbar(mappable=shd, ax=ax)
    ax.set_title('near-surface equivalent potential temperature')
    plt.savefig('ventilation_c.png', bbox_inches='tight', facecolor='white')
    #
    # PANEL-D: outflow temperature perturbation, and SLP
    #
    # Define shading and contouring ranges
    shdrng = np.arange(-4., 4.1, 0.5).astype('float16')
    mask = np.ones(np.shape(shdrng), dtype='bool')
    mask[np.where(shdrng==0.)] = False
    cntrng = np.arange(900., 1040.1, 4.)
    colmap = 'bwr'
    # Define figure and axis, set projection to datProjLores
    fig,axs = plt.subplots(ncols=1,nrows=1,figsize=(16,12),subplot_kw={'projection':datProj})
    ax=axs
    # 4. Plot hi-res shading, zorder=3
    shd=ax.contourf(lonHires, latHires, ptdTOHires-unpTOHires, shdrng[mask], cmap=colmap, transform=plotProj,zorder=3, extend='both')
    ax.contour(lonHires, latHires, get_wrf_slp(unpHdlHires), cntrng, colors='black', transform=plotProj, zorder=3)
    # 8. Plot map
    ax.add_feature(cfeature.COASTLINE, edgecolor='brown',zorder=7)
    # 9. Plot colorbar for shd
    plt.colorbar(mappable=shd, ax=ax)
    ax.set_title('outflow temperature (K)')
    plt.savefig('ventilation_d.png', bbox_inches='tight', facecolor='white')
#
# end
#

