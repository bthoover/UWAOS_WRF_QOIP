################################################################################################
#
# calc_<kinematic>(): Compute some kinematic property (vor, div, str, shr) from
# the wind components, with map-factors used in computing grid spacing (dx,dy) at
# each point via:
#
# dx[j,i] = ds/mf[j,i]
# dy[j,i] = ds/mf[j,i]
#
# Where ds is the nominal grid spacing and mf is the map-factor. These computations
# are all performed on the mass-points, which requires de-staggering the u- and v-
# components of the wind first, using wrf.destagger(), and then rotating into earth-
# relative coordinates using wrf.uvmet(). See Compute_sens_kinematics.py for an
# example.
# 
# INPUTS:
#    u ......................................................... zonal flow grid, may be an xarray directly from wrf.getvar() or similar
#    v ......................................................... merid flow grid, may be an xarray directly from wrf.getvar() or similar
#    dx ........................................................ grid spacing in x-direction at each [j,i] point, with map-factor adjustment (see above)
#    dy ........................................................ grid spacing in y-direction at each [j,i] point, with map-factor adjustment (see above)
#
# OUTPUTS:
#    some kinematic field (vor,div,str,shr) on mass-points
#
# NOTES:
#    uses numPy
#    
def calc_div(u,v,dx,dy):
    import numpy as np
    ug  = np.asarray(u).squeeze()
    vg  = np.asarray(v).squeeze()
    dxg = np.asarray(dx).squeeze()
    dyg = np.asarray(dy).squeeze()
    nz,ny,nx = np.shape(ug)
    
    div = np.nan * np.ones((nz,ny,nx))
    for k in range(nz):
        for j in range(1,ny-1,1):
            for i in range(1,nx-1,1):
                dudx = (ug[k,j,i+1]-ug[k,j,i-1])/(2.*dxg[j,i])
                dvdy = (vg[k,j+1,i]-vg[k,j-1,i])/(2.*dyg[j,i])
                div[k,j,i] = dudx + dvdy
    return div

def calc_vor(u,v,dx,dy):
    import numpy as np
    ug  = np.asarray(u).squeeze()
    vg  = np.asarray(v).squeeze()
    dxg = np.asarray(dx).squeeze()
    dyg = np.asarray(dy).squeeze()
    nz,ny,nx = np.shape(ug)
    
    vor = np.nan * np.ones((nz,ny,nx))
    for k in range(nz):
        for j in range(1,ny-1,1):
            for i in range(1,nx-1,1):
                dvdx = (vg[k,j,i+1]-vg[k,j,i-1])/(2.*dxg[j,i])
                dudy = (ug[k,j+1,i]-ug[k,j-1,i])/(2.*dyg[j,i])
                vor[k,j,i] = dvdx - dudy
    return vor


def calc_str(u,v,dx,dy):
    import numpy as np
    ug  = np.asarray(u).squeeze()
    vg  = np.asarray(v).squeeze()
    dxg = np.asarray(dx).squeeze()
    dyg = np.asarray(dy).squeeze()
    nz,ny,nx = np.shape(ug)
    
    str = np.nan * np.ones((nz,ny,nx))
    for k in range(nz):
        for j in range(1,ny-1,1):
            for i in range(1,nx-1,1):
                dudx = (ug[k,j,i+1]-ug[k,j,i-1])/(2.*dxg[j,i])
                dvdy = (vg[k,j+1,i]-vg[k,j-1,i])/(2.*dyg[j,i])
                str[k,j,i] = dudx - dvdy
    return str


def calc_shr(u,v,dx,dy):
    import numpy as np
    ug  = np.asarray(u).squeeze()
    vg  = np.asarray(v).squeeze()
    dxg = np.asarray(dx).squeeze()
    dyg = np.asarray(dy).squeeze()
    nz,ny,nx = np.shape(ug)
    
    shr = np.nan * np.ones((nz,ny,nx))
    for k in range(nz):
        for j in range(1,ny-1,1):
            for i in range(1,nx-1,1):
                dvdx = (vg[k,j,i+1]-vg[k,j,i-1])/(2.*dxg[j,i])
                dudy = (ug[k,j+1,i]-ug[k,j-1,i])/(2.*dyg[j,i])
                shr[k,j,i] = dvdx + dudy
    return shr
#
################################################################################################
#
# sensitivity_to_kinematic_by_level(): A function that will compute the quantity:
#
#   -<inverse-laplacian>(kin)
#
# for some kinematic field: vorticity, divergence, stretching-deformation, or
# shearing-deformation. This function exists because the above quantity is the
# calculation for the sensitivity to that kinematic field, when the kinematic
# field is computed from the sensitivity to the u- and v-components of the wind.
# For example, the sensitivity with respect to vorticity is calculated as:
#
# -<inverse-laplacian>(d/dx(v_a) - d/dy(u_a))
#
# where u_a and v_a are the sensitivity to the u- and v-components of the wind.
# Thus, you can compute the sensitivity to vorticity by computing the "vorticity"
# of the sensitivity to the u- and v-components of the wind, and then applying
# the -<inverse-laplacian>() function.
#
# INPUTS:
#
#    kin .............................................. "kinematic" field computed from sensitivity to u and v [nz,ny,nx]
#    k ................................................ vertical level to operate on (output is for single level at lev=k)
#    nx ............................................... number of points in x-direction
#    ny ............................................... number of points in y-direction
#    nz ............................................... number of vertical levels
#    ds_x ............................................. grid spacing in x-direction
#    ds_y ............................................. grid spacing in y-direction
#    msfm ............................................. map-factors on mass-points
#
# OUTPUTS:
#
#    full_psi ......................................... sensitivity with respect to kinematic field computed as -<inverse-laplacian>(kin)
#
# NOTES:
#
#    uses numPy, wrf(-python), sciPy routines
#    uses convention of i-points are meridional and j-points are zonal
#
def sensitivity_to_kinematic_by_level(kin,k,nx,ny,nz,ds_x,ds_y,msfm):
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
    # Define psi
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
    forcegrid = -1.*kin[k,1:ny-1,1:nx-1]/msfm[1:ny-1,1:nx-1]**2.0 #.......................... forcegrid (initialized to -kin, divided by square of map-factors)
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
    field, value, info = scipy.optimize.fmin_l_bfgs_b(fun, psi0.flatten(), grad, pgtol=1e-18, factr=.0001,
                                                  maxiter=150000,maxfun=15000*50)
    # Stop clock
    end = time.time() #...................................................................... Ending time
    #
    #########################################################################################
    #
    # Report timing statistics for solving on level-k
    print('time for level ', k,  '= ', end - start, ' seconds.' )
    print('converged after {:d} iterations'.format(info['nit']))
    # Define full_psi as reshaped matrix of converged psi
    full_psi[:,:] = psi.copy()
    full_psi[1:-1,1:-1] = reshape(field, (nym2, nxm2)).copy()    
    # Return full_psi
    return full_psi[:,:]
    #
    # END
    #

