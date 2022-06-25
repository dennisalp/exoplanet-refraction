# Calculates the a light curve with quadratic limb darkening and
# refraction. The lensing model is presented in Hui & Seager (2002).
#
# CHECK HOW MANY IMAGES ARE THROWN AWAY AND SOLVE PHI FOR DIVIDE BETWEEN PRIMARY AND SECONDARY
# AND THEN REDO ALL OPTIMIZATIONS AND EDIT NOTES
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import pdb

from exo_trans_occult3 import occultquad

__author__ = "Dennis Alp"
__copyright__ = "Copyright 2017"
__credits__ = ["Dennis Alp"]
__date__ = "2017-01-16"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Dennis Alp"
__email__ = "dalp@kth.se"
__status__ = "Production"



################################################################
# Define some wisely chosen values that probably do not need to be
# changed.
EPSILON = 1e-12
LARGE_EXP = 24. # Make exp(-LARGE_EXP) small

# These are the number of r and theta values that will be integrated
# over. These are much cheaper than the image positions that need to
# be tessellated. Therefore, set more of these. Choose number of
# thetas (NTs) to be even to avoid pi/2 difficulties.
GRID_NR = 2048
GRID_NT = 1024

# These are the number of primary image positions that will be mapped
# back to the source plane and tessellated into the regularly spaced
# integration grid. Therefore, these are expensive.
PRI_NR = 1024
PRI_NT = 512
SEC_NR = 1024
SEC_NT = 512

# Extra fine
#GRID_NR = 4096
#GRID_NT = 2048
#PRI_NR = 2048
#PRI_NT = 1024
#SEC_NR = 2048
#SEC_NT = 1024

## Super extra fine
#GRID_NR = 8192
#GRID_NT = 4096
#PRI_NR = 4096
#PRI_NT = 2048
#SEC_NR = 4096
#SEC_NT = 2048

# Sparse, enough for shoulders
#GRID_NR = 128
#GRID_NT = 64
#PRI_NR = 64
#PRI_NT = 32
#SEC_NR = 128
#SEC_NT = 32

# Sparse, enough for shoulders
#GRID_NR = 256
#GRID_NT = 128
#PRI_NR = 128
#PRI_NT = 64
#SEC_NR = 256
#SEC_NT = 64

################################################################
# Function for calculating the transit light curve
def refraction_transit(zz, g1, g2, BB, HH, p0):
    ########
    # Some smaller help functions
    # Stellat intensity profile
    def II(rr):
        mu = np.sqrt(1-rr**2)
        return 1  - g1*(1-mu) - g2*(1-mu)**2
    
    # Polar magnitude from cartesian coordinates
    def rho(xx, yy):
        return np.sqrt(xx**2+yy**2)

    # Lensing potential phi(u)
    def phi(uu):
        return -BB*np.sqrt(np.pi*HH/(2.*uu))*np.exp(-(uu-p0)/HH)

    # Lensing potential u^2*tildephi(u)
    def u2phi(uu):
        return BB*np.sqrt(np.pi*HH/(2*uu))*(1/2.+uu/HH)*np.exp(-(uu-p0)/HH)

    # Help function for computation of critical curve
    def fcrit(xx):
        return 1+phi(xx)

    # Source grid for primary images
    def get_pri_grid(shift, r_max):
        r_min = np.max((shift-1+EPSILON, EPSILON))
        rr_int = np.linspace(r_min, r_max, GRID_NR)
        rr = np.tile(rr_int, GRID_NT)
        
        th = np.linspace(0, 1., GRID_NT)
        th = np.repeat(th , GRID_NR)
        
        # Law of cosines, restricting to source area covered by star
        # This convoluted way prevents unnecessary warnings from arccos
        hlp = rr + shift > 1.
        th[hlp] *= np.arccos((rr[hlp]**2+z0**2-1)/(2*rr[hlp]*z0))
        th[-hlp] *= np.pi
        grid1 = rr*np.cos(th)
        grid2 = rr*np.sin(th)
        
        # Converting back to star coordinate system
        rr_star = np.sqrt((grid1-z0)**2+grid2**2)
        # Catch some numerical instabilities
        rr_star = np.where((rr_star > 1.) & (rr_star < 1+EPSILON), 1., rr_star)
        return rr, grid1, grid2, th.reshape((GRID_NT, GRID_NR)), rr_int, rr_star

    # Variables for primary images
    def get_pri_vars():
        # Define the r-theta points in the image plane
# #        imRR = np.tile(np.linspace(p0-EPSILON,p0+LARGE_EXP*HH, PRI_NR), PRI_NT)
        imRR = np.tile(np.linspace(critical-EPSILON,p0+LARGE_EXP*HH, PRI_NR), PRI_NT)
        imTH = np.repeat(np.linspace(0., np.pi, PRI_NT), PRI_NR)
        img1 = imRR*np.cos(imTH)
        img2 = imRR*np.sin(imTH)
        
        # Help variables
        uu = rho(img1, img2) # Note that uu_P is u*D_OL in Hui & Seager (2002)
        phiu = phi(uu)
        u2phiu = u2phi(uu)
        
        # Map image positions to source plane
        src1 = img1*(1+phiu)
        src2 = src1*img2/img1
        
        # Compute magnification
        AA = 1/((1+phiu)*(1+phiu+u2phiu))
        
        # Screen unwanted source positions 
        WW  = np.where(uu < p0, 0., 1.) # Planet occultation kernel
        pri = np.where(img1*src1 >= 0.) # Selects primary images 
        AA  = AA[pri]*WW[pri]
        src1 = src1[pri]
        src2 = src2[pri]
        return src1, src2, AA

    # Variables for secondary images
    def get_sec_vars():
        rr_int = np.linspace(EPSILON, 1., GRID_NR) # int for integration
        th_int = np.linspace(0, np.pi, GRID_NT)
        rr = np.tile(rr_int, GRID_NT)
        th = np.repeat(th_int , GRID_NR)
        grid1 = rr*np.cos(th)
        grid2 = rr*np.sin(th)

        # Define the r-theta points in the image plane. Note that secondary
        # images must be on the far side of the star. Therefore, no images can
        # be on the upper half of the stellar disk and only images close to
        # planet limb are necessary.
# #        imRR = np.tile(np.linspace(p0-EPSILON,p0+LARGE_EXP*HH, SEC_NR), SEC_NT)
        imRR = np.tile(np.linspace(p0-EPSILON, critical+EPSILON, SEC_NR), SEC_NT)
        imTH = np.repeat(np.linspace(np.pi, 2*np.pi, SEC_NT), SEC_NR)
        img1 = imRR*np.cos(imTH)
        img2 = imRR*np.sin(imTH)
        
        # Help variables
        uu = rho(img1, img2) # Note that uu is u*D_OL in Hui & Seager (2002)
        phiu = phi(uu)
        u2phiu = u2phi(uu)
        
        # Map image positions to source plane
        src1 = img1*(1+phiu)
        src2 = src1*img2/img1
        
        # Compute magnification
        AA = 1/((1+phiu)*(1+phiu+u2phiu))
        
        # Screen unwanted source positions
        WW  = np.where(uu < p0, 0., 1.) # Planet occultation kernel
        sec = np.where(img1*src1 < 0.) # Selects secondary images
        AA = AA[sec]*WW[sec]
        src1 = src1[sec]
        src2 = src2[sec]
        return rr, grid1, grid2, th_int, rr_int, src1, src2, AA
    
    ################################################################
    # The main function that computes the flux at each point
    def calc_flux(z0, g1, g2, BB, HH, p0):
        # Catch trivial case
        if z0 >= 1+p0+LARGE_EXP*HH and not strong_lensing:
            return 0.

        flux = 0
        ########
        # Primary images
        if z0 < 1+p0+LARGE_EXP*HH:
            rr, grid1, grid2, th_int, rr_int, rr_star = get_pri_grid(z0, p0+LARGE_EXP*HH)
            
            AA = griddata(np.vstack((pri_src1, pri_src2)).T, pri_AA,\
                 np.vstack((grid1, grid2)).T, method='nearest', fill_value=0)
            
            #Integrate using composite Simpson's rule
            integrand = 2*rr*II(rr_star)*AA
            integrand = np.where(np.isnan(integrand) | np.isinf(integrand), 0, integrand) #Should not be needed
            integrand = integrand.reshape((GRID_NT, GRID_NR))
            integrand = simps(integrand, th_int, axis=0)
            flux += simps(integrand, rr_int, axis=0)
    
        ########
        # Secondary images
        if strong_lensing:
            AA = griddata(np.vstack((sec_src1-z0,sec_src2)).T, sec_AA,\
                 np.vstack((sec_grid1, sec_grid2)).T, method='nearest', fill_value=0)
    
            #Integrate using composite Simpson's rule
            integrand = 2*sec_rr*II(sec_rr)*np.abs(AA) # Note abs() here!
            integrand = np.where(np.isnan(integrand) | np.isinf(integrand), 0, integrand) #Should not be needed
            integrand = integrand.reshape((GRID_NT, GRID_NR))
            integrand = simps(integrand, sec_th_int, axis=0)
            flux += simps(integrand, sec_rr_int, axis=0)

        return flux/norm


    
    ################################################################
    # Parse input and prepare
    if np.any(zz) < 0:
        print("WARNING: z < 0, interpreting as abs(z)")
    zz = abs(zz)
    
    if p0 < 0:
        print("WARNING: planet radius p0 < 0, interpreting as abs(p0)")
    p0 = abs(p0)

    ########
    # Unocculted flux
    norm = np.pi/6*(6-2*g1-g2)
    # Non-refracted part
    FF = occultquad(zz, g1, g2, p0+LARGE_EXP*HH)[0]

    ########
    # Common variables for all z
    strong_lensing = 1-np.sqrt(np.pi/2)*np.sqrt(HH/p0)*BB < 0

    # Find critical curve (radius of circle centered on planet)
    if strong_lensing:
        critical = fsolve(fcrit, p0)[0]
        #print("Critical radius:", critical)
    else:
        critical = p0-EPSILON
        
    pri_src1, pri_src2, pri_AA = get_pri_vars()    
    sec_rr, sec_grid1, sec_grid2, sec_th_int, sec_rr_int, sec_src1, sec_src2, sec_AA = get_sec_vars()
    
    ########
    # Computing the remaining contribution
    for idx, z0 in enumerate(zz):
        print(idx,z0)
        FF[idx] += calc_flux(z0, g1, g2, BB, HH, p0)

    return FF




################################################################
# Test cases
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
########
# Constants, cgs
    kB = 1.380658e-16 # Boltzmann constant
    mu = 2. # Mean molecular wight, probably H2 for Jovians
    mH = 1.6733e-24 # Hydrogen mass
    
########
# Beta Pictoris
#    Rstar = 1.253e11
#    # Beta Pictoris b
#    # Observables
#    TT = 1600 # Temperature
#    rho0 = 3e-6 # Density at optical depth 1. Pretty much taken this out of nowhere.
#    alpha = 1.2 # Refractive index, shouldn't vary too much
#    gg = 10**3.8 # Surface gravity
#    MM = 11*1.898e30 # Mass
#    RR = 1.65*6.9911e+9/Rstar # Radius
#    Dls = 9*1.496e13/Rstar # Orbital distance
#    
#    # Parameters
#    HH = kB*TT/(gg*mu*mH)/Rstar
#    BB = 2*alpha*rho0*Dls/HH
#    print HH, BB, RR/HH, HH*Rstar
    
########
# Sun
#    Rstar = 6.957e10
#    # Jupiter
#    # Observables
#    TT = 128 # Temperature
#    rho0 = 3.5e-5 # Density at optical depth 1. Pretty much taken this out of nowhere.
#    alpha = 1.2 # Refractive index, shouldn't vary too much
#    gg = 2479 # Surface gravity
#    MM = 1.898e30 # Mass
#    RR = 7.1e+9/Rstar # Radius
#    Dls = 5.2*1.496e13/Rstar # Orbital distance
#    
#    # Parameters
#    HH = kB*TT/(gg*mu*mH)/Rstar
#    BB = 2*alpha*rho0*Dls/HH
#    print(HH, BB, RR/HH, HH*Rstar, RR)
    
########
# Orbital parameters
    t=np.linspace(0.,1.3,200)
    z=np.sqrt(t**2+(2*0.084)**2) # hui02
#    z=t

#    exec( "\n".join( sys.argv[1:] ))  # run this.py npt= dim= ...
    
########
# Compute the light curve
    x = refraction_transit(z, 0.35, 0.225, 40.3, 0.084/117.3, 0.084) # Fiducial model of Hui & Seager (2002)
#    x = refraction_transit(z, 0.35, 0.225, BB, HH, RR)
    
    x=np.concatenate((x[1:][::-1],x))
    z=np.concatenate((-z[1:][::-1],z))
    t=np.concatenate((-t[1:][::-1],t))
    np.savetxt('/Users/silver/Dropbox/sci/pro_exo/models/hui02_test.dat', np.c_[t,x])
#    plt.plot(t,x-1,'ok')
#    plt.plot(t,x-1)
#    plt.show()
