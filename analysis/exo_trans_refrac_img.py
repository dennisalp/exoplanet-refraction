# Calculates the a light curve with quadratic limb darkening and
# refraction. The lensing model is presented in Hui & Seager (2002).
#
# Temporary
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import griddata

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
LARGE_EXP = 40. # Make exp(-LARGE_EXP) small
MAJOR = 0.5

# These are the number of r and theta values that will be integrated
# over. These are much cheaper than the image positions that need to
# be tessellated. Therefore, set more of these. Choose number of
# thetas (NTs) to be even to avoid pi/2 difficulties.
GRID_NR = 4096
GRID_NT = 1024

# These are the number of primary image positions that will be mapped
# back to the source plane and tessellated into the regularly spaced
# integration grid. Therefore, these are expensive.
STAR_NR = 4096
STAR_NT = 16
PRI_PLA_NR = 1024
PRI_PLA_NT = 512
SEC_PLA_NR = 1024
SEC_PLA_NT = 512

# Sparse
#GRID_NR = 256
#GRID_NT = 128
#STAR_NR = 64
#STAR_NT = 32
#PRI_PLA_NR = 256
#PRI_PLA_NT = 128
#SEC_PLA_NR = 256
#SEC_PLA_NT = 128



################################################################
# Function for calculating the transit light curve
def refraction_transit(z0, g1, g2, BB, HH, p0):
########
# Catch trivial case
    caustics = 1-np.sqrt(np.pi/2)*np.sqrt(HH/p0)*BB < 0
    if z0 > 1+p0+LARGE_EXP*HH and not caustics:
        return 1.

########
# Help functions
    def II(rr): # Stellat intensity profile
        mu = np.sqrt(1-rr**2)
        return 1  - g1*(1-mu) - g2*(1-mu)**2
    
    def rho(xx, yy): # Polar magnitude from cartesian coordinates
        return np.sqrt(xx**2+yy**2)
        
    def phi(uu): # Lensing potential phi(u)
        return -BB*np.sqrt(np.pi*HH/(2.*uu))*np.exp(-(uu-p0)/HH)

    def u2phi(uu): # Lensing potential u^2*tilde-phi(u)
        return BB*np.sqrt(np.pi*HH/(2*uu))*(1/2.+uu/HH)*np.exp(-(uu-p0)/HH)

########
# Parse input and prepare
    if z0 < 0:
        print("WARNING: z < 0, interpreting as abs(z)")
    z0 = abs(z0)
    
    if p0 < 0:
        print("WARNING: planet radius p0 < 0, interpreting as abs(p0)")
    p0 = abs(p0)

########
# Unocculted flux
    norm = np.pi/6*(6-2*g1-g2)

########
# Interpolation grid to get regularly spaced source points.
# Choose the r-theta points wisely, i.e. where the gradient is large.
    nr_many = int(round(MAJOR*GRID_NR))
    nr_few = GRID_NR - nr_many
    if z0 < p0+LARGE_EXP*HH:
        boundary = z0+p0+LARGE_EXP*HH
        rr = np.linspace(EPSILON, boundary, nr_many)
        rr = np.hstack((rr, np.linspace(boundary+1/GRID_NR, 1, nr_few)))
        
    elif z0 > 1-p0-LARGE_EXP*HH:
        boundary = 1-p0-LARGE_EXP*HH
        rr = np.linspace(EPSILON,boundary-1/GRID_NR, nr_few)
        rr = np.hstack((rr, np.linspace(boundary, 1, nr_many)))
        
    else:
        lower = z0-p0-LARGE_EXP*HH
        upper = z0+p0+LARGE_EXP*HH
        rr = np.linspace(EPSILON,lower-1/GRID_NR, np.ceil(nr_few/2))
        rr = np.hstack((rr, np.linspace(lower, upper, nr_many)))
        rr = np.hstack((rr, np.linspace(upper+1/GRID_NR, 1, np.floor(nr_few/2))))

    nt_many = int(round(MAJOR*GRID_NT))
    nt_few = GRID_NT - nt_many
    if z0 < p0+LARGE_EXP*HH:
        th = np.linspace(0., np.pi, GRID_NT)
    else:
        boundary = np.pi-np.arcsin((p0+LARGE_EXP*HH)/z0) # Law of sines
        th = np.linspace(0., boundary-1/GRID_NT, nt_few)
        th = np.hstack((th, np.linspace(boundary , np.pi, nt_many)))

    rr_S = np.tile(rr, GRID_NT)
    th_S = np.repeat(th, GRID_NR)
    grid1_S = rr_S*np.cos(th_S)
    grid2_S = rr_S*np.sin(th_S)
    th_S_2d = th_S.reshape((GRID_NT, GRID_NR)) # Help variables for integration
    rr_S_2d = rr_S.reshape((GRID_NT, GRID_NR))[0]

########
# Primary images first
# Define the r-theta points in the image plane
    imRR_S = np.tile(np.linspace(EPSILON,1., STAR_NR), STAR_NT)
    imTH_S = np.repeat(np.linspace(0., np.pi, STAR_NT), STAR_NR) 
    imRR_P = np.tile(np.linspace(p0-EPSILON,p0+LARGE_EXP*HH, PRI_PLA_NR), PRI_PLA_NT)
    imTH_P = np.repeat(np.linspace(0., np.pi, PRI_PLA_NT), PRI_PLA_NR)

    img1_S = imRR_S*np.cos(imTH_S)
    img2_S = imRR_S*np.sin(imTH_S)
    img1_P = imRR_P*np.cos(imTH_P)
    img2_P = imRR_P*np.sin(imTH_P)

    img1_P = np.hstack((img1_S+z0, img1_P)) # Note shift of coordinates +z0 here
    img2_P = np.hstack((img2_S, img2_P))

########
# Help variables
    uu_P = rho(img1_P, img2_P) # Note that uu_P is u*D_OL in Hui & Seager (2002)
    phiu = phi(uu_P)
    u2phiu = u2phi(uu_P)

# Map image positions to source plane
    src1_P = img1_P*(1+phiu)
    src2_P = src1_P*img2_P/img1_P

# Compute magnification
    AA = (1+phiu)*(1+phiu+u2phiu)
    AA = 1/AA

# Screen unwanted source positions 
    WW = np.where(uu_P < p0, 0., 1.) # Planet occultation kernel
    pri= np.where(img1_P*src1_P >= 0.) # Selects primary images 

########
# Carry out the actual integration
    AA2= AA
    AA = AA[pri]*WW[pri]
    src1_P = src1_P[pri]
    src2_P = src2_P[pri]
    AA = griddata(np.vstack((src1_P-z0,src2_P)).T, AA,\
                      np.vstack((grid1_S, grid2_S)).T, method='nearest', fill_value=0)

#    plt.plot(img1_P, img2_P, 'ob')
#    plt.plot(src1_P, src2_P, '.g')
#    plt.plot(grid1_S+z0, grid2_S, '.')
#    temp = np.linspace(0,2*np.pi,10000)
#    plt.plot(p0*np.cos(temp),p0*np.sin(temp))
#    plt.plot(np.cos(temp)+z0,np.sin(temp))
#    plt.plot((p0+HH)*np.cos(temp),(p0+HH)*np.sin(temp))
###    plt.axis([-1,1+z0,-1,1])
#    plt.show()

#    plt.plot(src1_P, src2_P, '.k')
#    plt.tricontourf(src1_P, src2_P, WW*pri*AA2, 30, cmap='summer')
#    plt.show()
#
#    plt.tricontourf(img1_P, img2_P, AA2*WW*pri, cmap='summer')
#    plt.show()
#    plt.tricontourf(src1_P, src2_P, AA2[pri]*WW[pri], cmap='summer')
#    plt.show()
#    
#    plt.tricontourf(grid1_S, grid2_S, AA, cmap='summer')
#    plt.scatter(grid1_S, grid2_S)
#    plt.show()

#    plt.plot(img1_P, src1_P,'.-')
#    plt.show()
    
#    plt.plot(img1_P, AA2)
#    plt.show()
#
#    plt.plot(src1_P, AA2)
#    plt.show()
    
#    plt.plot(WW)
#    plt.plot(pri)
#    plt.plot(AA)
#    plt.plot(AA2)
#    plt.plot(AA2*WW*pri)
#    plt.show()

########
#Integrate using composite Simpson's rule
    integrand = 2*rr_S*II(rr_S)*AA
    integrand = np.where(np.isnan(integrand) | np.isinf(integrand), 0, integrand) #Should not be needed
    integrand = integrand.reshape((GRID_NT, GRID_NR))
#    plt.plot(th_S_2d, integrand)
#    plt.show()
    integrand = simps(integrand, th_S_2d, axis=0)
#    plt.plot(rr_S_2d, integrand)
#    plt.show()
    FF = simps(integrand, rr_S_2d, axis=0)
#    plt.plot(rr_S_2d, integrand)
#    plt.show()

########
# Secondary images
    if caustics:
# Revert to uniform grid because less is known about position of
# sources to secondary images.
        rr = np.linspace(EPSILON,1,GRID_NR)
        th = np.linspace(0, np.pi, GRID_NT)

        rr_S = np.tile(rr, GRID_NT)
        th_S = np.repeat(th, GRID_NR)
        grid1_S = rr_S*np.cos(th_S)
        grid2_S = rr_S*np.sin(th_S)
        th_S_2d = th_S.reshape((GRID_NT, GRID_NR)) # Help variables for integration
        rr_S_2d = rr_S.reshape((GRID_NT, GRID_NR))[0]

        
# Define the r-theta points in the image plane. Note that secondary
# images must be on the far side of the star, and no images can be on
# the upper half of the stellar disk. Therefore, only images close to
# planet limb are necessary.
        imRR_P = np.tile(np.linspace(p0-EPSILON,p0+LARGE_EXP*HH, SEC_PLA_NR), SEC_PLA_NT)
        imTH_P = np.repeat(np.linspace(np.pi, 2*np.pi, SEC_PLA_NT), SEC_PLA_NR)

        img1_P = imRR_P*np.cos(imTH_P)
        img2_P = imRR_P*np.sin(imTH_P)

########
# Help variables
        uu_P = rho(img1_P, img2_P) # Note that uu_P is u*D_OL in Hui & Seager (2002)
        phiu = phi(uu_P)
        u2phiu = u2phi(uu_P)

# Map image positions to source plane
        src1_P = img1_P*(1+phiu)
        src2_P = src1_P*img2_P/img1_P
#        plt.plot(img1_P, src1_P)
#        plt.show()
        
# Compute magnification
        AA = (1+phiu)*(1+phiu+u2phiu)
        AA = 1/AA

# Screen unwanted source positions 
        WW = np.where(uu_P < p0, 0., 1.) # Planet occultation kernel
#        sec1= np.where(AA < 0.) # Only dealing with secondary images here
        sec= np.where(img1_P*src1_P < 0.) # Selects primary images SHOULD BE SAME AS AA > 0
#        print(np.mean(sec[0]),sec[0].size)
#        print(np.mean(sec1[0]),sec1[0].size)
        

########
# Remove 
        AA = AA[sec]*WW[sec]
        src1_P = src1_P[sec]
        src2_P = src2_P[sec]
        AA = griddata(np.vstack((src1_P-z0,src2_P)).T, AA,\
                          np.vstack((grid1_S, grid2_S)).T, method='nearest', fill_value=0)

#        plt.plot(img1_P, img2_P, '.')
#        plt.plot(src1_P, src2_P, '.')
##        plt.plot(grid1_S+z0, grid2_S, '.')
#        temp = np.linspace(0,2*np.pi,10000)
#        plt.plot(p0*np.cos(temp),p0*np.sin(temp))
#        plt.plot(np.cos(temp)+z0,np.sin(temp))
#        plt.plot((p0+HH)*np.cos(temp),(p0+HH)*np.sin(temp))
##        plt.axis([-1,1+z0,-1,1])
#        plt.show()

#        plt.tricontourf(grid1_S, grid2_S, AA, cmap='summer')
#        plt.show()

#        plt.plot(img1_P,WW)
#        plt.plot(img1_P,sec)
#        plt.plot(AA)
#        plt.plot((1+phiu)*(1+phiu+u2phiu))
#        plt.plot(img1_P,AA2)
#        plt.plot(img1_P,AA2*WW*sec)
#        plt.show()

########
#Integrate using composite Simpson's rule
        integrand = 2*rr_S*II(rr_S)*np.abs(AA) # Note abs() here!
        integrand = np.where(np.isnan(integrand) | np.isinf(integrand), 0, integrand) #Should not be needed
        integrand = integrand.reshape((GRID_NT, GRID_NR))
        integrand = simps(integrand, th_S_2d, axis=0)
#        plt.plot(rr_S_2d, integrand)
#        plt.show()
        FF = FF+simps(integrand, rr_S_2d, axis=0)

    return FF/norm



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
    Rstar = 1.253e11
    # Beta Pictoris b
    # Observables
    TT = 1600 # Temperature
    rho0 = 3e-6 # Density at optical depth 1. Pretty much taken this out of nowhere.
    alpha = 1.2 # Refractive index, shouldn't vary too much
    gg = 10**3.8 # Surface gravity
    MM = 11*1.898e30 # Mass
    RR = 1.65*6.9911e+9/Rstar # Radius
    Dls = 9*1.496e13/Rstar # Orbital distance
    
    # Parameters
    HH = kB*TT/(gg*mu*mH)/Rstar
    BB = 2*alpha*rho0*Dls/HH
#    print HH, BB, RR/HH, HH*Rstar
    
#########
## Sun
    #Rstar = 6.957e10
    ## Jupiter
    ## Observables
    #TT = 128 # Temperature
    #rho0 = 3.5e-5 # Density at optical depth 1. Pretty much taken this out of nowhere.
    #alpha = 1.2 # Refractive index, shouldn't vary too much
    #gg = 2479 # Surface gravity
    #MM = 1.898e30 # Mass
    #RR = 7.1e+9/Rstar # Radius
    #Dls = 5.2*1.496e13/Rstar # Orbital distance
    #
    ## Parameters
    #HH = kB*TT/(gg*mu*mH)/Rstar
    #BB = 2*alpha*rho0*Dls/HH
    #print HH, BB, RR/HH, HH*Rstar
    
########
# Orbital parameters
    t=np.linspace(0.,1.3,2)
    z=np.sqrt(t**2+(2*0.084)**2) # hui02
#    z=t

#    exec( "\n".join( sys.argv[1:] ))  # run this.py npt= dim= ...
    
########
# Compute the light curve
    x=np.zeros(z.size)
    for i in range(0,z.size):
        print(i, z[i]) # g1, g2, B, H, p0
        x[i] = refraction_transit(z[i], 0.35, 0.225, 40.3, 0.084/117.3, 0.084) # Should be Hui & Seager (2002)
#        x[i] = refraction_transit(z[i], 0.35, 0.225, BB, HH, RR)
    
    x=np.concatenate((x[1:][::-1],x))
    z=np.concatenate((-z[1:][::-1],z))
    t=np.concatenate((-t[1:][::-1],t))
    np.savetxt('/Users/silver/Dropbox/sci/project_exo/models/hui02_absref.dat', np.c_[t,x])
    plt.plot(t,x-1,'ok')
    plt.plot(t,x-1)
    plt.show()
