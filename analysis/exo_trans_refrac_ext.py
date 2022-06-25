################################################################
# Computes light curve of external refraction
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

################################################################
# Help functions
def print_pars():
    print("Refractive coefficient, alpha:", alpha)
    print("Temperature:", TT)
    print("Mean molecular weight:", mu)
    print("Mass:", MM)
    print("Radius:", RR)
    print("Observer-lens distance:", DOL)    
    print("Lens-source distance:", DLS)
    print("Surface gravity:", gg)
    print("Density, rho0:", rho0)
    print("Scale height, H:", HH)
    print("R0/H:", RR/HH)
    print("B parameter:", BB)
    print("Impact parameter:", impact)

def cauchy(AA, BB, lam):
    return AA*(1+BB/lam**2)

################################################################
# Main part, computes the light curve
def refract_simps(zin, g1, g2, BB, HH, p0):
    def newrap(func, dfunc, x0):
        x1 = x0 + 2*TOL
        func1 = np.zeros(zz.size)
        while np.max(np.abs(x0 - x1)) > TOL:
            func1 = func(x1)
            x0 = x1
            x1 = x0 - func1/dphi(x0)
        return x1

    def phi(xx):
        uu = np.abs(xx)
        res = zz-xx*(1-BB*np.sqrt(np.pi*HH/(2*uu))*np.exp(-(uu-p0)/HH))
#        print("G")
#        if np.min(xx) < 0:
#            pdb.set_trace()
        return res

    def dphi(xx):
        uu = np.abs(xx)
        dudx = xx/uu
        expu = np.exp(-(uu-p0)/HH)
        temp = np.sqrt(np.pi/2)*BB*(HH+2*uu)*expu/(2*np.sqrt(HH*uu**3))*dudx
        res = -1+BB*np.sqrt(np.pi*HH/(2*uu))*expu-xx*temp
        return res

    def gen_guess(signs):
        if signs == 1:
            guess = np.where(zz < p0+LARGE_EXP*HH, p0+TOL+zz, zz-TOL) #Adding TOL here makes phi run 20x faster
        elif signs == -1:
            guess = -p0-LARGE_EXP*HH*np.ones(zz.size)
        func  = phi(guess)
        left = func*signs < 0
        while np.sum(left) > 0: #Find good guess by stepnp.ping images into the lens
            if signs == 1:
                guess[left] = (guess[left]-zz[left])*0.99+zz[left]
            elif signs == -1:
                guess[left] = 0.999*guess[left]
            func = phi(guess)
            left = func*signs < 0

        return guess

#########
#Stuff
    TOL = 1e-12
    LARGE_EXP = 24.
    zlim = min(1.+LARGE_EXP*HH+p0, np.max(zin))
    zz = np.linspace(TOL, np.max(zin), zin.size)
    p0 = np.abs(p0)

#########
#Images
    guess = gen_guess(1)
    uu = newrap(phi, dphi, guess.copy())
    
    expu = np.exp(-(uu-p0)/HH)
    Aphi = -BB*np.sqrt(np.pi*HH/(2*uu))*expu
    u2phi= BB*np.sqrt(np.pi/2)*np.sqrt(HH/uu)*(1/2.+uu/HH)*expu
           
    A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
    A   = 1/A
    W   = np.where(uu > p0, 1., 0.)
    FF = A*W

#########
#Caustics
    if 1-np.sqrt(np.pi/2)*np.sqrt(HH/p0)*BB < 0: 
        guess = gen_guess(-1)
        xx = newrap(phi, dphi, guess.copy())
        uu = np.abs(xx)
        plt.plot(zz, xx)
        plt.show()
        expu = np.exp(-(uu-p0)/HH)
        Aphi = -BB*np.sqrt(np.pi*HH/(2*uu))*expu
        u2phi= BB*np.sqrt(np.pi/2)*np.sqrt(HH/uu)*(1/2.+uu/HH)*expu

        A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
        A   = 1/A
        W   = np.where(uu > p0, 1., 0.)
        FF += np.abs(A)*W

    FF = griddata(zz, FF, np.abs(zin), method='linear', fill_value=1)
    return FF

################################################################
# Constants, cgs
AU = 1.496e13 # Astronomical unit
pc = 3.086e18 # parsec
kB = 1.380658e-16 # Boltzmann constant
mH = 1.6733e-24 # Hydrogen mass (Atomic mass unit)
GG = 6.6743e-8 # Gravitational constant
LAM = 6500e-8 # Wavelength
sig = 1e-27*(5000e-8/LAM)**4 # Rayleigh scattering, eq. (20) in Hui & Seager (2002)

# 273.15 K, 1 atm densities. These are used for the refraction coefficient alpha.
rhoH2 = 8.988e-5
rhoHe = 1.786e-4
rhoN2 = 1.251e-3
rhoO2 = 1.429e-3

########
# Star parameters
# The Sun seems to be quite representative for the Kepler population
Rstar = 6.957e10 # Radius of the star
g1 = 0.37 # Fiducial parameters close to Kepler population mean/median
g2 = 0.27

########
# Jupiter parameters
ID = "jupiter"
period = 4332.59*24*60
TT = 150 # Temperature of relevant part of atmosphere
mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
MM = 1.898e30 # Mass of planet
RR = 7.1e+9 # Radius
tt=np.linspace(0.,30*60.,1801) # Time range [min]

########
# 1 yr Jovian
#ID = "1yr_jovian"
#period = 365*24*60
#TT = 150*np.sqrt(5.2) # Temperature (Scaled from Jupiter based on effective equilibrium temperature, i.e. insolation = -blackbody. This method takes reflectivity of Jupiter into account.)
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 1.898e30 # Mass of planet, 1.898e30 g = 1 M_Jupiter
#RR = 7.1e+9 # Radius, 7.1e+9 cm = 1 R_Jupiter. ASSUMING THIS TO BE SAME AS FOR JUPITER, I.E. AVERAGE DENSITY INDEPENDENT OF SURFACE TEMPERATURE
#tt=np.linspace(0.,10*60.,601) # Time range [min]

########
# Earth
# These values give a density of 0.9e-3 g cm-3, compared to air; 1.2e-3 g cm-3 at STP.
# So, most of the modelling is probably pretty good.
#ID = "earth"
#period = 365.256*24*60
#TT = 255 # Temperature (Equilibrium temperature, more appropriate for the relevant part of the atmosphere)
#mu = 0.78*14+0.22*16 # Mean molecular wight, N2+O2, 22% O2 by number
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 5.972e27 # Mass of planet
#RR = 6.371e8 # Radius
#tt=np.linspace(0.,12*60.,721) # Time range [min]

########
# 80 day super-Earth
#ID = "80d_super_earth"
#period = 80*24*60
#TT = 255/np.sqrt((80/365.26)**(2/3.)) # Temperature (Scaled from Earth.)
#mu = 0.78*14+0.22*16 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 2*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
#RR = 1.5*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
#tt=np.linspace(0.,6*60.,361) # Time range [min]

########
# 20 day Jovian
#ID = "20d_jovian"
#period = 20*24*60
#TT = 150*np.sqrt(5.2/(20/365.26)**(2/3.)) # Temperature (Scaled from Jupiter based on effective equilibrium temperature, i.e. insolation = -blackbody. This method takes reflectivity of Jupiter into account.)
#mu = 2 # Mean molecular wight
#rho_stp = rhoH2 # Help variable
#alpha = cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp
#MM = 1.898e30 # Mass of planet, 1.898e30 g = 1 M_Jupiter
#RR = 7.1e+9 # Radius, 7.1e+9 cm = 1 R_Jupiter. ASSUMING THIS TO BE SAME AS FOR JUPITER, I.E. AVERAGE DENSITY INDEPENDENT OF SURFACE TEMPERATURE
#tt=np.linspace(0.,3*60.,181) # Time range [min]

########
# Geometrical/spatial arrangement
DOL = 40*pc
DLS = 40*pc
impact = 0.13
zz = np.linspace(0,9,1001)
zz = np.sqrt(zz**2+impact**2)

########
# Refraction parameters
gg = GG*MM/RR**2 # Surface gravity
HH = kB*TT/(gg*mu*mH) # Scale height
rho0 = mu*mH/(sig*np.sqrt(2*np.pi*RR*HH)) # Eq. (19) in Hui & Seager (2002)
BB = 2*alpha*rho0/HH*DOL*DLS/(DOL+DLS) # Eq. (9) in Hui & Seager (2002)
BB = 2*alpha*rho0/HH*5*AU # Eq. (9) in Hui & Seager (2002)
print_pars()

################################################################
# Do computations beyond this point
FF = refract_simps(zz, 0.35, 0.225, BB, HH/Rstar, RR/Rstar)
#np.savetxt('/home/dalp/Dropbox/astrophysics/project_exoplanets/models/ext.dat', np.c_[z,x])
plt.semilogy(zz, FF-1)
plt.show()
