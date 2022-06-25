import pdb
import sys

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from exo_trans_refrac_lc import refraction_transit
from exo_trans_occult3 import occultquad
from exo_sap_hlp import poor_mans_bound

########
# Help functions
def print_pars():
    print("Refractive coefficient, alpha:", alpha)
    print("Temperature:", TT)
    print("Mean molecular weight:", mu)
    print("Mass:", MM)
    print("Radius:", RR)
    print("Orbital distance:", Dls)
    print("Surface gravity:", gg)
    print("Density, rho0:", rho0)
    print("Scale height, H:", HH*Rstar)
    print("R0/H:", RR/HH)
    print("B parameter:", BB)
    print("Impact parameter:", impact)
    print("Orbital period:", period/60/24)
    print("Surface pressure (cgs):", rho0*gg*HH*Rstar)
    print("Surface pressure (bar):", rho0*gg*HH*Rstar*1e-6)

def cauchy(AA, BB, lam):
    return AA*(1+BB/lam**2)

def get_z(tt, period, impact, dist):
    incl = np.rad2deg(np.arccos(impact/dist))
    cos = np.cos(np.deg2rad(incl))*np.cos(2*np.pi*tt/period)
    sin = np.sin(2*np.pi*tt/period)
    return dist*np.sqrt(sin**2+cos**2)

def fit_lc(time, fit_prad, fit_incl, fit_dor, f1, f2): 
    fit_dor = np.abs(fit_dor)
    high_res = np.arange(np.min(time)-CADENCE, np.max(time)+CADENCE)
    impact = fit_dor*np.cos(np.deg2rad(fit_incl))
    if HUI:
        zz = np.sqrt((high_res/233.66)**2+impact**2)
    else:
        zz = get_z(high_res, period, impact, fit_dor)

    #Far side of the star
#    backside = np.where(abs(2*np.pi*high_res/period)>np.pi/2)[0]
#    if np.any(backside):
#        zz[backside] = 10+fit_prad
    
    if GFIT:
        h1 = poor_mans_bound(f1, GFIT)
        h2 = poor_mans_bound(f2, GFIT)
    else:
        h1 = h2 = 1.
        
    quad, lin = occultquad(zz, h1*g1, h2*g2, fit_prad)

    global mod
    mod = np.convolve(quad-1, np.ones(CADENCE)/CADENCE,'same')
    mod = griddata(high_res, mod, time, method='linear', fill_value=0)
    return mod

def fix_par(par):
    par[3] = poor_mans_bound(par[3], GFIT)
    par[4] = poor_mans_bound(par[4], GFIT)
    return par

########
# Constants, cgs
AU = 1.496e13 # Astronomical unit
kB = 1.380658e-16 # Boltzmann constant
mH = 1.6733e-24 # Hydrogen mass (Atomic mass unit)
GG = 6.6743e-8 # Gravitational constant
LAM = 6500e-8 # Wavelength
LAM = 45000e-8 # Wavelength
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
Mstar = 1.989e33 # Mass of the star
g1 = 0.37 # Fiducial parameters close to Kepler population mean/median
g2 = 0.27

# Trappist 1
#Rstar = 0.117*6.957e10 # Radius of the star
#Mstar = 0.0802*1.989e33 # Mass of the star
#g1 = 0.
#g2 = 0.

########
# Best-case parameters
#ID = "best_case"
#TT = 4*150. # Temperature of relevant part of atmosphere
#mu = 2. # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = rhoH2
#alpha = cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 1.898e30 # Mass of planet
#RR = 7.1e+9 # Radius
#Dls = 5.2*AU # Orbital distance
#tt=np.linspace(0., 30*60., 30*60+1) # Time range [min]

########
# Jupiter parameters
#ID = "jupiter"
#TT = 150 # Temperature of relevant part of atmosphere
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 1.898e30 # Mass of planet
#RR = 7.1e+9 # Radius
#Dls = 5.2*AU # Orbital distance
#tt=np.linspace(0.,30*60.,1801) # Time range [min]

#ID = "jupiter2"
#TT = 150 # Temperature of relevant part of atmosphere
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 0.5*1.898e30 # Mass of planet
#RR = 0.5*7.1e+9 # Radius
#Dls = 5.2/4*AU # Orbital distance
#tt=np.linspace(0.,30*60.,1801) # Time range [min]
#
#ID = "jupiter3"
#TT = 300 # Temperature of relevant part of atmosphere
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 2*1.898e30 # Mass of planet
#RR = 7.1e+9 # Radius
#Dls = 5.2*AU # Orbital distance
#tt=np.linspace(0.,30*60.,1801) # Time range [min]
#
#########
## 1 yr Jovian
#ID = "1yr_jovian"
#TT = 150*np.sqrt(5.2) # Temperature (Scaled from Jupiter based on effective equilibrium temperature, i.e. insolation = -blackbody. This method takes reflectivity of Jupiter into account.)
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 1.898e30 # Mass of planet, 1.898e30 g = 1 M_Jupiter
#RR = 7.1e+9 # Radius, 7.1e+9 cm = 1 R_Jupiter. ASSUMING THIS TO BE SAME AS FOR JUPITER, I.E. AVERAGE DENSITY INDEPENDENT OF SURFACE TEMPERATURE
#Dls = AU # Orbital distance
#tt=np.linspace(0.,10*60.,601) # Time range [min]

#ID = "1yr_jovian2"
#MM /= 2
#RR /= 2
#Dls /= 4

#ID = "1yr_jovian3"
#MM *= 2
#TT *= 2

########
# Earth
# These values give a density of 0.9e-3 g cm-3, compared to air; 1.2e-3 g cm-3 at STP.
# So, most of the modelling is probably pretty good.
#ID = "earth_cap"
#TT = 255 # Temperature (Equilibrium temperature, more appropriate for the relevant part of the atmosphere)
#mu = 0.78*28+0.22*32 # Mean molecular wight, N2+O2, 22% O2 by number
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 5.972e27 # Mass of planet
#RR = 6.371e8 # Radius
#Dls = AU # Orbital distance
#tt=np.linspace(0.,12*60.,721) # Time range [min]

########
# 80 day super-Earth
#ID = "80d_super_earth_cap"
#TT = 255/np.sqrt((80/365.26)**(2/3.)) # Temperature (Scaled from Earth.)
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 3.9*5.972e27
#RR = 1.5*6.371e8
#Dls = (80/365.26)**(2/3.)*AU # Orbital distance
#tt=np.linspace(0.,6*60.,361) # Time range [min]

########
# 20 day Jovian
#ID = "20d_jovian_air2"
#TT = 150*np.sqrt(5.2/(20/365.26)**(2/3.)) # Temperature (Scaled from Jupiter based on effective equilibrium temperature, i.e. insolation = -blackbody. This method takes reflectivity of Jupiter into account.)
#mu = 0.863636*2+0.136364*4 # Mean molecular wight, H2+He, 24% He by mass
#rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
#alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp
#mu = 0.78*28+0.22*32 # Mean molecular wight, N2+O2, 22% O2 by number
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
#MM = 1.898e30 # Mass of planet, 1.898e30 g = 1 M_Jupiter
#RR = 7.1e+9 # Radius, 7.1e+9 cm = 1 R_Jupiter. ASSUMING THIS TO BE SAME AS FOR JUPITER, I.E. AVERAGE DENSITY INDEPENDENT OF SURFACE TEMPERATURE
#Dls = (20/365.26)**(2/3.)*AU # Orbital distance
#tt=np.linspace(0.,3*60.,181) # Time range [min]

########
# Trappist-1h
ID = "trappist-1h"
TT = 168
mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp 
MM = 0.086*5.972e27
RR = 0.755*6.371e8
Dls = np.sqrt(1/0.000524)*63e-3*AU # Orbital distance
#tt=np.linspace(0.,3*60.,181) # Time range [min]
tt=np.linspace(0.,14*60.,841) # Time range [min]

########
# Trappist-1g
#ID = "trappist-1g"
#TT = 199
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 0.566*5.972e27
#RR = 1.127*6.371e8
#Dls = 45.1e-3*AU # Orbital distance
#tt=np.linspace(0.,3*60.,181) # Time range [min]

########
# Trappist-1f
#ID = "trappist-1f"
#TT = 219
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 0.36*5.972e27
#RR = 1.045*6.371e8
#Dls = 37.1e-3*AU # Orbital distance
#tt=np.linspace(0.,3*60.,181) # Time range [min]

########
# Trappist-1e
#ID = "trappist-1e_hirs"
#TT = 251.3
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 0.24*5.972e27
#RR = 0.918*6.371e8
#Dls = 28.17e-3*AU # Orbital distance
#tt=np.linspace(0.,60.,301) # Time range [min]

########
# Trappist-1d
#ID = "trappist-1d_hirs"
#TT = 288
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 0.33*5.972e27
#RR = 0.772*6.371e8
#Dls = 21.44e-3*AU # Orbital distance
#tt=np.linspace(0.,60.,301) # Time range [min]

########
# Trappist-1c
#ID = "trappist-1c_hirs"
#TT = 341.9
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 1.63*5.972e27
#RR = 1.056*6.371e8
#Dls = 15.21e-3*AU # Orbital distance
#tt=np.linspace(0.,1*40.,401) # Time range [min]

########
# Trappist-1b
#ID = "trappist-1b_hirs"
#TT = 400.1
#mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
#rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
#alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp
#MM = 0.79*5.972e27
#RR = 1.086*6.371e8
#Dls = 11.11e-3*AU # Orbital distance
#tt=np.linspace(0.,30.,301) # Time range [min]


########
# Refraction parameters
gg = GG*MM/RR**2 # Surface gravity
HH = kB*TT/(gg*mu*mH) # Scale height
rho0 = mu*mH/(sig*np.sqrt(2*np.pi*RR*HH)) # Eq. (19) in Hui & Seager (2002)
if rho0*gg*HH*1e-6 > 1.01325:
    print(rho0*gg*HH*1e-6)
    rho0 = 1.01325/(gg*HH*1e-6)
BB = 2*alpha*rho0*Dls/HH # Eq. (9) in Hui & Seager (2002)
HH /= Rstar # Change length unit to stellar radii beyond this line
RR /= Rstar
Dls /= Rstar
########
# Orbital parameters
HUI = False
impact = 0.5
period = 2/60.*np.pi*np.sqrt((Dls*Rstar)**3/(GG*Mstar))
incl = np.rad2deg(np.arccos(impact/Dls))
guess = np.array([RR, incl, Dls, 0., 0.])

zz=get_z(tt, period, impact, Dls)
print_pars()
print(1-np.sqrt(np.pi/(2*RR/HH))*BB)

# Compute minimum range
#keep = zz > 1+RR
#zz = zz[keep]
#tt = tt[keep]
#keep = tt < np.min(tt)+600
#zz = zz[keep]
#tt = tt[keep]

########
# Fiducial model of Hui & Seager (2002)
#print("OVERRIDE with fiducial model of Hui & Seager (2002)!")
#ID = "hui02"
#g1 = 0.35
#g2 = 0.225
#BB = 40.3
#RR = 0.084
#HH = RR/117.3
#guess = np.array([RR, 89.95, 250, 0., 0.])
#impact = 2*RR
#tt=np.linspace(0,5*60,301)
#zz=np.sqrt((tt/233.66)**2+impact**2) # hui02
#HUI = True
# OVERRIDE
########

########
refrac = refraction_transit(zz, g1, g2, BB, HH, RR)-1
refrac = np.concatenate((refrac[1:][::-1], refrac))
zz = np.concatenate((zz[1:][::-1],zz))
tt = np.concatenate((-tt[1:][::-1],tt))
plain = occultquad(zz, g1, g2, RR)[0]-1 # No refraction/reference
# TEMPORARY
#dat = np.loadtxt("C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\80d_super_earth.dat")
#refrac = dat[:,3]
# TEMPORARY

CADENCE = 1
GFIT = 0
curve_par, covar = curve_fit(fit_lc, tt, refrac, guess, maxfev=20000)
fix_ex_mod = mod
fix_ex_mod_par = fix_par(curve_par)
print(curve_par[0],curve_par[1],curve_par[2],curve_par[3],curve_par[4])

GFIT = 0.1
GFIT = 2.*GFIT/np.pi # This is because arctans are used as boundaries
curve_par, covar = curve_fit(fit_lc, tt, refrac, guess, maxfev=20000)
ldc_ex_mod = mod
ldc_ex_mod_par = fix_par(curve_par)
print(curve_par[0],curve_par[1],curve_par[2],curve_par[3],curve_par[4])

CADENCE = 29
conv_refrac = np.convolve(refrac, np.ones(CADENCE)/CADENCE,'same')
conv_plain = np.convolve(plain, np.ones(CADENCE)/CADENCE,'same')

curve_par, covar = curve_fit(fit_lc, tt, conv_refrac, guess, maxfev=20000)
ldc_conv_mod = mod
ldc_conv_mod_par = fix_par(curve_par)
print(curve_par[0],curve_par[1],curve_par[2],curve_par[3],curve_par[4])

GFIT = 0
curve_par, covar = curve_fit(fit_lc, tt, conv_refrac, guess, maxfev=20000)
fix_conv_mod = mod
fix_conv_mod_par = fix_par(curve_par)
print(curve_par[0],curve_par[1],curve_par[2],curve_par[3],curve_par[4])

output = np.c_[tt, zz, plain, refrac, fix_ex_mod, ldc_ex_mod, conv_plain, conv_refrac, fix_conv_mod, ldc_conv_mod]
output_pars = np.c_[fix_par(guess), fix_ex_mod_par, ldc_ex_mod_par, fix_conv_mod_par, ldc_conv_mod_par].T

MAC = True
if MAC:
    np.savetxt('/Users/silver/Dropbox/sci/pro_exo/models/' + ID + "_fit_par.dat", output_pars)
    np.savetxt('/Users/silver/Dropbox/sci/pro_exo/models/' + ID + ".dat", output)             
else:
    np.savetxt("C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\" + ID + "_fit_par.dat", output_pars)
    np.savetxt("C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\" + ID + ".dat", output)
