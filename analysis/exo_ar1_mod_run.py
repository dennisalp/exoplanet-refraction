import pdb

import numpy as np

from exo_trans_refrac_lc import refraction_transit
from exo_trans_occult3 import occultquad
from exo_sap_hlp import poor_mans_bound



################################################################
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
    print("")

def cauchy(AA, BB, lam):
    return AA*(1+BB/lam**2)

def get_z(tt, period, impact, dist):
    incl = np.rad2deg(np.arccos(impact/dist))
    cos = np.cos(np.deg2rad(incl))*np.cos(2*np.pi*tt/period)
    sin = np.sin(2*np.pi*tt/period)
    return dist*np.sqrt(sin**2+cos**2)

def dry_run(zz):
    return np.where(zz > 1.+RR, 0., -0.01)



################################################################
# Constants, cgs
AU = 1.496e13 # Astronomical unit
kB = 1.380658e-16 # Boltzmann constant
mH = 1.6733e-24 # Hydrogen mass (Atomic mass unit)
GG = 6.6743e-8 # Gravitational constant

# 273.15 K, 1 atm densities. These are used for the refraction coefficient alpha.
rhoH2 = 8.988e-5
rhoHe = 1.786e-4
rhoN2 = 1.251e-3
rhoO2 = 1.429e-3

# Computational
MAC = True

########
# Star parameters
# The Sun seems to be quite representative for the Kepler population
Rstar = 6.957e10 # Radius of the star
Mstar = 1.989e33 # Mass of the star
g1 = 0.37 # Fiducial parameters close to Kepler population mean/median
g2 = 0.27

# Trappist 1
Rstar = 0.117*6.957e10 # Radius of the star
Mstar = 0.0802*1.989e33 # Mass of the star
g1 = 0.
g2 = 0.

################################################################
# Allocate
NP = 6
NN = 12
sav_PH = np.zeros((NP, NN, 2))
sav_BC = np.zeros((NP, NN, 2))

# The grid for parameter running
# jupiter
t_int = 600 # min
OVRSMP = 1 # must be int as it is used in linspace
run = np.array([np.logspace(np.log10( 0.4), np.log10( 4), NN),
                np.logspace(np.log10( 0.5), np.log10(20), NN),
                np.logspace(np.log10( 0.1), np.log10(20), NN),
                np.logspace(np.log10( 0.5), np.log10( 3), NN),
                np.logspace(np.log10( 0.1), np.log10(20), NN),
                np.logspace(np.log10(0.75), np.log10(20), NN)])

# 80d_super_earth
t_int = 60 # min
OVRSMP = 20 # must be int as it is used in linspace
run = np.array([np.logspace(np.log10( 0.4), np.log10( 4), NN),
                np.logspace(np.log10( 0.2), np.log10( 5), NN),
                np.logspace(np.log10( 0.2), np.log10(20), NN),
                np.logspace(np.log10(0.25), np.log10( 5), NN),
                np.logspace(np.log10( 0.1), np.log10(20), NN),
                np.logspace(np.log10( 0.1), np.log10( 2), NN)])

# best_case_jupiter
#t_int = 60 # min
#OVRSMP = 1 # must be int as it is used in linspace
#run = np.array([np.logspace(np.log10( 0.4), np.log10( 4), NN),
#                np.logspace(np.log10( 0.5), np.log10(20), NN),
#                np.logspace(np.log10( 0.1), np.log10(20), NN),
#                np.logspace(np.log10( 0.5), np.log10( 3), NN),
#                np.logspace(np.log10( 0.1), np.log10(20), NN),
#                np.logspace(np.log10(0.75), np.log10(20), NN)])

# best_case_earth
t_int = 600 # min
OVRSMP = 1 # must be int as it is used in linspace
run = np.array([np.logspace(np.log10( 0.4), np.log10( 4), NN),
                np.logspace(np.log10( 0.2), np.log10( 5), NN),
                np.logspace(np.log10( 0.2), np.log10(20), NN),
                np.logspace(np.log10(0.25), np.log10( 5), NN),
                np.logspace(np.log10( 0.1), np.log10( 6), NN),
                np.logspace(np.log10( 0.1), np.log10( 2), NN)])

# trappist-1h
t_int = 10 # min
OVRSMP = 3 # must be int as it is used in linspace
run = np.array([np.logspace(np.log10( 0.4), np.log10( 4), NN),
                np.logspace(np.log10( 0.2), np.log10( 5), NN),
                np.logspace(np.log10( 0.2), np.log10(20), NN),
                np.logspace(np.log10(0.25), np.log10( 5), NN),
                np.logspace(np.log10( 0.1), np.log10(20), NN),
                np.logspace(np.log10( 0.1), np.log10( 2), NN)])

################################################################
for pp in range(0, NP):
    for ii in range(0, NN):
        # SPECIAL CASE
        LAM = 6500e-8 # Wavelength
        LAM = 45000e-8 # Spitzer
        if pp == 0:
            LAM *= run[pp, ii]
            
        ########
        # Jupiter parameters
        ID = "jupiter"
        TT = 150 # Temperature of relevant part of atmosphere
        mu = 0.863636*2+0.136364*4 # Mean molecular weight, H2+He, 24% He by mass
        rho_stp = 0.863636*rhoH2+0.136364*rhoHe # Help variable
        alpha = 0.863636*cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp+0.136364*cauchy(3.48e-5, 2.3e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        MM = 1.898e30 # Mass of planet
        RR = 7.1e+9 # Radius
        Dls = 5.2*AU # Orbital distance
        tt=np.linspace(0., 1000*24*60., 1000*24*60+1) # Time range [min]
    
        ########
        # 80 day super-Earth
        ID = "80d_super_earth"
        TT = 255/np.sqrt((80/365.26)**(2/3.)) # Temperature (Scaled from Earth.)
        mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
        rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        MM = 3.9*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
        RR = 1.5*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
        Dls = (80/365.26)**(2/3.)*AU # Orbital distance
        tt=np.linspace(0., 1000*24*60., OVRSMP*(1000*24*60)+1) # Time range [min]

        ########
        # Best-case parameters
        #ID = "best_case_jupiter"
        #TT = 4*150. # Temperature of relevant part of atmosphere
        #mu = 2. # Mean molecular wight, H2+He, 24% He by mass
        #rho_stp = rhoH2
        #alpha = cauchy(1.36e-4, 7.7e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        #MM = 1.898e30 # Mass of planet
        #RR = 7.1e+9 # Radius
        #Dls = 5.2*AU # Orbital distance
        #tt=np.linspace(0., 1000*24*60., 1000*24*60+1) # Time range [min]

        ID = "best_case_earth"
        TT = 255 # Temperature (Equilibrium temperature, more appropriate for the relevant part of the atmosphere)
        mu = 0.78*28+0.22*32 # Mean molecular wight, N2+O2, 22% O2 by number
        rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        MM = 5.972e27 # Mass of planet
        RR = 6.371e8 # Radius
        Dls = AU # Orbital distance
        tt=np.linspace(0.,16*60., OVRSMP*(16*60)+1) # Time range [min]

        ########
        # Trappist-1h
        ID = "trappist-1h"
        TT = 168
        mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
        rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        MM = 0.086*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
        RR = 0.755*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
        Dls = 63e-3*AU # Orbital distance
        tt=np.linspace(0., 1000*24*60., OVRSMP*(1000*24*60)+1) # Time range [min]

        ########
        # Trappist-1g
        #ID = "trappist-1g"
        #TT = 199
        #mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
        #rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        #alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        #MM = 0.566*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
        #RR = 1.127*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
        #Dls = 45.1e-3*AU # Orbital distance
        #tt=np.linspace(0., 1000*24*60., OVRSMP*(1000*24*60)+1) # Time range [min]

        ########
        # Trappist-1f
        #ID = "trappist-1f"
        #TT = 219
        #mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
        #rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        #alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        #MM = 0.36*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
        #RR = 1.045*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
        #Dls = 37.1e-3*AU # Orbital distance
        #tt=np.linspace(0., 1000*24*60., OVRSMP*(1000*24*60)+1) # Time range [min]

        ########
        # Trappist-1e
        #ID = "trappist-1e"
        #TT = 251.3
        #mu = 0.78*28+0.22*32 # Mean molecular wight, ASSUMING EARTH ATMOSPHERE
        #rho_stp = 0.78*rhoN2+0.22*rhoO2 # Help variable
        #alpha = 0.78*cauchy(2.919e-4, 7.7e-11, LAM)/rho_stp+0.22*cauchy(2.663e-4, 5.07e-11, LAM)/rho_stp # Refractive index, cf. Table 1 in Hui & Seager (2002), Cauchy's formula
        #MM = 0.24*5.972e27 # Mass of planet, 2 M_Earth and 1.5 R_Earth is close to MgSiO3
        #RR = 0.918*6.371e8 # Radius, http://exoplanetarchive.ipac.caltech.edu/exoplanetplots/exo_massradius.png
        #Dls = 28.17e-3*AU # Orbital distance
        #tt=np.linspace(0., 1000*24*60., OVRSMP*(1000*24*60)+1) # Time range [min]
        
        # Run the parameters
        if pp == 1:
            TT *= run[pp, ii]
        elif pp == 2:
            MM *= run[pp, ii]
        elif pp == 3:
            RR *= run[pp, ii]
        elif pp == 4:
            Dls *= run[pp, ii]
        elif pp == 5:
            mu *= run[pp, ii]
            alpha /= run[pp, ii] # Cause rho_stp goes into alpha, and rho_stp should scale with mu
        
        # Refraction parameters
        sig = 1e-27*(5000e-8/LAM)**4 # Rayleigh scattering, eq. (20) in Hui & Seager (2002)
        gg = GG*MM/RR**2 # Surface gravity
        HH = kB*TT/(gg*mu*mH) # Scale height
        rho0 = mu*mH/(sig*np.sqrt(2*np.pi*RR*HH))
        if rho0*gg*HH*1e-6 > 19999.01325:
            print(rho0*gg*HH*1e-6)
            rho0 = 1.01325/(gg*HH*1e-6)
        BB = 2*alpha*rho0*Dls/HH # Eq. (9) in Hui & Seager (2002)
        HH /= Rstar # Change length unit to stellar radii beyond this line
        RR /= Rstar
        Dls /= Rstar

        ########
        # Orbital parameters
        impact = 0.5
        period = 2/60.*np.pi*np.sqrt((Dls*Rstar)**3/(GG*Mstar))
        incl = np.rad2deg(np.arccos(impact/Dls))
        guess = np.array([RR, incl, Dls, 0., 0.])
        zz=get_z(tt, period, impact, Dls)
#        print_pars()
        
        # Compute minimum range
        keep = zz > 1+RR+3*HH
        zz = zz[keep]
        tt = tt[keep]
        keep = tt < np.min(tt)+t_int
        zz = zz[keep]
        tt = tt[keep]
        
        ########
        #refrac = dry_run(zz)
        refrac = refraction_transit(zz, g1, g2, BB, HH, RR)-1

        sav_BC[pp, ii, :] = [BB, RR/HH]
        sav_PH[pp, ii, :] = [run[pp, ii], np.sum(refrac)/1e-6/(t_int*OVRSMP)]
        print(pp, ii, sav_PH[pp, ii, 0], sav_PH[pp, ii, 1], np.amax(refrac)/1e-6, np.amin(refrac)/1e-6, "Surface pressure (bar):", rho0*gg*HH*Rstar*1e-6)
        output = np.c_[tt, zz, refrac]
#        import matplotlib.pyplot as plt
#        plt.plot(tt, refrac/1e-6)
#        plt.show()
#        pdb.set_trace()
        if MAC:
            np.savetxt('/Users/silver/Dropbox/sci/pro_exo/models/run/' + ID + '/' + ID + '_' + str(pp) + '_' + str(ii) + '.dat', output)
        else:
            np.savetxt('C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\run\\' + ID + '\\' + ID + '_' + str(pp) + '_' + str(ii) + '.dat', output)

if MAC:
    np.save('/Users/silver/Dropbox/sci/pro_exo/models/run/' + ID + '/' + ID + '_bc', sav_BC)
    np.save('/Users/silver/Dropbox/sci/pro_exo/models/run/' + ID + '/' + ID + '_ph', sav_PH)
else:
    np.save('C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\run\\' + ID + '\\' + ID + '_bc', sav_BC)
    np.save('C:\\Users\\dalp\\Dropbox\\sci\\pro_exo\\models\\run\\' + ID + '\\' + ID + '_ph', sav_PH)
