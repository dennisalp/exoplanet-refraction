#Python 2.X Dennis Alp 2017-01-22
#
#It seems to leak memory, slowly. Takes Kepler SAP light curves and
#chop-stack-fits them.  Fitting is performed using Levenberg-Marquardt
#algorithm from scipy.optimize.fit_curve.

import os
import sys
import pdb
from datetime import datetime
from glob import glob

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import sem

from exo_sap_cls import *
from exo_sap_plt import *
from exo_sap_hlp import *
from exo_trans_occultquad import occultquad

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

########
#User input
BATCH_ID = sys.argv[1] #Unique ID to avoid conflict if several scripts run simultaneously. 
PLT_FMT = '.k' #Choose marker style (format)
SPARSE = 500. #Number of points in plots to improve performance
CADENCE = 0.02043359821692 #In days. 0.02043359821692 for LC
CHOP_WIDTH = 5. #In transit durations
NORM_PAD = 1.+1./6. # In transit durations
NORM_PAD = 2. # In transit durations
PTS_MIN = [10,9,10] # Minimum number of points [before, in, after] transit
PTS_MIN = [8,14,8] # Minimum number of points [before, in, after] transit
GFIT = 0.1 #Fit gamma and how much, arctan/(pi/2)*GFIT
MOD_RES = CADENCE/301. #Resolution when computing model, in units of minutes. Make odd multiple of cadence preferable, or discrete convolution (prevents an off-by-one shift).
TOL = 0.05 #Relative tolerance when comparing best fit parameter values between NASA Exoplanet Archive with those of the fits files. 
CLIP_SIG = 5 # This is the CLIP_SIG clip for removal of erroneous points
CHI_LIM = 5 #Reject garbage with reduced chi squares above this value
ORDER = 2 # Order of normalization polynomial
PROJ_LIM = 0.001 # Limit for projection test
SLO_LIM = 3 # Number of sigma tolerance for slope test
TTV_ITERATIONS = 1 # Number of iterations of TTV shift fitting
WD='/home/dalp/Dropbox/sci/pro_exo/analysis/' + str(BATCH_ID) #Where stuff is saved. ../aux/ is assumed to contain help files (additional data on objects).
DAT_DIR='/home/dalp/data/kepler/sap/llc/' + str(BATCH_ID)

########
#Functions
def del_trans(dat):
    other_planets = np.where((meta.kid == dat.kid) & -(meta.koi == dat.koi))[0]
    for oplan in other_planets:
        oepoch = meta.tepoch[oplan]
        otperiod = meta.tperiod[oplan]
        otdur = meta.tdur[oplan]
        
        skip = np.isnan(dat.flux) | np.isnan(dat.time)
        ophase = np.mod(dat.time[-skip]-oepoch, otperiod)
        ophase = np.where(ophase > otperiod/2., ophase-otperiod, ophase)
        set_nan = np.where(np.abs(ophase) < 0.75*NORM_PAD/2.*otdur)[0] # Safety margin of 50% here
        fluxes = dat.flux[-skip]
        fluxes[set_nan] = np.nan
        dat.flux[-skip] = fluxes

def clear_nan(dat):
    not_nan = -np.isnan(dat.flux)
#    double_print(output, "\nRemoved " + str(np.sum(-not_nan)) + " NaN values"'\n')
    dat.upt_lists(not_nan)

def is_sorted(dat):
    return np.all(dat.time[i] <= dat.time[i+1] for i in xrange(len(dat.time)-1))

def calc_phase(dat):
    phase = np.mod(dat.time-dat.tepoch, dat.tperiod)
    phase = np.where(phase > dat.tperiod/2., phase-dat.tperiod, phase)
    dat.phas = phase

def chop(dat):
    keep = abs(dat.phas) < CHOP_WIDTH/2*dat.tdur
    dat.upt_lists(keep)

def normalize(dat):
    ntrans = 0
    diff = np.diff(dat.phas)
    cuts = np.where((diff[0:-1]<0)==True)[0]+1
    cuts = np.insert(cuts,0,0) # Inserts 0 at index 0
    cuts = np.append(cuts,len(dat.phas))

    dat.flux = dat.orig_flux.copy()
    dat.ferr = dat.orig_ferr.copy()
    
    for i in range(0,len(cuts)-1):            
        phase= dat.phas[cuts[i]:cuts[i+1]]
        flux = dat.flux[cuts[i]:cuts[i+1]]
        ferr = dat.ferr[cuts[i]:cuts[i+1]]
        oot = abs(phase) > NORM_PAD*dat.tdur/2+CADENCE

        before = np.sum(phase[oot] < 0) < PTS_MIN[0]
        in_trans = np.sum(-oot) < PTS_MIN[1]
        after = np.sum(phase[oot] > 0) < PTS_MIN[2]
        if before or in_trans or after:
            dat.flux[cuts[i]:cuts[i+1]] = np.nan
            continue

        coefs = np.polyfit(phase[oot],flux[oot], ORDER)
            
        correction = np.polyval(coefs, phase)
        dat.flux[cuts[i]:cuts[i+1]] = flux/correction-1
        dat.ferr[cuts[i]:cuts[i+1]] = ferr/correction
        ntrans += 1

    clear_nan(dat)
    return ntrans

def reorder(dat, mode):
    if mode == "phase":
        order = dat.phas.argsort()
    elif mode == "time":
        order = dat.time.argsort()
    else:
        exit("ERROR: Reordering mode unknown. This should never happen.")
    dat.upt_lists(order)

def fit_help(phase, fit_prad, fit_inclin, fit_dor, f1, f2): 
    fit_dor = np.abs(fit_dor)
    high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
    z = get_z(high_res, dat.tperiod, fit_inclin, fit_dor)

    #Far side of the star
    backside = np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]
    if np.any(backside):
        z[backside] = 10+fit_prad 
    
    if GFIT:
        h1 = poor_mans_bound(f1, GFIT)
        h2 = poor_mans_bound(f2, GFIT)
    else:
        h1 = h2 = 1.
        
    quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
    dat.ex_mod = quad-1.
    dat.ex_pha = high_res

    pts = round(CADENCE/MOD_RES)
    if pts <= len(dat.ex_mod):
        dat.instr_mod = np.convolve(dat.ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
    else: #Off chance that transit is short and highly resolved due to large number of transits
        double_print(output, "WARNING: Check instrumental convolution in fit!\n")
        dat.instr_mod = np.convolve(dat.ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

    dat.instr_mod = griddata(high_res, dat.instr_mod, phase, method='linear', fill_value=0)
#    print chi2red(dat.instr_mod, dat.flux, dat.ferr), fit_prad, fit_inclin, fit_dor, h1, h2
    dat.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2])
    return dat.instr_mod

def fit(dat, *args):
    guess1 = dat.prad
    guess2 = dat.inclin
    guess3 = dat.dor
    guess = np.array([guess1, guess2, guess3, 0., 0.])

    if len(args) > 0:
        guess = args[0]

    try:
        curve_par, covar = curve_fit(fit_help, dat.phas, dat.flux, guess, sigma=dat.ferr) #Error bars seem to make little difference
        dat.fit_par = curve_par
    except Exception: # Sloppy exception catching here!
        double_print(output, "WARNING: curve_fit() unable to find a fit\n")
        last_par = dat.fit_par
        fit_help(dat.phas, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        fchi = chi2red(dat.instr_mod, dat.flux, dat.ferr) # Fit chi
        fit_help(dat.phas, guess[0], guess[1], guess[2], guess[3], guess[4])
        gchi = chi2red(dat.instr_mod, dat.flux, dat.ferr) # Guessed chi
        if fchi < gchi:
            fit_help(dat.phas, last_par[0], last_par[1], last_par[2], last_par[3], last_par[4])
        else:
            dat.fit_par = guess
        
    problems = find_flaws(dat, guess, GFIT)
    if not problems is None:
        double_print(output, problems)

    fit_help(dat.phas, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])

def fit_ttv_help(time, fit_prad, fit_inclin, fit_dor, f1, f2, ttv_aa, ttv_ww, ttv_phi):
    # Bounds to TTV parameters
    ttv_par = ttv_bounds(ttv_aa, ttv_ww, ttv_phi)
    # TTV correction to time
    time = time + ttv_par[0]*np.sin(ttv_par[1]*time+ttv_par[2])
    
    # Compute phase
    phase = np.mod(time-dat.tepoch, dat.tperiod)
    phase = np.where(phase > dat.tperiod/2., phase-dat.tperiod, phase)
    
    fit_dor = np.abs(fit_dor)
    high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
    z = get_z(high_res, dat.tperiod, fit_inclin, fit_dor)

    #Far side of the star
    backside = np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]
    if np.any(backside):
        z[backside] = 10+fit_prad 
    
    if GFIT:
        h1 = poor_mans_bound(f1, GFIT)
        h2 = poor_mans_bound(f2, GFIT)
    else:
        h1 = h2 = 1.
        
    quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
    dat.ex_mod = quad-1.
    dat.ex_pha = high_res

    pts = round(CADENCE/MOD_RES)
    if pts <= len(dat.ex_mod):
        dat.instr_mod = np.convolve(dat.ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
    else: #Off chance that transit is short and highly resolved due to large number of transits
        double_print(output, "WARNING: Check instrumental convolution in fit!\n")
        dat.instr_mod = np.convolve(dat.ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

    dat.instr_mod = griddata(high_res, dat.instr_mod, phase, method='linear', fill_value=0)
#    print chi2red(dat.instr_mod, dat.flux, dat.ferr), fit_prad, fit_inclin, fit_dor, h1, h2, ttv_par[0], ttv_par[1], ttv_par[2]
    dat.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2, ttv_aa, ttv_ww, ttv_phi])
    return dat.instr_mod

def fit_ttv(dat, *args):
    guess1 = dat.prad
    guess2 = dat.inclin
    guess3 = dat.dor
    guess = np.array([guess1, guess2, guess3, 0., 0., 1e-7, 0.1, 0.])

    if len(args) > 0:
        guess = args[0]

    try:
        curve_par, covar = curve_fit(fit_ttv_help, dat.time, dat.flux, guess, sigma=dat.ferr) #Error bars seem to make little difference
        dat.fit_par = curve_par
    except Exception: # Sloppy exception catching here!
        double_print(output, "WARNING: curve_fit() unable to find a fit\n")
        last_par = dat.fit_par
        fit_ttv_help(dat.time, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4], dat.fit_par[5], dat.fit_par[6], dat.fit_par[7])
        fchi = chi2red(dat.instr_mod, dat.flux, dat.ferr) # Fit chi
        fit_ttv_help(dat.time, guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6], guess[7])
        gchi = chi2red(dat.instr_mod, dat.flux, dat.ferr) # Guessed chi
        if fchi < gchi:
            fit_ttv_help(dat.time, last_par[0], last_par[1], last_par[2], last_par[3], last_par[4], last_par[5], last_par[6], last_par[7])
        else:
            dat.fit_par = guess
        
    problems = find_flaws(dat, guess, GFIT)
    if not problems is None:
        double_print(output, problems)

    fit_ttv_help(dat.time, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4], dat.fit_par[5], dat.fit_par[6], dat.fit_par[7])

#def fit_ttv_v2_help(time, shift):
#    fit_prad, fit_inclin, fit_dor, f1, f2 = dat.fit_par
#    # Bounds to TTV parameters
#    ttv_v2_par = ttv_v2_bounds(shift)
#    # TTV correction to time
#    time = time + dat.ttv_lst + dat.elapsed_orbits*ttv_v2_par
#    
#    # Compute phase
#    phase = np.mod(time-dat.tepoch, dat.tperiod)
#    phase = np.where(phase > dat.tperiod/2., phase-dat.tperiod, phase)
#    
#    fit_dor = np.abs(fit_dor)
#    high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
#    z = get_z(high_res, dat.tperiod, fit_inclin, fit_dor)
#
#    #Far side of the star
#    backside = np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]
#    if np.any(backside):
#        z[backside] = 10+fit_prad 
#    
#    if GFIT:
#        h1 = poor_mans_bound(f1, GFIT)
#        h2 = poor_mans_bound(f2, GFIT)
#    else:
#        h1 = h2 = 1.
#        
#    quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
#    ex_mod = quad-1.
#    ex_pha = high_res
#
#    pts = round(CADENCE/MOD_RES)
#    if pts <= len(ex_mod):
#        instr_mod = np.convolve(ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
#    else: #Off chance that transit is short and highly resolved due to large number of transits
#        double_print(output, "WARNING: Check instrumental convolution in fit!\n")
#        instr_mod = np.convolve(ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]
#        
#    return griddata(high_res, instr_mod, phase, method='linear', fill_value=0)
#
#def fit_ttv_v2(dat):
#    if not dat.ttv is None: # revert previous iterations
#        dat.time -= dat.ttv
#    dat.ttv = np.zeros(dat.time.size)
#
#    dat.ttv_lst = 0
#
#    diff = np.diff(dat.phas)
#    cuts = np.where((diff[0:-1]<0)==True)[0]+1
#    cuts = np.insert(cuts,0,0) # Inserts 0 at index 0
#    cuts = np.append(cuts,len(dat.phas))
#
#    for i in range(0,len(cuts)-1):
#        time = dat.time[cuts[i]:cuts[i+1]]
#        flux = dat.flux[cuts[i]:cuts[i+1]]
#        ferr = dat.ferr[cuts[i]:cuts[i+1]]
#
#        if i == 0:
#            dat.elapsed_orbits = 1
#        else:
#            dat.elapsed_orbits = dat.time[cuts[i]]-dat.time[cuts[i-1]]
#            dat.elapsed_orbits = dat.elapsed_orbits//dat.tperiod + 1
#        
#        try:
#            shift, covar = curve_fit(fit_ttv_v2_help, time, flux, 0, sigma=ferr)
#        except Exception: # Sloppy exception catching here!
#            double_print(output, "WARNING: curve_fit() unable to find a TTV shift!\n")
#            shift = 0.
#
#        dat.ttv[cuts[i]:cuts[i+1]] = dat.ttv_lst + dat.elapsed_orbits*ttv_v2_bounds(shift)
#        dat.ttv_lst = dat.ttv[cuts[i]]
#
#    dat.time += dat.ttv
#    calc_phase(dat)
##    plt.plot(dat.time, dat.ttv, '.')
##    plt.show()

#def fit_ttv_v3_help(time, flux, ferr, shift):
#    fit_prad, fit_inclin, fit_dor, f1, f2 = dat.fit_par
#    # TTV correction to time
#    time = time + shift/24./60.
#    
#    # Compute phase
#    phase = np.mod(time-dat.tepoch, dat.tperiod)
#    phase = np.where(phase > dat.tperiod/2., phase-dat.tperiod, phase)
#    
#    fit_dor = np.abs(fit_dor)
#    high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
#    z = get_z(high_res, dat.tperiod, fit_inclin, fit_dor)
#
#    #Far side of the star
#    backside = np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]
#    if np.any(backside):
#        z[backside] = 10+fit_prad 
#    
#    if GFIT:
#        h1 = poor_mans_bound(f1, GFIT)
#        h2 = poor_mans_bound(f2, GFIT)
#    else:
#        h1 = h2 = 1.
#        
#    quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
#    ex_mod = quad-1.
#    ex_pha = high_res
#
#    pts = round(CADENCE/MOD_RES)
#    if pts <= len(ex_mod):
#        instr_mod = np.convolve(ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
#    else: #Off chance that transit is short and highly resolved due to large number of transits
#        double_print(output, "WARNING: Check instrumental convolution in fit!\n")
#        instr_mod = np.convolve(ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]
#
#    instr_mod = griddata(high_res, instr_mod, phase, method='linear', fill_value=0)
#    plt.plot(phase, instr_mod)
#    plt.plot(phase, flux)
#    plt.show()
#    return np.sum((instr_mod-flux)**2/ferr**2)
#
#def fit_ttv_v3(dat):
#    if not dat.ttv is None: # revert previous iterations
#        dat.time -= dat.ttv
#    dat.ttv = np.zeros(dat.time.size)
#
#    diff = np.diff(dat.phas)
#    cuts = np.where((diff[0:-1]<0)==True)[0]+1
#    cuts = np.insert(cuts,0,0) # Inserts 0 at index 0
#    cuts = np.append(cuts,len(dat.phas))
#
#    for i in range(0,len(cuts)-1):
#        time = dat.time[cuts[i]:cuts[i+1]]
#        flux = dat.flux[cuts[i]:cuts[i+1]]
#        ferr = dat.ferr[cuts[i]:cuts[i+1]]
#
#        shifts = np.arange(-30,31)
#        gof_chi2 = np.zeros(shifts.size)
#        for idx, shift in enumerate(shifts):
#            gof_chi2[idx] = fit_ttv_v3_help(time, flux, ferr, shift)
#            
#        dat.ttv[cuts[i]:cuts[i+1]] = shifts[np.argmin(gof_chi2)]/24./60.
##        if np.random.rand() < 0.03:
##            plt.plot(shifts, gof_chi2)
##            plt.show()
##
##    plt.plot(dat.time, dat.ttv)
##    plt.show()
#    dat.time += dat.ttv
#    calc_phase(dat)
    
def purge_old(dat):
    for clip in range(0,2):
        true_err = np.std(dat.flux-dat.instr_mod)
        good = np.where(abs(dat.flux-dat.instr_mod) < 3.*true_err)[0] #This is effectively, highly heuristically, a 4 sigma limit
        if np.std(dat.flux[good]-dat.instr_mod[good]) > 0.9*true_err:
            good = np.where(abs(dat.flux-dat.instr_mod) < 4.*true_err)[0]

        dat.upt_lists(good)
        guess = dat.fit_par
        if guess.size < 8:
            fit(dat, guess)
        else:
            fit_ttv_v2(dat, guess)

        if  np.std(dat.flux-dat.instr_mod) < 0.001:
            break
        double_print(output, "WARNING: Applying clip condition twice! \n")
        
def purge(dat):
    delsig = (dat.flux-dat.instr_mod)/dat.ferr
    clip = np.abs(delsig) < CLIP_SIG
    dat.upt_lists(clip)

# Filter transits based on baseline projections
def filt_pro(dat):
    removed = 0
    rm_pro = 0
    rm_slo = 0
    diff = np.diff(dat.phas)
    cuts = np.where((diff[0:-1]<0)==True)[0]+1
    cuts = np.insert(cuts,0,0) # Inserts 0 at index 0
    cuts = np.append(cuts,len(dat.phas))
    
    for i in range(0,len(cuts)-1):
        phase= dat.phas[cuts[i]:cuts[i+1]]
        flux = dat.flux[cuts[i]:cuts[i+1]]
        ferr = dat.ferr[cuts[i]:cuts[i+1]]

        before = phase < -NORM_PAD*dat.tdur/2-CADENCE
        after =  phase >  NORM_PAD*dat.tdur/2+CADENCE

        acoefs, avar = np.polyfit(phase[after],flux[after],1,w=1./ferr[after], cov=True)
        bproj = np.polyval(acoefs, phase[before])
        bcoefs, bvar = np.polyfit(phase[before],flux[before],1,w=1./ferr[before], cov=True)
        aproj = np.polyval(bcoefs, phase[after])

        if np.abs(np.mean(aproj)) > PROJ_LIM or np.abs(np.mean(bproj)) > PROJ_LIM:
            dat.flux[cuts[i]:cuts[i+1]] = np.nan
            removed += 1
            rm_pro += 1

        elif np.abs(acoefs[0]/np.sqrt(avar[0,0])) > SLO_LIM or np.abs(bcoefs[0]/np.sqrt(bvar[0,0])) > SLO_LIM:
            dat.flux[cuts[i]:cuts[i+1]] = np.nan
            removed += 1
            rm_slo += 1

    clear_nan(dat)
    dat.calc_ntrans -= removed
    dat.npro += rm_pro
    dat.nslo += rm_slo

# Filter transits based on goodness of fit
def filt_gof(dat):
    removed = 0
    diff = np.diff(dat.phas)
    cuts = np.where((diff[0:-1]<0)==True)[0]+1
    cuts = np.insert(cuts,0,0) # Inserts 0 at index 0
    cuts = np.append(cuts,len(dat.phas))
    sig_tot = np.std(dat.flux-dat.instr_mod)
    chi2vals = np.zeros(len(cuts)-1)
    
    for i in range(0,len(cuts)-1):
        flux = dat.flux[cuts[i]:cuts[i+1]]
        tmp_mod = dat.instr_mod[cuts[i]:cuts[i+1]]
        chi2vals[i] = chi2red(flux, tmp_mod, sig_tot)
        
        if chi2vals[i] > CHI_LIM:
            dat.flux[cuts[i]:cuts[i+1]] = np.nan
            removed += 1

    clear_nan(dat)
    dat.calc_ntrans -= removed
    dat.ngof += removed
    if dat.calc_ntrans < 1:
        double_print(output, "WARNING: Negative number of transits. This should never happen!\n")

def vary_binning(dat):
    # Choose kernel slightly larger than largest gap, also catch initial and trailing holes
    # Should have been filtered in normalize, check through vetting
    kernel = (1+1e-4)*max([max(np.diff(dat.phas)[:-1]), dat.phas[0]+CHOP_WIDTH*dat.tdur/2, CHOP_WIDTH*dat.tdur/2-dat.phas[-1]])
    while True:
        pts = int(CHOP_WIDTH*dat.tdur/kernel)
        if pts < 2:
            if len(dat.std) == 0:
                dat.binsize = [kernel]
                dat.std = [1.]
            if len(dat.std) < 2:
                double_print(output, "WARNING: Binning unsuccessful, check distribution of points!\n")
                dat.binsize.append(dat.binsize[0])
                dat.std.append(dat.std[0])
                dat.set_conv(np.zeros(1), np.zeros(1))
            break

        conv_phase, conv_res = np.zeros(pts), np.zeros(pts)
        conv_phase = kernel*np.linspace(0.5,pts-0.5,pts)-CHOP_WIDTH*dat.tdur/2
        for i in range(0,pts):
            include = abs(dat.phas-conv_phase[i]) < kernel/2
            conv_res[i] = np.mean(dat.flux[include]-dat.instr_mod[include])
            conv_phase[i] = np.mean(dat.phas[include])

        dat.binsize.append(kernel*24*60)
        dat.std.append(np.std(conv_res))
#        plt.plot(conv_phase, conv_res)
#        plt.show()

        if kernel < CADENCE/3. or dat.conv_phase is None:
            dat.set_conv(conv_phase, conv_res)

        kernel *= 1.1
        
    dat.binsize = np.array(dat.binsize)
    dat.std = np.array(dat.std)

def print_plots(dat, meta, output, values):
    print_fit_par(dat, meta, output, GFIT, values)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(15, 24), dpi=300, facecolor='w', edgecolor='k')
    plot_pc(dat, ax1, SPARSE, PLT_FMT)
    plot_delchi(dat, ax2, meta, SPARSE, GFIT)
    plot_sem(dat, ax3, values)
    plot_gauss(dat, ax4)
    plot_cdelchi(dat, ax5, meta)
    plot_limb(dat, ax6, GFIT)
    fig.savefig(str(dat.kid) + "_" + str(dat.koi) + '.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close(fig) #DO NOT REMOVE THIS LINE, IT WILL DEVOUR YOUR MEMORY

################################################################
#Some preparation
GFIT = 2.*GFIT/np.pi # This is because arctans are used as boundaries
os.chdir(WD) #Move to designated directory
output = open('output_' + BATCH_ID + '.dat', 'w')
files = sorted(glob(DAT_DIR+'/*llc.fits')) #Find all files.

counter = 0
ntrans_tot = 0
ngof_tot = 0
npro_tot = 0
nslo_tot = 0
accept_tot = 0
processed = []
duplicates = []
rejected = []
meta = Meta(WD + '/../../aux/candidates.dat')
values = Values(BATCH_ID)
last_stamp = datetime.now()

########
# Relate KID to file paths
kid2file = {}
for kid in meta.kid:
    kid2file[kid] = []
    
for ff in files:
    kid = int(ff[-32:-23])
    kid2file[kid].append(ff)



################################################################
#Main part
for meta_ind, kid in enumerate(meta.kid):
    counter += 1
    if kid2file[kid] == []:
        double_print(output, "WARNING: Planet without data!\n\n")
        continue

    # Load data
    dat = Data(meta, meta_ind)
    for ff in kid2file[kid]:
        dat.add(ff)

    double_print(output, str(meta_ind) + "/" + str(len(meta.kid)) + " KIC/KID: " + str(dat.kid) + " KOI: " + str(dat.koi) + "\n")

    # Prepare data
    del_trans(dat)
    clear_nan(dat)
    if not is_sorted(dat):
        double_print(output, "WARNING: Input not sorted even though glob was sorted!\n\n")
    calc_phase(dat)
    chop(dat)
    if normalize(dat) == 0:
        double_print(output, "NOTE: No acceptable transits for " + str(dat.koi) + "! Probably because transit duration (" + str(dat.tdur) + ") too short.\n\n")
        continue
    reorder(dat, "phase")

    # Remove individual points
    fit(dat)
    purge(dat)
    reorder(dat, "time")
    dat.calc_ntrans = normalize(dat)
    dat.init_ntrans = dat.calc_ntrans
    if dat.init_ntrans == 0:
        double_print(output, "WARNING: No acceptable transits for " + str(dat.koi) + " after purge!\n\n")
        continue

    # Remove poor transits
    filt_pro(dat)
    if dat.calc_ntrans == 0:
        double_print(output, "WARNING: No acceptable transits for " + str(dat.koi) + " after projection/slope filter!\n\n")
        continue

    filt_gof(dat)
    fit(dat, dat.fit_par)
#    for ttv_iter in range(0, TTV_ITERATIONS):
#        fit_ttv_v3(dat)
#        fit(dat, dat.fit_par)
        
    reorder(dat, "phase")
    vary_binning(dat)

    values.save(dat, GFIT)
    dat.save_res()
    print_plots(dat, meta, output, values)

    accept_tot += 1
    ntrans_tot += dat.calc_ntrans
    npro_tot += dat.npro
    nslo_tot += dat.nslo
    ngof_tot += dat.ngof
    double_print(output, str(datetime.now()-last_stamp) + '\n\n')
    last_stamp = datetime.now()

double_print(output, "Total number of candidates: " + str(counter) + "\n")
double_print(output, "Total number of accepted planets: " + str(accept_tot) + "\n")
double_print(output, "Total number of projection filtered transits: " + str(npro_tot) + "\n")
double_print(output, "Total number of slope filtered transits: " + str(nslo_tot) + "\n")
double_print(output, "Total number of chi2 filtered transits: " + str(ngof_tot) + "\n")
double_print(output, "Total number of transits: " + str(ntrans_tot) + "\n")
output.close()
