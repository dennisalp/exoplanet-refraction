#Python 2.X
#Dennis Alp
#2016-08-30
import numpy as np
import os
import sys

from datetime import datetime
from exop_trans_occultquad import occultquad
from glob import glob
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import sem

from exop_dv_classes import *
from exop_dv_plotter import *
from exop_dv_helper import *

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

#########
#User input
BATCH_ID = sys.argv[1] #Unique ID to avoid conflict if several scripts run simultaneously. 
OUT_ID = sys.argv[2]
PLT_FMT = '.k' #Choose marker style (format)
SPARSE = 500. #Number of points in plots to improve performance
CADENCE = 29.4243814323648 #In minutes
CHOP_WIDTH = 3. #In transit durations
GFIT = 0.1 #Fit gamma and how much, arctan/(pi/2)*GFIT
MOD_RES = CADENCE/301. #Resolution when computing model, in units of minutes. Make odd multiple of cadence preferable, or discrete convolution (prevents an off-by-one shift).
MGT1 = 2 #Apply linear correction when more than much greater than 1 (MGT1) number of values
GT1 = 1 #Apply constant correction when more than greater than 1 (GT1) number of values
TOL = 0.05 #Relative tolerance when comparing best fit parameter values between NASA Exoplanet Archive with those of the fits files. 
CHI_LIM = 10 #Reject garbage with reduced chi squares above this value
WD = '/home/dalp/Dropbox/astrophysics/project_exoplanets/' #Where input is found.
OUT_DIR = '/home/dalp/Dropbox/astrophysics/project_exoplanets/' + str(OUT_ID) #Where output is saved.
DAT_DIR = '/home/dalp/data/kepler/koi/koi_' + str(BATCH_ID) #Where data is, will be globbed.

#########
#Functions
def clear_nan(dat):
    not_nan = -np.isnan(dat.flux)
    dat.upt_lists(not_nan)

    margin = 5
    sane = (dat.flux < 0.005)
    sane2 = np.copy(sane)
    for i in range(margin,len(sane)-margin):
        sane2[i] = not any(-sane[i-margin:i+margin])
    dat.upt_lists(sane2)

def chop(dat):
    include = abs(dat.phase) < CHOP_WIDTH/2*dat.tdur
    dat.upt_lists(include)

def normalize(dat):
    diff = np.diff(dat.phase)
    cuts = np.where((diff[0:-1]<0)==True)[0]+1
    cuts = np.insert(cuts,0,0)
    cuts = np.append(cuts,len(dat.phase))
    for i in range(0,len(cuts)-1):
        phase = dat.phase[cuts[i]:cuts[i+1]]
        flux = dat.flux[cuts[i]:cuts[i+1]]
        ferr = dat.ferr[cuts[i]:cuts[i+1]]
        omit = abs(phase) < dat.tdur/2+CADENCE
        if len(phase[-omit]) > MGT1:
            coefs = np.polyfit(phase[-omit],flux[-omit]+1,1)
        elif len(phase[-omit]) > GT1:
            coefs = np.polyfit(phase[-omit],flux[-omit]+1,0)
        if len(phase[-omit]) > GT1:
            correction = np.polyval(coefs, phase)
            dat.flux[cuts[i]:cuts[i+1]] = (flux+1)/correction-1
            dat.ferr[cuts[i]:cuts[i+1]] = (ferr)/correction
        else:
            dat.flux[cuts[i]:cuts[i+1]] = flux
            dat.ferr[cuts[i]:cuts[i+1]] = ferr

def ovrlay(dat, s1vals):
    def transit_model(phase, fit_prad, fit_inclin, fit_dor, f1, f2): 
        high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*high_res/dat.tperiod)
        sin = np.sin(2*np.pi*high_res/dat.tperiod)
        z = fit_dor*np.sqrt(sin**2+cos**2)
        z[np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]] = 2+fit_prad #Dark side of the star

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
            dat.instr_mod = np.convolve(dat.ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

        dat.instr_mod = griddata(high_res, dat.instr_mod, phase, method='linear', fill_value=0)
#        print chi2red(dat.instr_mod, dat.flux, dat.ferr), fit_prad, fit_inclin, fit_dor, h1, h2
        dat.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2]).T
#        print fit_prad, fit_inclin, fit_dor, f1, f2
#        print dat.fit_par, np.array([fit_prad, fit_inclin, fit_dor, f1, f2]).T[0], np.array([fit_prad, fit_inclin, fit_dor, f1, f2]).T
        dat.z = griddata(high_res, z, phase, method='linear', fill_value=2+fit_prad)

    order = dat.phase.argsort()
    dat.upt_lists(order)

#    chi2red(dat.instr_mod, dat.flux, dat.ferr)
    prad = s1vals.prad[s1ind]
    inclin = s1vals.inclin[s1ind]
    dor = s1vals.dor[s1ind]
    f1 = s1vals.f1[s1ind]
    f2 = s1vals.f2[s1ind]
    transit_model(dat.phase, prad, inclin, dor, f1, f2)

def purge(dat, s1vals):
    good = np.where(abs(dat.flux-dat.instr_mod) < 4.*s1vals.err[s1ind])[0]
    dat.z = dat.z[good]
    dat.instr_mod = dat.instr_mod[good]
    dat.upt_lists(good)

def trim(dat):
    prad = 3*s1vals.prad[s1ind] #Make sure to include atmosphere
    iind = np.where((dat.z > 1-prad) & (dat.z < 1+4*prad) & (dat.phase < CADENCE))[0]
    eind = np.where((dat.z > 1-prad) & (dat.z < 1+4*prad) & (dat.phase > -CADENCE))[0]
    edge = np.concatenate((iind, eind))
    return edge, np.in1d(iind,eind).any(), edge.size == 0

#Poor-mans fit, duh
def pmfit(dat, s2vals, ref_mod):
    def refr_help(x, t, a, shift):
        z = ref_mod[:,0].copy()
        y = ref_mod[:,1]-1

        res = (z[1]-z[0])
        pts = round(CADENCE/res)
        pts = 1 if pts < 1 else pts #This should never happen, only catches garbage.
        y = np.convolve(y, np.ones(pts)/pts,'same') #Fake Kepler binning
        z[pts/2:-pts/2] = np.convolve(z, np.ones(pts)/pts,'same')[pts/2:-pts/2] #Fake Kepler binning
        y = griddata((z-1.083)*dat.tdur*abs(t)+abs(shift)*dat.tdur/2., y*abs(a), abs(x), method='linear', fill_value=0)
##        print 'pmfit',t,a
        return y

    edge, grazing, miss = trim(dat)
    if grazing or miss:
        print 'Grazing, or miss: ' + str(grazing) + ' ' + str(miss)
        return True

    dat.tdur = calc_tdur(dat)
    guess = np.array([1., 1., 1.])
    try:
        par, covar = curve_fit(refr_help, dat.phase[edge], dat.flux[edge], guess, sigma=dat.ferr[edge])
    except Exception:
        print 'WARNING: curve_fit failed!'
        par = np.array([1., 0., 1.])

    s2vals.edge = edge
    s2vals.t, s2vals.a = par[0], par[1]
    s2vals.pm_mod = refr_help(dat.phase[edge], par[0], par[1], par[2])
#    plt.plot(dat.phase[edge], s2vals.pm_mod)
#    plt.plot(dat.phase[edge], dat.flux[edge])
#    plt.show()
    s2vals.std = np.std(dat.flux[edge]-s2vals.pm_mod)
    dat.ferr = s2vals.std*np.ones(edge.size)
    return False

#fits difference between models
def delta_fit(dat, s2vals, delta_mod):
    def transit_model(phase, fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA): 
        high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*high_res/dat.tperiod)
        sin = np.sin(2*np.pi*high_res/dat.tperiod)
        z = fit_dor*np.sqrt(sin**2+cos**2)
        z[np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]] = 2+fit_prad #Dark side of the star

        if GFIT:
            h1 = poor_mans_bound(f1, GFIT)
            h2 = poor_mans_bound(f2, GFIT)
        else:
            h1 = h2 = 1.
            
        quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
        ex_mod = quad-1.

        #############
#        MODIFICATION
        delT = delT if delT < 5. else 5.
        delz = (delta_mod[:,0]-1)*abs(delT)+1
        dely = delta_mod[:,1]*abs(delA)
#        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*phase/dat.tperiod)
#        sin = np.sin(2*np.pi*phase/dat.tperiod)
#        z = fit_dor*np.sqrt(sin**2+cos**2)
        ex_mod = ex_mod + griddata(delz, dely, z, method='linear', fill_value=0)
#        plt.plot(z,griddata(delz, dely, z, method='linear', fill_value=0))
#        plt.show()
        #############

        pts = round(CADENCE/MOD_RES)
        if pts <= len(ex_mod):
            s2vals.delta_mod = np.convolve(ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
        else: #Off chance that transit is short and highly resolved due to large number of transits
            s2vals.delta_mod = np.convolve(ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

##        print 'delta_fit',delT,delA
        s2vals.delta_mod = griddata(high_res, s2vals.delta_mod, phase, method='linear', fill_value=0)
        s2vals.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA])
        return s2vals.delta_mod

    prad = s1vals.prad[s1ind]
    inclin = s1vals.inclin[s1ind]
    dor = s1vals.dor[s1ind]
    f1 = s1vals.f1[s1ind]
    f2 = s1vals.f2[s1ind]
    guess = np.array([prad, inclin, dor, f1, f2, 1., 0.01*prad**2])

    try:
        curve_par, covar = curve_fit(transit_model, dat.phase[s2vals.edge], dat.flux[s2vals.edge], guess, sigma=dat.ferr) #Error bars seem to make little difference
        s2vals.fit_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit when delta fitting\n")
        last_par = s2vals.fit_par
        transit_model(dat.phase[s2vals.edge], s2vals.fit_par[0], s2vals.fit_par[1], s2vals.fit_par[2], s2vals.fit_par[3], s2vals.fit_par[4], s2vals.fit_par[5], s2vals.fit_par[6])
        fchi = chi2red(s2vals.delta_mod, dat.flux[s2vals.edge], dat.ferr)
        transit_model(dat.phase[s2vals.edge], guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6])
        gchi = chi2red(s2vals.delta_mod, dat.flux[s2vals.edge], dat.ferr)
        if fchi < gchi:
            transit_model(dat.phase[s2vals.edge], last_par[0], last_par[1], last_par[2], last_par[3], last_par[4], last_par[5], last_par[6])
        else:
            s2vals.fit_par = guess

    s2vals.delta_mod = transit_model(dat.phase[s2vals.edge], s2vals.fit_par[0], s2vals.fit_par[1], s2vals.fit_par[2], s2vals.fit_par[3], s2vals.fit_par[4], s2vals.fit_par[5], s2vals.fit_par[6])
    s2vals.delT, s2vals.delA = s2vals.fit_par[5], s2vals.fit_par[6]

#fits Gauss component
def gauss_fit(dat, s2vals):
    def transit_model(phase, fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA): 
        high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*high_res/dat.tperiod)
        sin = np.sin(2*np.pi*high_res/dat.tperiod)
        z = fit_dor*np.sqrt(sin**2+cos**2)
        z[np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]] = 2+fit_prad #Dark side of the star

        if GFIT:
            h1 = poor_mans_bound(f1, GFIT)
            h2 = poor_mans_bound(f2, GFIT)
        else:
            h1 = h2 = 1.
            
        quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
        ex_mod = quad-1.

        #############
#        MODIFICATION
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        delz = np.linspace(-4*abs(delT), 4*abs(delT), 100)+1.
        dely = gaussian(delz, 1., abs(delT))*delA
#        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*phase/dat.tperiod)
#        sin = np.sin(2*np.pi*phase/dat.tperiod)
#        z = fit_dor*np.sqrt(sin**2+cos**2)
        ex_mod = ex_mod + griddata(delz, dely, z, method='linear', fill_value=0)
#        plt.plot(z,griddata(delz, dely, z, method='linear', fill_value=0))
#        plt.show()
        #############

        pts = round(CADENCE/MOD_RES)
        if pts <= len(ex_mod):
            s2vals.gauss_mod = np.convolve(ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
        else: #Off chance that transit is short and highly resolved due to large number of transits
            s2vals.gauss_mod = np.convolve(ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

        s2vals.gauss_mod = griddata(high_res, s2vals.gauss_mod, phase, method='linear', fill_value=0)
#        print 'gauss_fit',chi2red(s2vals.gauss_mod, dat.flux[s2vals.edge], dat.ferr),fit_prad, fit_inclin, fit_dor, f1, f2,delT,delA
        s2vals.gauss_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA])
        return s2vals.gauss_mod

    prad = s1vals.prad[s1ind]
    inclin = s1vals.inclin[s1ind]
    dor = s1vals.dor[s1ind]
    f1 = s1vals.f1[s1ind]
    f2 = s1vals.f2[s1ind]
    guess = np.array([prad, inclin, dor, f1, f2, prad, 0.01*prad**2])

    try:
        curve_par, covar = curve_fit(transit_model, dat.phase[s2vals.edge], dat.flux[s2vals.edge], guess, sigma=dat.ferr) #Error bars seem to make little difference
        s2vals.gauss_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit when Gauss fitting\n")
        last_par = s2vals.gauss_par
        transit_model(dat.phase[s2vals.edge], s2vals.gauss_par[0], s2vals.gauss_par[1], s2vals.gauss_par[2], s2vals.gauss_par[3], s2vals.gauss_par[4], s2vals.gauss_par[5], s2vals.gauss_par[6])
        fchi = chi2red(s2vals.gauss_mod, dat.flux[s2vals.edge], dat.ferr)
        transit_model(dat.phase[s2vals.edge], guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6])
        gchi = chi2red(s2vals.gauss_mod, dat.flux[s2vals.edge], dat.ferr)
        if fchi < gchi:
            transit_model(dat.phase[s2vals.edge], last_par[0], last_par[1], last_par[2], last_par[3], last_par[4], last_par[5], last_par[6])
        else:
            s2vals.gauss_par = guess

    s2vals.gauss_mod = transit_model(dat.phase[s2vals.edge], s2vals.gauss_par[0], s2vals.gauss_par[1], s2vals.gauss_par[2], s2vals.gauss_par[3], s2vals.gauss_par[4], s2vals.gauss_par[5], s2vals.gauss_par[6])
    s2vals.gauss_delT, s2vals.gauss_delA = s2vals.gauss_par[5], s2vals.gauss_par[6]

def pppv(dat, s2vals, delta_mod):
    def transit_model(phase, fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA): 
        high_res = np.linspace(-CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/2., CHOP_WIDTH*dat.tdur/MOD_RES)
        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*high_res/dat.tperiod)
        sin = np.sin(2*np.pi*high_res/dat.tperiod)
        z = fit_dor*np.sqrt(sin**2+cos**2)
        z[np.where(abs(2*np.pi*high_res/dat.tperiod)>np.pi/2)[0]] = 2+fit_prad #Dark side of the star

        if GFIT:
            h1 = poor_mans_bound(f1, GFIT)
            h2 = poor_mans_bound(f2, GFIT)
        else:
            h1 = h2 = 1.
            
        quad, lin = occultquad(z, h1*dat.gamma1, h2*dat.gamma2, fit_prad) #Compute model
        ex_mod = quad-1.

        #############
#        MODIFICATION
        if ref_on:
            delT = delT if delT < 5. else 5.
            delz = (delta_mod[:,0]-1)*abs(delT)+1
            dely = delta_mod[:,1]*abs(delA)
#        cos = np.cos(np.deg2rad(fit_inclin))*np.cos(2*np.pi*phase/dat.tperiod)
#        sin = np.sin(2*np.pi*phase/dat.tperiod)
#        z = fit_dor*np.sqrt(sin**2+cos**2)
            ex_mod = ex_mod + griddata(delz, dely, z, method='linear', fill_value=0)
#        plt.plot(z,griddata(delz, dely, z, method='linear', fill_value=0))
#        plt.show()
        #############

        pts = round(CADENCE/MOD_RES)
        if pts <= len(ex_mod):
            s2vals.pppv_mod = np.convolve(ex_mod, np.ones(pts)/pts,'same') #Fake Kepler binning
        else: #Off chance that transit is short and highly resolved due to large number of transits
            s2vals.pppv_mod = np.convolve(ex_mod, np.ones(pts)/pts)[int(pts)/2:-int(pts)/2+1]

##        print 'pppv_fit',delT,delA, ref_on
        s2vals.pppv_mod = griddata(high_res, s2vals.pppv_mod, phase, method='linear', fill_value=0)
        s2vals.pppv_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2, delT, delA])
        return s2vals.pppv_mod

    s2vals.fake_phase = np.random.uniform(-CHOP_WIDTH/2*dat.tdur, CHOP_WIDTH/2*dat.tdur, dat.phase.size)
    epsilon = np.random.normal(0., s2vals.std, dat.phase.size)
    s2vals.fake_flux = griddata(dat.phase, dat.instr_mod, s2vals.fake_phase) + epsilon

    prad = s1vals.prad[s1ind]
    inclin = s1vals.inclin[s1ind]
    dor = s1vals.dor[s1ind]
    f1 = s1vals.f1[s1ind]
    f2 = s1vals.f2[s1ind]
    guess = np.array([prad, inclin, dor, f1, f2, 1., 0.01*prad**2])

    #########
    #Some code copying
    cos = np.cos(np.deg2rad(inclin))*np.cos(2*np.pi*s2vals.fake_phase/dat.tperiod)
    sin = np.sin(2*np.pi*s2vals.fake_phase/dat.tperiod)
    z = dor*np.sqrt(sin**2+cos**2)
    z[np.where(abs(2*np.pi*s2vals.fake_phase/dat.tperiod)>np.pi/2)[0]] = 2+prad #Dark side of the star
    prad3 = 3*prad #Make sure to include atmosphere
    iind = np.where((z > 1-prad3) & (z < 1+4*prad3) & (s2vals.fake_phase < CADENCE))[0]
    eind = np.where((z > 1-prad3) & (z < 1+4*prad3) & (s2vals.fake_phase > -CADENCE))[0]
    s2vals.pppv_edge = np.concatenate((iind, eind))
    #########

#########
#Without refraction
    ref_on = False
    s2vals.pppv_err = s2vals.std*np.ones(s2vals.pppv_edge.size)
    try:
        curve_par, covar = curve_fit(transit_model, s2vals.fake_phase[s2vals.pppv_edge], s2vals.fake_flux[s2vals.pppv_edge], guess, sigma=s2vals.pppv_err) #Error bars seem to make little difference
        s2vals.pppv_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit when nor fake fitting\n")
        last_par = s2vals.pppv_par
        transit_model(s2vals.fake_phase[s2vals.pppv_edge], s2vals.pppv_par[0], s2vals.pppv_par[1], s2vals.pppv_par[2], s2vals.pppv_par[3], s2vals.pppv_par[4], s2vals.pppv_par[5], s2vals.pppv_par[6])
        fchi = chi2red(s2vals.pppv_mod, s2vals.fake_flux[s2vals.pppv_edge], s2vals.pppv_err)
        transit_model(s2vals.fake_phase[s2vals.pppv_edge], guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6])
        gchi = chi2red(s2vals.pppv_mod, s2vals.fake_flux[s2vals.pppv_edge], s2vals.pppv_err)
        if fchi < gchi:
            transit_model(s2vals.fake_phase[s2vals.pppv_edge], last_par[0], last_par[1], last_par[2], last_par[3], last_par[4], last_par[5], last_par[6])
        else:
            s2vals.pppv_par = guess

    s2vals.pppv_mod_nor = transit_model(s2vals.fake_phase[s2vals.pppv_edge], s2vals.pppv_par[0], s2vals.pppv_par[1], s2vals.pppv_par[2], s2vals.pppv_par[3], s2vals.pppv_par[4], s2vals.pppv_par[5], s2vals.pppv_par[6])
    s2vals.pppv_delT_nor, s2vals.pppv_delA_nor = s2vals.pppv_par[5], s2vals.pppv_par[6]
    s2vals.pppv_par_nor = s2vals.pppv_par

#########
#Now with refraction
    ref_on = True
    try:
        curve_par, covar = curve_fit(transit_model, s2vals.fake_phase[s2vals.pppv_edge], s2vals.fake_flux[s2vals.pppv_edge], guess, sigma=s2vals.pppv_err) #Error bars seem to make little difference
        s2vals.pppv_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit when fake fitting\n")
        last_par = s2vals.pppv_par
        transit_model(s2vals.fake_phase[s2vals.pppv_edge], s2vals.pppv_par[0], s2vals.pppv_par[1], s2vals.pppv_par[2], s2vals.pppv_par[3], s2vals.pppv_par[4], s2vals.pppv_par[5], s2vals.pppv_par[6])
        fchi = chi2red(s2vals.pppv_mod, s2vals.fake_flux[s2vals.pppv_edge], s2vals.pppv_err)
        transit_model(s2vals.fake_phase[s2vals.pppv_edge], guess[0], guess[1], guess[2], guess[3], guess[4], guess[5], guess[6])
        gchi = chi2red(s2vals.pppv_mod, s2vals.fake_flux[s2vals.pppv_edge], s2vals.pppv_err)
        if fchi < gchi:
            transit_model(s2vals.fake_phase[s2vals.pppv_edge], last_par[0], last_par[1], last_par[2], last_par[3], last_par[4], last_par[5], last_par[6])
        else:
            s2vals.pppv_par = guess

    s2vals.pppv_mod = transit_model(s2vals.fake_phase[s2vals.pppv_edge], s2vals.pppv_par[0], s2vals.pppv_par[1], s2vals.pppv_par[2], s2vals.pppv_par[3], s2vals.pppv_par[4], s2vals.pppv_par[5], s2vals.pppv_par[6])
    s2vals.pppv_delT, s2vals.pppv_delA = s2vals.pppv_par[5], s2vals.pppv_par[6]

def s2res(dat, s2vals):
    e = s2vals.edge
    s2vals.chi2nor = chi2(dat.instr_mod[e], dat.flux[e], s2vals.std)
    s2vals.chi2nor_red = chi2red(dat.instr_mod[e], dat.flux[e], s2vals.std)

    s2vals.chi2ref = chi2(s2vals.pm_mod, dat.flux[e], s2vals.std)
    s2vals.delchi2 = s2vals.chi2nor-s2vals.chi2ref
    s2vals.chi2ref_red = chi2red(s2vals.pm_mod, dat.flux[e], s2vals.std)
    s2vals.delchi2_red = s2vals.chi2nor_red-s2vals.chi2ref_red

    s2vals.del_chi2ref = chi2(s2vals.delta_mod, dat.flux[e], s2vals.std)
    s2vals.del_delchi2 = s2vals.chi2nor-s2vals.del_chi2ref
    s2vals.del_chi2ref_red = chi2red(s2vals.delta_mod, dat.flux[e], s2vals.std)
    s2vals.del_delchi2_red = s2vals.chi2nor_red-s2vals.del_chi2ref_red

    s2vals.gauss_chi2ref = chi2(s2vals.gauss_mod, dat.flux[e], s2vals.std)
    s2vals.gauss_delchi2 = s2vals.chi2nor-s2vals.gauss_chi2ref
    s2vals.gauss_chi2ref_red = chi2red(s2vals.gauss_mod, dat.flux[e], s2vals.std)
    s2vals.gauss_delchi2_red = s2vals.chi2nor_red-s2vals.gauss_chi2ref_red
    
    e = s2vals.pppv_edge
    s2vals.pppv_chi2nor = chi2(s2vals.pppv_mod_nor, s2vals.fake_flux[e], s2vals.std)
    s2vals.pppv_chi2ref = chi2(s2vals.pppv_mod, s2vals.fake_flux[e], s2vals.std)
    s2vals.pppv_delchi2 = s2vals.pppv_chi2nor-s2vals.pppv_chi2ref
    s2vals.pppv_chi2nor_red = chi2red(s2vals.pppv_mod_nor, s2vals.fake_flux[e], s2vals.std)
    s2vals.pppv_chi2ref_red = chi2red(s2vals.pppv_mod, s2vals.fake_flux[e], s2vals.std)
    s2vals.pppv_delchi2_red = s2vals.pppv_chi2nor_red-s2vals.pppv_chi2ref_red

#########
#Some preparation
CADENCE = float(CADENCE)/(24*60) #Convert to days
MOD_RES = float(MOD_RES)/(24*60)
GFIT = 2.*GFIT/np.pi
files=glob(DAT_DIR+'/*dvt.fits') #Find all files. 
os.chdir(OUT_DIR) #Move to designated directory
output = open('output_' + OUT_ID + '.dat', 'w')

counter = 0
meta = Meta(WD + '/aux/confirmed.txt')
s1vals = S1vals(WD + 'analysis_' + BATCH_ID + '/values_*.dat')# + '../analysis_?nea/values_?.dat')
s2vals = S2vals(BATCH_ID)
ref_mod = np.loadtxt(WD + '/models/cfslow_nc.dat')
delta_mod = np.loadtxt(WD + '/models/delta_sandbox.dat')
last_stamp = datetime.now()

#########
#Main part
for system in files:
    for tce in range(1,fits.getval(system,'NEXTEND')-1):
        meta_ind = meta.get_ind(system, tce, TOL)
        if meta_ind >= 0:
            counter += 1
            print '\n', counter, meta_ind, system, tce
            dat = Data(meta, system, tce, meta_ind)

            s1ind = np.where(float(dat.koi.replace("_","")) == s1vals.koi)[0]
            if s1ind.size > 1:
                dummy = tce-1 if tce <= s1ind.size else -1  #The guess of tce-1 is risky business! Probably not related to globbing, though.
                s1ind = s1ind[dummy]
            else:
                s1ind = s1ind[0]
            
            clear_nan(dat)
            chop(dat)
            normalize(dat)
            ovrlay(dat, s1vals)
            purge(dat, s1vals)
            if pmfit(dat, s2vals, ref_mod):
                continue
            delta_fit(dat, s2vals, delta_mod)
            gauss_fit(dat, s2vals)
            pppv(dat, s2vals, delta_mod)
            s2res(dat, s2vals)
            s2vals.save(dat, s1vals, s1ind)

#            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20), dpi=300, facecolor='w', edgecolor='k')
#            plot_pc(dat, ax1, SPARSE, PLT_FMT)
#            plot_delchi(dat, ax2, meta, SPARSE)
#            plot_pmfit(dat, s2vals, ax3, SPARSE, PLT_FMT)
#            plot_delmod(dat, s2vals, ax4, SPARSE)
#            plot_binres(dat, s2vals, ax5)
#            fig.savefig(str(dat.koi) + '_' + str(tce) + '.pdf',bbox_inches='tight', pad_inches=0.01)
#            plt.close(fig) #DO NOT REMOVE THIS LINE, IT WILL DEVOUR YOUR MEMORY

            double_print(output, str(datetime.now()-last_stamp) + '\n\n')
            last_stamp = datetime.now()

output.close()
s2vals.f_handle.close()
s2vals.residuals.close()
plot_res(BATCH_ID, PLT_FMT)
