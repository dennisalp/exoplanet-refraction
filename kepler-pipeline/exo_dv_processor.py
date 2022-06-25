#Python 2.X Dennis Alp 2017-01-15
#
#It seems to leak memory, slowly. Takes Kepler dv light curves and
#chop-stack-fits them.  Fitting is performed using Levenberg-Marquardt
#algorithm from scipy.optimize.fit_curve.
#
#There are 4696 KOI candidates in the NASA Exoplanet Archive (NEA) as
#of 2016-08-05 2272 confirmed planets have a dv summary and stellar
#data (defined as upper limit to metallicity, this exludes some bad
#stellar data objects). Note that 9 of these are flagged as false
#positives in the Kepler database but are included anyway.
#
#The candidates without dv .fits data either constitute a system
#with another planet that has dv data, i.e. even though the system has
#been processed by the dv pipeline, the candidate still has no dv
#data. Alternatively, the period supplied in the dv .fits file is a
#factor of 0.5,2,3,4,... off, which results in the aforementioned
#case. Note that it is not possible to identify which planet each
#extension in the .fits file corresponds to, which is why the period
#and epoch have been used within each system. The KOI name is not in
#the dv .fits file and one would think that the tables in the .fits
#file, named TCE_#, would match the TCE number found in NEA, but this
#is not the case.
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.table import Table
from datetime import datetime
from glob import glob
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import sem

from exo_trans_occultquad import occultquad
from exo_dv_classes import *
from exo_dv_plotter import *
from exo_dv_helper import *

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

#########
#User input
BATCH_ID = sys.argv[1] #Unique ID to avoid conflict if several scripts run simultaneously. 
PLT_FMT = '.k' #Choose marker style (format)
SPARSE = 500000. #Number of points in plots to improve performance
CADENCE = 29.4243814323648 #In minutes
CHOP_WIDTH = 3. #In transit durations
GFIT = 0.1 #Fit gamma and how much, arctan/(pi/2)*GFIT
MOD_RES = CADENCE/301. #Resolution when computing model, in units of minutes. Make odd multiple of cadence preferable, or discrete convolution (prevents an off-by-one shift).
MGT1 = 2 #Apply linear correction when more than much greater than 1 (MGT1) number of values
GT1 = 1 #Apply constant correction when more than greater than 1 (GT1) number of values
TOL = 0.05 #Relative tolerance when comparing best fit parameter values between NASA Exoplanet Archive with those of the fits files. 
CHI_LIM = 10 #Reject garbage with reduced chi squares above this value
WD='/home/dalp/Dropbox/sci/project_exo/analysis_' + str(BATCH_ID) #Where stuff is saved. ../aux/ is assumed to contain help files (additional data on objects).
DAT_DIR='/home/dalp/data/kepler/koi/koi_' + str(BATCH_ID)

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
            print 'WARNING: Few points for linear normalization'
        if len(phase[-omit]) > GT1:
            correction = np.polyval(coefs, phase)
            dat.flux[cuts[i]:cuts[i+1]] = (flux+1)/correction-1
            dat.ferr[cuts[i]:cuts[i+1]] = (ferr)/correction
        else:
            dat.flux[cuts[i]:cuts[i+1]] = flux
            dat.ferr[cuts[i]:cuts[i+1]] = ferr

def fit(dat):
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
        dat.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2])
        return dat.instr_mod

    order = dat.phase.argsort()
    dat.upt_lists(order)
    
    if dat.prad < 1 and dat.dor > 1+dat.prad and dat.dor*np.cos(np.deg2rad(dat.inclin)) < 1+dat.prad:
        guess1 = dat.prad
        guess2 = dat.inclin
        guess3 = dat.dor
    else:
        guess1 = dat.dvprad
        guess2 = dat.dvinclin
        guess3 = dat.dvdor

    guess = np.array([guess1, guess2, guess3, 0., 0.])
    try:
        curve_par, covar = curve_fit(transit_model, dat.phase, dat.flux, guess, sigma=dat.ferr) #Error bars seem to make little difference
        dat.fit_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit\n")
        last_par = dat.fit_par
        transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        fchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        transit_model(dat.phase, guess[0], guess[1], guess[2], guess[3], guess[4])
        gchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        if fchi < gchi:
            transit_model(dat.phase, last_par[0], last_par[1], last_par[2], last_par[3], last_par[4])
        else:
            dat.fit_par = guess
        
    if not sanity_check(dat, guess, GFIT):
        transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        double_print(output, "WARNING: Fit not sound\n")

    transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
#    transit_model(dat.phase, guess[0], guess[1], guess[2], guess[3], guess[4])

def logfit(dat):
    def transit_model(phase, fit_prad, fit_b, fit_dor, f1, f2): 
        fit_inclin = b2inclin(fit_b, fit_dor)

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
#        print chi2red(dat.instr_mod, dat.flux, dat.ferr), fit_prad, fit_b, fit_dor, h1, h2
        dat.fit_par = np.array([fit_prad, fit_b, fit_dor, f1, f2])
        return dat.instr_mod

    order = dat.phase.argsort()
    dat.upt_lists(order)
    
    if dat.prad < 1 and dat.dor > 1+dat.prad and np.log10(abs(inclin2b(dat.inclin,dat.dor))) < 1+dat.prad:
        guess1 = dat.prad
        guess2 = inclin2b(dat.inclin, dat.dor)
        guess3 = dat.dor
        double_print(output,'NASA EA\n')
    else:
        double_print(output,'DV\n')
        guess1 = dat.dvprad
        guess2 = inclin2b(dat.dvinclin, dat.dvdor)
        guess3 = dat.dvdor

    guess = np.array([guess1, guess2, guess3, 0., 0.])
    try:
        curve_par, covar = curve_fit(transit_model, dat.phase, dat.flux, guess, sigma=dat.ferr) #Error bars seem to make little difference
        dat.fit_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit\n")
        last_par = dat.fit_par

        transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        fchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        transit_model(dat.phase, guess[0], guess[1], guess[2], guess[3], guess[4])

        gchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        if fchi < gchi:
            transit_model(dat.phase, last_par[0], last_par[1], last_par[2], last_par[3], last_par[4])
        
    if not sanity_check(dat, guess, GFIT):
        double_print(output, "WARNING: Fit not sound\n")

    transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
    dat.fit_par[1] = b2inclin(dat.fit_par[1], dat.fit_par[2])

def purge(dat):
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
        dat.fit_par = np.array([fit_prad, fit_inclin, fit_dor, f1, f2])
        return dat.instr_mod

    true_err = np.std(dat.flux-dat.instr_mod)
    good = np.where(abs(dat.flux-dat.instr_mod) < 3.*true_err)[0] #This is effectively, highly heuristically, a 4 sigma limit
    if np.std(dat.flux[good]-dat.instr_mod[good]) > 0.9*true_err:
        good = np.where(abs(dat.flux-dat.instr_mod) < 4.*true_err)[0]

    dat.upt_lists(good)

    guess = dat.fit_par
    try:
        curve_par, covar = curve_fit(transit_model, dat.phase, dat.flux, guess, sigma=dat.ferr) #Error bars seem to make little difference
        dat.fit_par = curve_par
    except Exception:
        double_print(output, "WARNING: curve_fit() unable to find a fit\n")
        last_par = dat.fit_par
        transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        fchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        transit_model(dat.phase, guess[0], guess[1], guess[2], guess[3], guess[4])
        gchi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        if fchi < gchi:
            transit_model(dat.phase, last_par[0], last_par[1], last_par[2], last_par[3], last_par[4])
        else:
            dat.fit_par = guess
        
    if not sanity_check(dat, guess, GFIT):
        transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
        double_print(output, "WARNING: Fit not sound\n")

    transit_model(dat.phase, dat.fit_par[0], dat.fit_par[1], dat.fit_par[2], dat.fit_par[3], dat.fit_par[4])
#    transit_model(dat.phase, guess[0], guess[1], guess[2], guess[3], guess[4])

def vary_binning(dat):
    kernel = (1+1e-4)*max([max(np.diff(dat.phase)[:-1]), dat.phase[0]+CHOP_WIDTH*dat.tdur/2, CHOP_WIDTH*dat.tdur/2-dat.phase[-1]])
    while True:
        pts = int(CHOP_WIDTH*dat.tdur/kernel)
        if pts < 2:
            if len(dat.std) == 0:
                dat.binsize = [kernel]
                dat.std = [1.]
            if len(dat.std) < 2:
                dat.binsize.append(dat.binsize[0])
                dat.std.append(dat.std[0])
                dat.set_conv(np.zeros(1), np.zeros(1))
            break

        conv_phase, conv_res = np.zeros(pts), np.zeros(pts)
        conv_phase = kernel*np.linspace(0.5,pts-0.5,pts)-CHOP_WIDTH*dat.tdur/2
        for i in range(0,pts):
            include = abs(dat.phase-conv_phase[i]) < kernel/2
            conv_res[i] = np.mean(dat.flux[include]-dat.instr_mod[include])
            conv_phase[i] = np.mean(dat.phase[include])

        dat.binsize.append(kernel*24*60)
        dat.std.append(np.std(conv_res))
#        plt.plot(conv_phase, conv_res)
#        plt.show()

        if kernel < CADENCE/3. or dat.conv_phase is None:
            dat.set_conv(conv_phase, conv_res)

        kernel *= 1.1
        
    dat.binsize = np.array(dat.binsize)
    dat.std = np.array(dat.std)

#########
#Some preparation
CADENCE = float(CADENCE)/(24*60) #Convert to days
MOD_RES = float(MOD_RES)/(24*60)
GFIT = 2.*GFIT/np.pi
files=glob(DAT_DIR+'/*dvt.fits') #Find all files. 
os.chdir(WD) #Move to designated directory
output = open('output_' + BATCH_ID + '.dat', 'w')

counter = 0
ntrans_tot = 0
processed = []
duplicates = []
rejected = []
meta = Meta(WD + '/../aux/confirmed.csv')
values = Values(BATCH_ID)
last_stamp = datetime.now()

#########
#Main part
for system in files:
    for tce in range(1,fits.getval(system,'NEXTEND')-1):
        meta_ind = meta.get_ind(system, tce, TOL)
        if meta_ind >= 0:
            if meta_ind in processed:
                duplicates.append(system + ' ' + str(tce))
                double_print(output, "WARNING: Two candidates identified as the same candidate!\n")

            counter += 1
            print '\n', counter, meta_ind, system, tce
            processed.append(meta_ind)
            dat = Data(meta, system, tce, meta_ind)
            ntrans_tot += dat.ntrans
            
            clear_nan(dat)
            chop(dat)
            normalize(dat)
            fit(dat)
            purge(dat)
            vary_binning(dat)
            values.save(dat, GFIT)

            print_fit_par(dat, meta, output, GFIT, tce, values)
            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(15, 24), dpi=300, facecolor='w', edgecolor='k')
            plot_pc(dat, ax1, SPARSE, PLT_FMT)
            plot_delchi(dat, ax2, meta, SPARSE)
            plot_sem(dat, ax3, values)
            plot_gauss(dat, ax4)
            plot_cdelchi(dat, ax5, meta)
            plot_limb(dat, ax6, GFIT)
            fig.savefig(str(dat.koi) + '_' + str(tce) + '.pdf',bbox_inches='tight', pad_inches=0.01)
            plt.close(fig) #DO NOT REMOVE THIS LINE, IT WILL DEVOUR YOUR MEMORY
            double_print(output, str(datetime.now()-last_stamp) + '\n\n')
            last_stamp = datetime.now()
        else:
            rejected.append(system + ' ' + str(tce))

double_print(output, "\n\nTotal number of transits:" + str(ntrans_tot) + '\n\n')
output.close()
