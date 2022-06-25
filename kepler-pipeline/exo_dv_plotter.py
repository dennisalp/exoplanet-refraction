import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.stats import norm

from exo_dv_helper import *

#########
#Prints and plots
def print_fit_par(dat, meta, output, GFIT, tce, vals):
    coefs = np.polyfit(np.log(dat.binsize[0:dat.binsize.size/2]),np.log(dat.std[0:dat.binsize.size/2]),1)
    double_print(output, "KOI Name: " + dat.koi + "_(" + str(tce) + "), Reduced chi2: {0:.5}, Precision: {1:.5}\n".format(chi2red(dat.instr_mod, dat.flux, dat.ferr), min(dat.std)))
    double_print(output, "\t\t\tFitted\t\tNASA\t\tDV\n")
    double_print(output, "Transit duration:\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(calc_tdur(dat), meta.tdur[dat.meta_ind], dat.dvtdur))
    double_print(output, "Radius ratio:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[0], meta.prad[dat.meta_ind], dat.dvprad))
    double_print(output, "Inclination:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[1], meta.inclin[dat.meta_ind], dat.dvinclin))
    double_print(output, "Distance dor:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[2], meta.dor[dat.meta_ind], dat.dvdor))
    double_print(output, "Number of transits:\t\t\t{0}\t\t{1}\n".format(int(dat.ntrans), int(dat.dvntrans)))
    double_print(output, "gamma1:\t\t\t{0:.5}\t\t{1:.5}\n".format(poor_mans_bound(dat.fit_par[3], GFIT), dat.gamma1))
    double_print(output, "gamma2:\t\t\t{0:.5}\t\t{1:.5}\n".format(poor_mans_bound(dat.fit_par[4], GFIT), dat.gamma2))
    double_print(output, "alpha = {0:.7}, xi_5 = {1:.7}, xi_30 = {2:.7}, zeta_5 = {3}, zeta_10 = {4}, zeta_30 = {5}\n".format(coefs[0], vals.xi5, vals.xi30, vals.zeta5, vals.zeta10, vals.zeta30))
    output.flush()

def plot_pc(dat, ax, SPARSE, PLT_FMT):
    inter = int(len(dat.phase)/SPARSE)+1
    ax.plot(dat.ex_pha, dat.ex_mod, 'r-')
    ax.plot(dat.phase[::inter], dat.instr_mod[::inter], 'b-')
    ax.errorbar(dat.phase[::inter], dat.flux[::inter], yerr=dat.ferr[::inter], ecolor='k', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Instrumental binning','Stacked data'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_cdelchi(dat, ax, meta):
    ax.plot(dat.conv_phase, dat.conv_res, 'k')
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('$\Delta$F')

def plot_delchi(dat, ax, meta, SPARSE):
    inter = int(len(dat.phase)/SPARSE)+1
    ax.plot(dat.phase[::inter], (dat.flux[::inter]-dat.instr_mod[::inter])/dat.ferr[::inter], 'b.')
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    vls  = "KOI Name: " + dat.koi.replace('_','\_') + ", Reduced $\chi^2$: {0:<10.5}\n".format(chi2red(dat.instr_mod, dat.flux, dat.ferr))
    vls += "\t\t\tFitted\t\tNASA\t\tDV\n"
    vls += "Transit duration:\t\t\t{0:<10.5}\t{1:<10.5}\n".format(meta.tdur[dat.meta_ind], dat.dvtdur)
    vls += "Radius ratio:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[0], meta.prad[dat.meta_ind], dat.dvprad)
    vls += "Inclination:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[1], meta.inclin[dat.meta_ind], dat.dvinclin)
    vls += "Distance dor:\t\t{0:<10.5}\t{1:<10.5}\t{2:<10.5}\n".format(dat.fit_par[2], meta.dor[dat.meta_ind], dat.dvdor)
    vls += "Number of transits:\t\t\t{0}\t\t{1}\n".format(int(dat.ntrans), int(dat.dvntrans))
    vls += "$\gamma_1$:\t\t\t\t\t{0:.5}\n".format(dat.gamma1)
    vls += "$\gamma_2$:\t\t\t\t\t{0:.5}".format(dat.gamma2)
    ax.annotate(vls, xy=(0, 1), xytext=(12, -12), va='top', xycoords='axes fraction', textcoords='offset points')
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('$\Delta\chi$ [$\sigma$]')
    ax.set_ylim(1.1*min(((dat.flux-dat.instr_mod)/dat.ferr)[::inter]), 2*max(((dat.flux-dat.instr_mod)/dat.ferr)[::inter])) #To fit the vls

def plot_sem(dat, ax, vals):
    ax.loglog(dat.binsize, dat.std, 'k.')
    coefs = np.polyfit(np.log(dat.binsize[0:dat.binsize.size/2]),np.log(dat.std[0:dat.binsize.size/2]),1)
    xbins = np.array([min(dat.binsize), max(dat.binsize)])
    ax.loglog(xbins, np.e**coefs[1]*xbins**coefs[0])
    norm = xbins[0]**coefs[0]/xbins[0]**-0.5
    ax.loglog(xbins, np.e**coefs[1]*xbins**-0.5*norm, 'r') #Sorry, but it works!
    ls = "$\\alpha = {0:.7}$, $\\xi_5 = {1:.7}$, $\\xi_{{30}} = {2:.7}$".format(coefs[0], vals.xi5, vals.xi30)
    ax.annotate(ls, xy=(0, 0.3), xytext=(12, -12), va='top', xycoords='axes fraction', textcoords='offset points')
    ax.set_xlabel('Bin width [min]')
    ax.set_ylabel('$\sigma$ [Normalized flux]')
    ax.set_xlim(min(dat.binsize)*0.9,max(dat.binsize)*1.1)

def plot_limb(dat, ax, GFIT):
    r = np.linspace(0,1,1001)
    mu = np.sqrt(1-r**2)
    ax.plot(r, 1-poor_mans_bound(dat.fit_par[3], GFIT)*dat.gamma1*(1-mu)-poor_mans_bound(dat.fit_par[4], GFIT)*dat.gamma2*(1-mu)**2, 'k')
    ax.set_xlabel('Normalized radius')
    ax.set_ylabel('I(r)')

def plot_gauss(dat, ax):
    res = dat.flux-dat.instr_mod
    n, bins, patches = ax.hist(res, 100, normed=1)
    (mu, sigma) = norm.fit(res)
    y = mlab.normpdf( bins, mu, sigma)
    ax.plot(bins, y, 'r')
    ax.set_xlabel('$\sigma$ [Normalized flux]')
    ax.set_ylabel('Probability density')

#Plots the residuals from all fits. 
def plot_res(BATCH_ID, PLT_FMT):
    res = np.loadtxt('residuals_' + BATCH_ID + '.dat', delimiter = ',')
    weights = 1./res[:,1]

    pha, res_nr, err_nr, dummy = tbin(res[:,0], res[:,2], 0.25, weights) #No refraction  
    pha, res_wr, err_wr, dummy = tbin(res[:,0], res[:,4], 0.25, weights) #With refraction

    pha = np.where(np.isnan(pha), 0., pha)
    gd = np.where(abs(pha) > 2.6)
    plt.plot(pha[gd], res_nr[gd], 'bo')
#    plt.plot(pha[gd], res_wr[gd], 'bo')
    plt.errorbar(pha[gd], res_nr[gd], yerr=err_nr[gd], ecolor='b', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
#    plt.errorbar(pha[gd], res_wr[gd], yerr=err_wr[gd], ecolor='b', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
#    plt.errorbar(pha, res_wr, yerr=err_wr, ecolor='b', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
    plt.axvline(x=-5, linewidth=2, color = 'k')
    plt.axvline(x=-4, linewidth=2, color = 'k')
    plt.axvline(x=5, linewidth=2, color = 'k')
    plt.axvline(x=4, linewidth=2, color = 'k')
    plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
#    plt.legend(['Without refraction','With refraction'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel('Shifted phase (horizontal bands are ingress/egress) [Ingress duration]')
    plt.ylabel('Normalized flux')
    plt.savefig('residuals_' + BATCH_ID + '.pdf',bbox_inches='tight', pad_inches=0.01)
#    plt.show()

def plot_par(res):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), dpi=300, facecolor='w', edgecolor='k')
    bins = np.linspace(0, 1, 100)
    ax1.hist(results.radii[:,0], bins, alpha=1/3., label='Fitted')
    ax1.hist(results.radii[:,1], bins, alpha=1/3., label='NASA')
    ax1.hist(results.radii[:,2], bins, alpha=1/3., label='DV')
    ax1.set_xlabel('Planet radius [Star radius]')
    ax1.set_ylabel('Number')
    ax1.legend(loc='best')
    bins = np.linspace(0, 90, 100)
    ax2.hist(results.inclins[:,0], bins, alpha=1/3., label='Fitted')
    ax2.hist(results.inclins[:,1], bins, alpha=1/3., label='NASA')
    ax2.hist(results.inclins[:,2], bins, alpha=1/3., label='DV')
    ax2.set_xlabel('Inclination [Degrees]')
    ax2.set_ylabel('Number')
    ax2.legend(loc='best')
    bins = np.linspace(0, 256, 100)
    ax3.hist(results.distances[:,0], bins, alpha=1/3., label='Fitted')
    ax3.hist(results.distances[:,1], bins, alpha=1/3., label='NASA')
    ax3.hist(results.distances[:,2], bins, alpha=1/3., label='DV')
    ax3.set_xlabel('Planet distance [Star radius]')
    ax3.set_ylabel('Number')
    ax3.legend(loc='best')
    fig.savefig('parameters_' + BATCH_ID + '.pdf',bbox_inches='tight', pad_inches=0.01)

def plot_pmfit(dat, s2vals, ax, SPARSE, PLT_FMT):
    inter = int(len(dat.phase)/SPARSE)+1
    ax.plot(s2vals.ex_pha, s2vals.ex_mod, 'r-')
    ax.plot(dat.phase[::inter], s2vals.pm_mod[::inter], 'b-')
    ax.errorbar(dat.phase[::inter], dat.flux[::inter], yerr=s2vals.std, ecolor='k', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Instrumental binning','Stacked data'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_delmod(dat, s2vals, ax, SPARSE):
    inter = int(len(dat.phase)/SPARSE)+1
    temp = griddata(dat.ex_pha, dat.ex_mod, s2vals.ex_pha)
    ax.plot(s2vals.ex_pha, s2vals.ex_mod-temp, 'r')
    ax.plot(dat.phase[::inter], s2vals.pm_mod[::inter]-dat.instr_mod[::inter], 'b')
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Convolved model'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_binres(dat, s2val, ax):
    t, tmp= tbin(dat.phase, dat.flux-dat.instr_mod, 4.5/(24.*60.))
    ax.plot(t, tmp, 'r')
    t, tmp= tbin(dat.phase, dat.flux-s2val.conv_mod, 4.5/(24.*60.))
    ax.plot(t, tmp, 'b')

    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    ax.legend(['Without refraction','With refraction'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('$\Delta$F')

#def plot_val(vals, BATCH_ID):
#    plt.plot(vals.prad, vals.alpha, '.')
#    plt.xlabel('Planet radius [Host star radius]')
#    plt.ylabel('$\\alpha$')
#    plt.savefig('values_' + BATCH_ID + '.pdf',bbox_inches='tight', pad_inches=0.01)
#    plt.show()
