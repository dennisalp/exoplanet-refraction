import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.stats import norm

from exo_sap_hlp import *

#########
#Prints and plots
def print_fit_par(dat, meta, output, GFIT, vals):
    coefs = np.polyfit(np.log(dat.binsize[0:dat.binsize.size/2]),np.log(dat.std[0:dat.binsize.size/2]),1)
    double_print(output, "Reduced chi2: {0:.5}, Precision: {1:.5}\n".format(chi2red(dat.instr_mod, dat.flux, dat.ferr), min(dat.std)))
    double_print(output, "\t\t\tFitted\t\tNASA\n")
    double_print(output, "Transit duration:\t{0:<10.5}\t{1:<10.5}\n".format(calc_tdur(dat), dat.tdur))
    double_print(output, "Radius ratio:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[0], dat.prad))
    double_print(output, "Inclination:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[1], dat.inclin))
    double_print(output, "Distance dor:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[2], dat.dor))
    double_print(output, "Number of transits:\t{0}\t\t{1}\n".format(dat.calc_ntrans, dat.ntrans))
    double_print(output, "gamma1:\t\t\t{0:.5}\t\t{1:.5}\n".format(poor_mans_bound(dat.fit_par[3], GFIT), dat.gamma1))
    double_print(output, "gamma2:\t\t\t{0:.5}\t\t{1:.5}\n".format(poor_mans_bound(dat.fit_par[4], GFIT), dat.gamma2))
    double_print(output, "alpha = {0:.7}\n".format(coefs[0])) 
    double_print(output, "xi_5 = {0:.7}, xi_10 = {1:.7}, xi_30 = {2:.7}\n".format(vals.xi5, vals.xi10, vals.xi30))
    double_print(output, "zeta_5 = {0}, zeta_10 = {1}, zeta_30 = {2}\n".format(vals.zeta5, vals.zeta10, vals.zeta30))
    output.flush()

def plot_pc(dat, ax, SPARSE, PLT_FMT):
    inter = int(len(dat.phas)/SPARSE)+1
    ax.plot(dat.ex_pha, dat.ex_mod, 'r-')
    ax.plot(dat.phas[::inter], dat.instr_mod[::inter], 'b-')
    ax.errorbar(dat.phas[::inter], dat.flux[::inter], yerr=dat.ferr[::inter], ecolor='k', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Instrumental binning','Stacked data'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_cdelchi(dat, ax, meta):
    ax.plot(dat.conv_phase, dat.conv_res, 'k')
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('$\Delta$F')

def plot_delchi(dat, ax, meta, SPARSE, GFIT):
    inter = int(len(dat.phas)/SPARSE)+1
    ax.plot(dat.phas[::inter], (dat.flux[::inter]-dat.instr_mod[::inter])/dat.ferr[::inter], 'b.')
    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    vls  = "KOI Name: " + str(dat.koi) + ", Reduced $\chi^2$: {0:<10.5}\n".format(chi2red(dat.instr_mod, dat.flux, dat.ferr))
    vls += "\t\t\tFitted\t\tNASA\n"
    vls += "Transit duration:\t{0:<10.5}\t{1:<10.5}\n".format(calc_tdur(dat), dat.tdur)
    vls += "Radius ratio:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[0], dat.prad)
    vls += "Inclination:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[1], dat.inclin)
    vls += "Distance dor:\t\t{0:<10.5}\t{1:<10.5}\n".format(dat.fit_par[2], dat.dor)
    vls += "Number of transits:\t{0}\t\t{1}\n".format(dat.calc_ntrans, dat.ntrans)
    vls += "gamma1:\t\t\t{0:.5}\t\t{1:.5}\n".format(poor_mans_bound(dat.fit_par[3], GFIT), dat.gamma1)
    vls += "gamma2:\t\t\t{0:.5}\t\t{1:.5}".format(poor_mans_bound(dat.fit_par[4], GFIT), dat.gamma2)
    ax.annotate(vls, xy=(0, 1), xytext=(12, -12), va='top', xycoords='axes fraction', textcoords='offset points')
#    ax.set_xlabel('Phase [days]')
#    ax.set_ylabel('$\Delta\chi$ [$\sigma$]')
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
    inter = int(len(dat.phas)/SPARSE)+1
    ax.plot(s2vals.ex_pha, s2vals.ex_mod, 'r-')
    ax.plot(dat.phas[::inter], s2vals.pm_mod[::inter], 'b-')
    ax.errorbar(dat.phas[::inter], dat.flux[::inter], yerr=s2vals.std, ecolor='k', fmt=PLT_FMT, capsize=0, elinewidth=2, markersize=0.7)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Instrumental binning','Stacked data'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_delmod(dat, s2vals, ax, SPARSE):
    inter = int(len(dat.phas)/SPARSE)+1
    temp = griddata(dat.ex_pha, dat.ex_mod, s2vals.ex_pha)
    ax.plot(s2vals.ex_pha, s2vals.ex_mod-temp, 'r')
    ax.plot(dat.phas[::inter], s2vals.pm_mod[::inter]-dat.instr_mod[::inter], 'b')
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('Relative flux')
    ax.legend(['Exact model','Convolved model'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)

def plot_binres(dat, s2val, ax):
    t, tmp= tbin(dat.phas, dat.flux-dat.instr_mod, 4.5/(24.*60.))
    ax.plot(t, tmp, 'r')
    t, tmp= tbin(dat.phas, dat.flux-s2val.conv_mod, 4.5/(24.*60.))
    ax.plot(t, tmp, 'b')

    ax.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
    ax.legend(['Without refraction','With refraction'],numpoints=1,loc='best', fancybox=True, framealpha=0.3)
    ax.set_xlabel('Phase [days]')
    ax.set_ylabel('$\Delta$F')

#def plt_res_all(res):
#    plt.imshow(np.log10(res.hmap.T), interpolation="nearest", cmap="gray", aspect='auto', origin='lower')
#    plt.plot(res.mean, linewidth=3)
#    plt.errorbar(np.arange(0,res.NX), res.mean, fmt='ok', yerr=res.sem)
#    plt.plot(res.NY/2*res.count/np.mean(res.count), linewidth=2)
#    plt.plot(res.NY/2*res.err/np.mean(res.err), linewidth=2)
#    print res.ngood, np.mean(res.count), np.sum(res.count)
#
#    plt.axvline(x=7.5, linewidth=2, color = 'k')
#    plt.axvline(x=15.5, linewidth=2, color = 'k')
#    plt.axvline(x=21.5, linewidth=2, color = 'k')
#    plt.axvline(x=25.5, linewidth=2, color = 'k')
#    plt.axvline(x=23.5, linewidth=2, color = 'k')
#    plt.axvline(x=5.5, linewidth=2, color = 'k')
#    plt.axvline(x=9.5, linewidth=2, color = 'k')
#    plt.axhline(y=128, xmin=0, xmax=1, linewidth=2, color = 'k')
#    
#    plt.show()

def plt_res_ppm(res):
    nx = float(res.NX)
    xx = np.arange(0, res.NX)
    fig = plt.figure(figsize=(5, 3.75))
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)
    
#    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#    ax1.plot(xx[:res.NX/2], res.mean[:res.NX/2], linewidth=2, alpha=0.5)
    ax1.errorbar(xx[:res.NX/2], res.mean[:res.NX/2], fmt='.k', yerr=res.sem[:res.NX/2], zorder=20)
#    ax1.plot(xx, res.NY/2*res.count/np.mean(res.count), linewidth=2)
#    ax1.plot(xx, res.NY/2*res.err/np.mean(res.err), linewidth=2, color="r")
#    ax1.plot(xx, res.err/100., linewidth=2, color="r")
    
#    ax2.plot(xx[res.NX/2:], res.mean[res.NX/2:], linewidth=2)
#    ax2.plot(res.mvx, res.mva)
    ax2.errorbar(xx[res.NX/2:], res.mean[res.NX/2:], fmt='.k', yerr=res.sem[res.NX/2:], zorder=20)
#    ax2.plot(xx, res.NY/2*res.count/np.mean(res.count), linewidth=2)
#    ax2.plot(xx, res.NY/2*res.err/np.mean(res.err), linewidth=2, color="r")
#    ax2.plot(xx, res.err/100., linewidth=2, color="r")
    print res.ngood, np.mean(res.count), np.sum(res.count)

    ax1.set_title("Ingress")
    ax2.set_title("Egress")
    ax.set_xlabel("Planet-centre to star-limb distance (planet radius)", labelpad=18)
    ax1.set_ylabel("$\Delta F$ (ppm)")

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')
    ax2.tick_params(labelleft='off')
    ax2.yaxis.tick_right()

    # Add the prediction
    dat = np.loadtxt('/Users/silver/Dropbox/sci/pro_exo/models/20d_jovian.dat')
    zz               = dat[:,1]
    conv_refraction  = dat[:,7][zz.size/2:]
    conv_fit_mod_ldc = dat[:,9][zz.size/2:]
    p0 = np.loadtxt('/Users/silver/Dropbox/sci/pro_exo/models/20d_jovian_fit_par.dat')[0,0]
    scaled = (zz-1)/p0
    scaled = scaled[zz.size/2:]
    
    mvx = scaled*res.uni+res.midpt2
    mva = np.zeros(mvx.size)
#    tmp_zz = np.where(res.phase > 0, zz, -zz)
    for ii, mv_pos in enumerate(mvx):
        inside = np.abs(mvx - mv_pos) < 1/4.
        if np.sum(inside) == 0:
            mva[ii] = 0.
            continue
        
        mva[ii] = np.mean(conv_refraction[inside]-conv_fit_mod_ldc[inside])

    mva /= 1e-6
    ax2.plot(mvx-0.5, mva, zorder=10, lw=2)
    mvx = res.NX/2-(mvx-res.NX/2)
    ax1.plot(mvx-0.5, mva, zorder=10, lw=2)
    
    # Ticks and stuff
    mid = nx/2.
    qrt = nx/4.
    pos = np.arange(-qrt-0.5+2, qrt-0.5, 4)
    lab = np.arange(-qrt/2.+1, qrt/2)[::2].astype('int')
    xlabel_pos = list(qrt+pos) + list(3*qrt+pos)
    xlabel_str = list(lab) + list(lab)
    ax1.set_xticks(xlabel_pos)
    ax1.set_xticklabels(xlabel_str[::-1])
    ax2.set_xticks(xlabel_pos)
    ax2.set_xticklabels(xlabel_str)
    ax1.set_xlim(    -0.5, nx/2-0.5)
    ax2.set_xlim(nx/2-0.5,   nx-0.5)

    ax1.axvline(x=qrt-2.5, linewidth=1, color = 'k', zorder=15)
    ax1.axvline(x=qrt-0.5, linewidth=1, color = 'k', zorder=15)
    ax1.axvline(x=qrt+1.5, linewidth=1, color = 'k', zorder=15)
    ax2.axvline(x=3*qrt-2.5, linewidth=1, color = 'k', zorder=15)
    ax2.axvline(x=3*qrt-0.5, linewidth=1, color = 'k', zorder=15)
    ax2.axvline(x=3*qrt+1.5, linewidth=1, color = 'k', zorder=15)
    ax1.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=15)
    ax2.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=15)

    # Cut and spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelbottom='off', labeltop='off', labelright='off', labelleft='off')

    dd = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-dd, 1+dd), (-dd, +dd), **kwargs)
    ax1.plot((1-dd, 1+dd), (1-dd, 1+dd), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-dd, +dd), (1-dd, 1+dd), **kwargs)
    ax2.plot((-dd, +dd), (-dd, +dd), **kwargs)
#    ax1.set_ylim(-120, 120)
    plt.show()
    fig.savefig('/Users/silver/Desktop/test.pdf',bbox_inches='tight', pad_inches=0.01)

def plt_res_wings(res):
    xx = (np.arange(0,res.NX)-res.NX/2.)*res.resolution+res.resolution/2.
    fig = plt.figure(figsize=(5, 3.75))

#    plt.plot(res.mvx, res.mva, linewidth=1, color='b', alpha=0.5, zorder=1)
#    plt.plot(xx, res.mean, linewidth=2)
    plt.errorbar(xx, res.mean, fmt='.k', yerr=res.sem, zorder=10)
#    plt.plot(xx, res.err/np.sqrt(np.mean(res.count)), linewidth=2, color="r")
    
    print res.ngood, np.mean(res.count), np.sum(res.count)
    plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=11)
    plt.axvline(x=-1, linewidth=1, color = 'k', zorder=11)
    plt.axvline(x=1, linewidth=1, color = 'k', zorder=11)
    plt.xlabel("Planet-centre to star-centre distance (star radius)")
    plt.ylabel("$\Delta F$ (ppm)")

    # Add the prediction
    dat = np.loadtxt('/Users/silver/Dropbox/sci/pro_exo/models/1yr_jovian.dat')
    tt               = dat[:,0]
    zz               = dat[:,1]
    conv_refraction  = dat[:,7]
    conv_fit_mod_ldc = dat[:,9]
    
    mvn = 1000
    mvx = np.linspace(-res.NX/2.*res.resolution, res.NX/2.*res.resolution, mvn)
    mva = np.zeros(mvn)
#    tmp_zz = np.where(res.phase > 0, zz, -zz)
    for ii, mv_pos in enumerate(mvx):
        inside = np.abs(zz - mv_pos) < res.resolution/2.
        if np.sum(inside) == 0:
            mva[ii] = 0.
            continue
        
        mva[ii] = np.mean(conv_refraction[inside]-conv_fit_mod_ldc[inside])

    mva /= 1e-6
    mva = np.where(mvx >= 0, mva, mva[::-1])
    plt.plot(mvx, mva, 'b', lw=2, zorder=4)
    fig.savefig('/Users/silver/Desktop/test.pdf',bbox_inches='tight', pad_inches=0.01)
    plt.show()
#    pdb.set_trace()
