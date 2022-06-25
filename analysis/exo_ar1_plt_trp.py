from __future__ import print_function, division

import sys
import os
import pdb
from glob import glob

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

path = "/Users/silver/Dropbox/sci/pro_exo/models/trp/"
os.chdir(path)

# Select which files
files = sorted(glob('*.dat'))
files = [ x for x in files if 'fit_par' not in x and 'hirs' not in x]
cols = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'y', 'y', 'y']
zord = [ 9,   8 ,   7 ,  6,  5, 4, 3,2,1]
XMIN = -60.

# This is all just a plot
fig = plt.figure(figsize=(5, 3.75))
for ii, ff in enumerate(files):
#    if ii < 4 and not ii==2:
#        continue
    dat = np.loadtxt(ff)
    tt               = dat[:,0]
    zz               = dat[:,1]
    plain            = dat[:,2]
    refraction       = dat[:,3]
    fit_mod          = dat[:,4]
    fit_mod_ldc      = dat[:,5]
    conv_plain       = dat[:,6]
    conv_refraction  = dat[:,7]
    conv_fit_mod     = dat[:,8]
    conv_fit_mod_ldc = dat[:,9]

    par = np.loadtxt(ff[:-4] + "_fit_par.dat")
    par[:,1] = par[:,2]*np.cos(np.deg2rad(par[:,1])) 
    correct              = par[0,:]
    par_fit_mod          = par[1,:]
    par_fit_mod_ldc      = par[2,:]
    par_conv_fit_mod     = par[3,:]
    par_conv_fit_mod_ldc = par[4,:]
    
    columns = ["Radius", "Impact param.", "Distance", "g1 factor", "g2 factor"]
#    print("Parameters      {0:>15}{1:>15}{2:>15}{3:>15}{4:>15}".format(*columns))
#    print("True values:    {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*correct))
#    print("1 min, fix LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod))
#    print("1 min, fit LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod_ldc))
#    print("30 min, fix LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod))
#    print("30 min, fit LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod_ldc))

    mid_contact = -griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1, method='linear')
    dF = refraction-fit_mod_ldc
    if np.min(tt - mid_contact) > XMIN:
        print(XMIN)
        tt = np.insert(tt, 0, XMIN+mid_contact)
        dF = np.insert(dF, 0, 0)
    before = tt < mid_contact

    # Interpolate for nicer plots
    tmin = np.min(tt[before]-mid_contact)
    tmax = np.max(tt[before]-mid_contact)
    ttt  = np.linspace(tmin, tmax, 100000)
    dF = griddata(tt[before]-mid_contact, dF[before]*1e6, ttt, method='linear')
    plt.semilogy(ttt, dF, lw=2, alpha=1, color=cols[ii], zorder=zord[ii])

    # Find peak amplitude and half time
    peak = np.max(dF)
    pein = np.argmax(dF)
    peti = ttt[pein]
    halt = -griddata(dF[:pein], ttt[:pein], peak/2., method='linear')
    print(ff, '\t', peak, '\t', halt+ttt[pein])

    
plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=999)
plt.ylabel("$\Delta F$ (ppm)")
plt.xlabel("$t$ (min)")
plt.xlim([XMIN, 0])
plt.ylim([1e0, 50])
fig.savefig('trp.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
