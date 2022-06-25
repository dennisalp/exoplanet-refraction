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

path = "/Users/silver/Dropbox/sci/pro_exo/models/pl3/"
os.chdir(path)

# Select which files
files = sorted(glob('*.dat'))
files = [ x for x in files if 'fit_par' not in x ]
cols = ['g', 'y', 'm', 'r', 'b', 'y', 'y', 'y', 'y', 'y', 'y']
zord = [ 10,  24 ,  5, 20 ,  9,  8]
XMIN = -200.
print(files)
# This is all just a plot
fig = plt.figure(figsize=(5, 3.75))
for ii, ff in enumerate(files):
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
#    print("Parameters      {0:>15}{1:>15}{2:>15}{3:>15}{4:>15}".format(*columns))
#    print("True values:    {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*correct))
#    print("1 min, fix LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod))
#    print("1 min, fit LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod_ldc))
#    print("30 min, fix LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod))
#    print("30 min, fit LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod_ldc))
    
    mid_contact = -griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1, method='linear')
    dF = refraction-fit_mod_ldc
    if np.min(tt - mid_contact) > XMIN:
        tt = np.insert(tt, 0, XMIN+mid_contact)
        dF = np.insert(dF, 0, 0)
    before = tt < mid_contact
        
    plt.semilogy(tt[before]-mid_contact, dF[before]*1e6, lw=2, alpha=1, color=cols[ii], zorder=zord[ii])

    # Find peak amplitude and half time
    peak = np.max(dF)
    pein = np.argmax(dF)
    peti = tt[pein]
    halt = -griddata(dF[:pein], tt[:pein], peak/2., method='linear')
    print(ff[:12], '\t', str(peak*1e6)[:12], '\t', str(halt+tt[pein])[:12])

    
plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=999)
plt.ylabel("$\Delta F$ (ppm)")
plt.xlabel("$t$ (min)")
plt.xlim([XMIN, 0])
plt.ylim([1e-1, 30])
fig.savefig('sample_wings.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
