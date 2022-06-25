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

path = "/Users/silver/Dropbox/sci/pro_exo/models/pl2"
name_root = sys.argv[1] + '_'
os.chdir(path)

# Select which files
files = sorted(glob(name_root + '*.dat'))
files = [ x for x in files if 'fit_par' not in x ]
files = files[2:]
cols = ['g', 'y', 'r', 'c', 'b', 'm','y','y','y','y']
zord = [ 10,  4 ,  6 ,  17,  8, 0]
print(files)

# This is all just a plot
fig = plt.figure(figsize=(5, 3.75))
for ii, ff in enumerate(files):
    print(ff)
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
    print("Parameters      {0:>15}{1:>15}{2:>15}{3:>15}{4:>15}".format(*columns))
    print("True values:    {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*correct))
    print("1 min, fix LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod))
    print("1 min, fit LDC: {0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_fit_mod_ldc))
    print("30 min, fix LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod))
    print("30 min, fit LDC:{0:>15.8}{1:>15.8}{2:>15.8}{3:>15.8}{4:>15.8}".format(*par_conv_fit_mod_ldc))
    
    plt.plot(tt,(refraction-fit_mod_ldc)*1e6, lw=2, alpha=1, color=cols[ii], zorder=zord[ii])
#    plt.plot(tt,(refraction-plain)*1e6, lw=2, alpha=1, color=cols[ii], zorder=zord[ii])

mid_contact = griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1, method='linear')
    
plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k', zorder=999)
plt.axvline(x=-mid_contact, ymin=0, ymax=1, linewidth=1, color = 'k', zorder=999)
plt.axvline(x=mid_contact, ymin=0, ymax=1, linewidth=1, color = 'k', zorder=999)

plt.ylabel("$\Delta F$ (ppm)")
plt.xlabel("$t$ (min)")
plt.xlim([np.amin(tt), np.amax(tt)])
fig.savefig(sys.argv[1] + '_param_space.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
