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

path = "/Users/silver/Dropbox/sci/pro_exo/models/tab_par/"
os.chdir(path)

# Select which files
files = sorted(glob('*.dat'))
files = [ x for x in files if 'fit_par' not in x ]

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
    
    dF = refraction-fit_mod_ldc
    max_amp = np.amax(dF)
    jj = 0
    while dF[jj] < max_amp/2.:
        jj += 1

    t_half = griddata(np.array([dF[jj-1], dF[jj]]), np.array([jj-1, jj]), max_amp/2)
    t_half = np.argmax(dF)-t_half

    print('{0:<20}& {1:>15.8} & {2:>15.8}\\\\'.format(ff, max_amp*1e6, t_half))
