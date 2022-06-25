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

ID = sys.argv[1]
BC = np.load('/Users/silver/Dropbox/sci/pro_exo/models/run/' + ID + '/' + ID + '_bc.npy')
PH = np.load('/Users/silver/Dropbox/sci/pro_exo/models/run/' + ID + '/' + ID + '_ph.npy')

# This is all just a plot
fig = plt.figure(figsize=(5, 3.75))
fmt = ['or', 'sg', 'db', 'pc', 'vm', 'xy']
# Red: Lambda
# Green: Temperature
# Blue: Mass
# Cyan: Radius
# Magenta: Distance
# Yellow: Atm. comp.

for ii in range(0, BC.shape[0]):
    plt.loglog(BC[ii, :, 1], BC[ii, :, 0], fmt[ii])

tmp = np.logspace(2,5,3)
plt.loglog(tmp, np.sqrt(2*tmp/np.pi), 'k')
plt.ylabel("$B$")
plt.xlabel("$C$")
#fig.savefig('/Users/silver/Dropbox/sci/pro_exo/art/figs/run_bc_' + ID + '.pdf', bbox_inches='tight', pad_inches=0.03)

fig = plt.figure(figsize=(5, 3.75))
for ii in range(0, BC.shape[0]):
    plt.loglog(PH[ii, :, 0], PH[ii, :, 1], fmt[ii][1], lw=2)
plt.ylabel("$S_{60}$ (ppm)")
plt.xlabel("Parameter value relative to 80-day super-Earth")
plt.xlim([0.1, 20])
plt.ylim([0.02, 2])
fig.savefig('/Users/silver/Dropbox/sci/pro_exo/art/figs/run_ph_' + ID + '.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
