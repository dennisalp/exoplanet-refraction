from __future__ import print_function, division

import sys
import os
import pdb
from glob import glob

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

ID = ['20-day Jovian', '80-day super-Earth', 'Earth', '1-year Jovian', 'Jupiter', 'Best-case Jovian']
BB = np.array([5.76354696267, 7302.48539363, 23047.5098301, 170.814201535, 3058.65030192, 350.889274137])
CC = np.array([545.584184354, 1345.85600156, 858.741840300431, 1436.76523849, 3276.3288331373788, 720.7921126367476])

# This is all just a plot
fig = plt.figure(figsize=(5, 3.75))
for ii, label in enumerate(ID):
    plt.semilogy(CC[ii], BB[ii], '.k')
    if label == 'Best-case jovianDISABLE':
        plt.annotate(label, xy=[CC[ii]*1.105, BB[ii]*0.9])
    else:
        plt.annotate(label, xy=[CC[ii]+60, BB[ii]*1.04])

tmp = np.logspace(-4,4,1000)
plt.semilogy(tmp, np.sqrt(2*tmp/np.pi), 'k')
plt.ylabel("$B$")
plt.xlabel("$C$")
plt.xlim([0, 4e3])
plt.ylim([1e0, 1e5])
#plt.gca().xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
fig.savefig('/Users/silver/Dropbox/sci/pro_exo/art/figs/bcp.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
