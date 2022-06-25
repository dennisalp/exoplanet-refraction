# Python 2.X Dennis Alp 2017-02-07
#
# Plots the output of the SAP fitter.

import os
import sys
import pdb
#from datetime import datetime
from glob import glob

#from astropy.io import fits
#from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import griddata
#from scipy.optimize import curve_fit
#from scipy.stats import sem

from exo_sap_cls import *
from exo_sap_con import *
from exo_sap_plt import *
from exo_sap_hlp import *

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


        
################################################################
# Major functions
def get_dat(files, res, vals, meta, conditions):
    for idx, ff in enumerate(files):
        kic, koi = get_ids(ff)
        res.set_ids(vals, meta, kic, koi)
        res.add(idx, ff, vals, meta)

        invalid = False
        for cond in conditions:
            if not cond(res, vals, meta):
                invalid = True
                break
        if invalid:
            res.rm_last()
            continue

        print koi
        res.ngood += 1

    res.finalize()



################################################################
#User input
BATCH_ID = sys.argv[1] #Unique ID to avoid conflict if several scripts run simultaneously.
#Where stuff is saved. ../aux/ is assumed to contain help files (additional data on objects).
WD='/Users/silver/Dropbox/sci/pro_exo/analysis/' + str(BATCH_ID) 

# Parameters
NX = 32
#NX = 48
NY = 256
resolution = 0.125

# Prepare
os.chdir(WD) #Move to designated directory
files = sorted(glob("*.npy")) #Find all files.

vals = np.loadtxt("values_" + str(BATCH_ID) + ".dat", delimiter=",")
meta = Meta('../../aux/candidates.dat')
#res = Residuals(NX, NY, vals)
res = Wings(NX, resolution, vals)

conditions = [test]
#conditions = [sane, chi, acceptance, noise, far, deep] # This makes +-2 ppm
conditions = [sane, noise, far, deep, sample] # This makes +-20 ppm
conditions = [sane, deep, noise]
get_dat(files, res, vals, meta, conditions)
res.mk_map()
res.print_diag()

# Save stuff to files

#plt_res_ppm(res)
plt_res_wings(res)

#4 resppm

#7 resppm

#4 wings
#Mean: 0.25154874216
#Error: (0.38398229024207287, 0.32279047473608635, 0.21853755490672064, 0.69004170786988828)
#chi2: 5.7344687184
#2395 126163.59375 4037235.0

#7 wings
#Mean: 0.306389578754
#Error: (0.46891596501057164, 0.3875451999059627, 0.30865874341436239, 0.95270629119663275)
#chi2: 3.32826988745
#2394 84368.625 2699796.0
