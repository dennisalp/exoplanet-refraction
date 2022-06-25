from __future__ import print_function, division

import sys
import os
import pdb

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#For LaTeX style font in plots
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

path = "/Users/silver/Dropbox/sci/pro_exo/models/"
os.chdir(path)
path = path + sys.argv[1]
    
dat = np.loadtxt(path + ".dat")
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

par = np.loadtxt(path + "_fit_par.dat")
# Change to impact parameter (fits are made using inclination, tested
# to fit with impact parameter but it doesn't change much if I recall
# correctly)
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

fir_contact = griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1+correct[0], method='linear')
mid_contact = griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1, method='linear')
nir_contact = griddata(zz[int(tt.shape[0]/2):], tt[int(tt.shape[0]/2):], 1-correct[0], method='linear')
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

########
# Working plot
# Short cadence
ax1.plot(tt,(refraction-plain)*1e6,'r')
ax1.plot(tt,(refraction-fit_mod)*1e6,'g')
ax1.plot(tt,(refraction-fit_mod_ldc)*1e6,'b')

ax1.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
ax1.axvline(x=-mid_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax1.axvline(x=mid_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax1.axvline(x=fir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax1.axvline(x=-fir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax1.axvline(x=nir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax1.axvline(x=-nir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')

ax1.set_ylabel("Flux difference (ppm)")
ax1.set_xlabel("Time (min)")
ax1.set_title("1 min cadence, " + path.split("/")[-1].replace('_','\_'))
ax1.legend(["Plain model", "Fitted plain model", "Fitted plain model with 10% freedom in LDC"], loc='best')

# Long cadence
ax2.plot(tt,(conv_refraction-conv_plain)*1e6,'r')
ax2.plot(tt,(conv_refraction-conv_fit_mod)*1e6,'g')
ax2.plot(tt,(conv_refraction-conv_fit_mod_ldc)*1e6,'b')

ax2.axhline(y=0, xmin=0, xmax=1, linewidth=2, color = 'k')
ax2.axvline(x=-mid_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=mid_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=-mid_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=fir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=-fir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=nir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')
ax2.axvline(x=-nir_contact, ymin=0, ymax=1, linewidth=2, color = 'k')

ax2.set_ylabel("Flux difference (ppm)")
ax2.set_xlabel("Time (min)")
ax2.set_ylim(ymin=np.min(refraction-plain)*1.5e6)
ax2.set_title("30 min cadence, " + path.split("/")[-1].replace('_','\_'))
ax2.legend(["Plain model", "Fitted plain model", "Fitted plain model with 10% freedom in LDC"], loc='best')

########
# Product plot
fig = plt.figure(figsize=(5, 3.75))
plt.plot(tt,(refraction-plain)*1e6,'r-.', lw=2)
plt.plot(tt,(refraction-fit_mod)*1e6,'g--', lw=2)
plt.plot(tt,(refraction-fit_mod_ldc)*1e6,'b', lw=2)

plt.axhline(y=0, xmin=0, xmax=1, linewidth=1, color = 'k')
plt.axvline(x=-mid_contact, ymin=0, ymax=1, linewidth=1, color = 'k')
plt.axvline(x=mid_contact, ymin=0, ymax=1, linewidth=1, color = 'k')
plt.axvline(x=fir_contact, ymin=0, ymax=1, linewidth=1, color = 'k')
plt.axvline(x=-fir_contact, ymin=0, ymax=1, linewidth=1, color = 'k')
plt.axvline(x=nir_contact, ymin=0, ymax=1, linewidth=1, color = 'k')
plt.axvline(x=-nir_contact, ymin=0, ymax=1, linewidth=1, color = 'k')

plt.ylabel("$\Delta F$ (ppm)")
plt.xlabel("$t$ (min)")
plt.xlim([np.amin(tt), np.amax(tt)])
fig.savefig(sys.argv[1] + '_fit_res.pdf', bbox_inches='tight', pad_inches=0.03)

plt.show()
