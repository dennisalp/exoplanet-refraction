import matplotlib.pyplot as plt
import numpy as np

from exop_trans_occultquad import occultquad
from scipy.optimize import curve_fit

def transit_model(z, g1, g2, prad, b, time): 
#    time = 1.5
#    prad = 0.1
#    g1 = 0.381445454545
#    g2 = 0.269718181818
    z = np.linspace(0, time, z.size)
    z = zin*time
    z = np.sqrt(b**2+z**2)
    quad, lin = occultquad(abs(z), g1, g2, prad) #Compute model
    return quad

def fit(zin, f, guess):
    pars, covar = curve_fit(transit_model, zin, f, guess)
    return pars

def plot_res(zin, f, pars):
    g1, g2, prad, b, time = pars
    mod = transit_model(zin, g1, g2, prad, b, time)
    plt.plot(zin, f, linewidth=4)
    plt.plot(zin, mod)
    plt.show()

    normed = f-mod
    normed=np.concatenate((normed[1:][::-1],normed))
    zhe=np.concatenate((-zin[1:][::-1],zin))
    plt.plot(zhe, normed, linewidth=4)
    plt.xlabel('Planet-star distance [Star radius]')
    plt.ylabel('Difference in normalized flux')
    print np.amax(abs(f-mod))

    sav = np.where(abs(zin-1) < 0.3)[0]
    normed = (f-mod)/np.amax(abs(f-mod))
#    np.savetxt(delta, np.c_[zin[sav], normed[sav]])
    plt.show()
    
dat = np.loadtxt('/home/dalp/Dropbox/astrophysics/project_exoplanets/models/sandbox.dat')
#delta = open('/home/dalp/Dropbox/astrophysics/project_exoplanets/models/delta_sandbox.dat','w')
#dat = np.loadtxt('/home/dalp/Dropbox/astrophysics/project_exoplanets/models/sun.dat')
#delta = open('/home/dalp/Dropbox/astrophysics/project_exoplanets/models/delta_sun.dat','w')
zin, f = dat[:,0], dat[:,1]
guess = np.array([0.35, 0.225, 0.01, 0., 1.]) #last is rescaling of time
pars = fit(zin, f, guess)
print pars
plot_res(zin, f, pars)
