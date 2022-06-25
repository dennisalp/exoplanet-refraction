import numpy as np
from scipy.interpolate import griddata
from scipy.stats import sem

#########
#Help functions
def chi2red(y1, y2, err):
    return np.sum(((y1-y2)/err)**2)/len(y1)

def chi2(y1, y2, err):
    return np.sum(((y1-y2)/err)**2)

def double_print(output, out_str):
    output.write(out_str)
    print(out_str.replace('\n',''))

def calc_tdur(dat):
    if min(dat.ex_mod) == 0:
        return 0.
    ingress = np.where(dat.ex_mod < 0)[0][0]
    oot = ingress - 1 if ingress > 0 else ingress
    sol = abs(np.mean(dat.ex_pha[oot:ingress+1]))
    return sol*2

def calc_tingr(dat):
    if min(dat.ex_mod) == 0:
        return 0.

    s1 = np.where(dat.z < 1+dat.fit_par[0])[0]
    if s1.size == 0:
        return 0.
    s1 = s1[0]

    e1 = np.where(dat.z < 1-dat.fit_par[0])[0]
    if e1.size == 0:
        return 0.
    e1  = e1[0]

    s0 = s1 - 1 if s1 > 0 else s1
    e0 = e1 - 1 if e1 > 0 else e1
    s = griddata(dat.z[s0:s1+1], dat.phase[s0:s1+1], 1+dat.fit_par[0], method='linear')
    e = griddata(dat.z[e0:e1+1], dat.phase[e0:e1+1], 1-dat.fit_par[0], method='linear')
    return e-s

def sanity_check(dat, guess, GFIT):
    dat.fit_par[0] = abs(dat.fit_par[0]) #Negative radii are treated as positive in occult anyway.
    dat.fit_par[1] = np.mod(abs(dat.fit_par[1]), 180) #Sign never matters, also, cos symmetric
    dat.fit_par[1] = 180-dat.fit_par[1] if dat.fit_par[1] > 90 else dat.fit_par[1]
    par = dat.fit_par.copy()
    
    if par[0] < 10 and par[0] > 1e-5:
        if par[2] > 0 and par[2] < 500:
            if poor_mans_bound(par[3], GFIT)*dat.gamma1 + poor_mans_bound(par[4], GFIT)*dat.gamma2 <= 1:
                return True

    dat.fit_par = guess
    return False

def poor_mans_bound(x, GFIT):
    return np.arctan(x)*GFIT+1.

def inclin2b(inclin, dor):
    return 10**(dor*np.cos(np.deg2rad(inclin)))

def b2inclin(b, dor):
    b = np.log10(abs(b))
#    print b, dor
    return np.rad2deg(np.arccos(b/dor))

def tbin(t, f, kernel, *args):
    if len(args) > 0:
        w = args[0]
    else:
        w = np.ones(t.size)

    span = max(t)-min(t)
    pts = int(span/kernel)

    tres, fres, err, err2 = np.zeros(pts), np.zeros(pts), np.zeros(pts), np.zeros(pts)
    tres = kernel*np.linspace(0.5,pts-0.5,pts)+min(t)
    for i in range(0,pts):
        include = np.where(abs(t-tres[i]) < kernel/2.)[0]
        err[i] = np.mean(1./w[include])/np.sqrt(include.size)
#        print err[i]-np.mean(1./w[include])/np.sqrt(include.size)
#        err[i]  = np.sqrt(np.sum((1./w[include])**2))/include.size
        err2[i]  = sem(f[include])
        fres[i] = np.sum(w[include]*f[include])/np.sum(w[include])
        tres[i] = np.sum(w[include]*t[include])/np.sum(w[include])
    return tres, fres, err, err2
