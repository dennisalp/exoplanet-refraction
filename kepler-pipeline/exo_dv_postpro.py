import matplotlib.pyplot as plt
import numpy as np
import sys

from glob import glob

from exop_dv_classes import *

WD = sys.argv[1] #Give path to dir with parent to values-files
files=glob(WD + '/analysis_?/values_?.dat') #Find all files. 
ofiles=glob(WD + '/analysis_?/output_?.dat') #Find all files. 
meta = Meta(WD + '/aux/confirmed.txt')
G = 6.673e-11
Rs = 695700000 #Sun radius
Ms = 1.989e30 #Sun mass
SIG = 5.67e-8 #S-B constant
KB = 1.38064852e-23 #Boltzmann constant
Mu = 1.660539040e-27 #Atomic mass
MHe = 4.002602 #Helium mass
MH2 = 2.01588 #Molecular hydrogen mass

dat = []
for i in range(0,len(files)):
    dat.append(np.loadtxt(files[i], delimiter=','))

dat = np.vstack(dat)
n = dat.shape[0]

koi = dat[:,0]
alpha = dat[:,1]
prad = dat[:,2]
nea_prad = dat[:,3]
dv_prad = dat[:,4]
inclin = dat[:,5]
nea_inclin = dat[:,6]
dv_inclin = dat[:,7]
dor = dat[:,8]
nea_dor = dat[:,9]
dv_dor = dat[:,10]
g1 = dat[:,11]
h1 = dat[:,12]
g2 = dat[:,13]
h2 = dat[:,14]
chi = dat[:,15]
xi5 = dat[:,16]
xi30 = dat[:,17]
zeta5 =  np.log10(1./dat[:,18])
zeta10 = np.log10(1./dat[:,19])
zeta30 = np.log10(1./dat[:,20])
gfit = dat[:,21]
f1 = dat[:,22]
f2 = dat[:,23]
err = dat[:,24]

#THIS IS MY MISTAKE, SAVE TDUR NEXT TIME
print files, ofiles
tdur = np.zeros(n)
counter = 0
for i in range(0,len(ofiles)):
    for line in file(ofiles[i]):
        if 'Transit duration' in line:
            tdur[counter] = float(line[18:29])
            counter += 1

#########
#Help
def get_good():
    idx = np.where((xi30 > 1000) & (prad < 11.306))[0]
    idx = np.where((zeta30 > 6) & (prad > 0.))[0] 
#    idx = np.where((my_heur > 3e3) & (prad < 0.5) & (incl2b(dor, inclin) < 1.1))[0] #tgheur
#    idx = np.where(caustics > 1.) #caustics
#    idx = np.where(cheur > 1e4 ) #snr
#    idx = np.where((snr_correct > 0.03)  & (prad < 0.5) & (incl2b(dor, inclin) < 1.1)) #Above is more heuristic, this is more how it's supposed to be done
    return idx

def gen_list(idx):
    good =  koi[idx]
    print good.size
    for g in good:
        print "ln -s /home/dalp/data/kepler/koi/koi_?/*" + str(g)[:-4] +"* /home/dalp/data/kepler/koi/koi_snr_correct/"

def get_mind():
    mind = np.zeros(koi.size)
    for ind in range(0,koi.size):
        mind[ind] = int(np.where(abs(meta.kid+np.mod(meta.koi,1)-0.01*koi[ind]) < 1e-6)[0][0])
    return list(mind)

def parse_args(args):
    if len(args) > 0:
        return args[0]
    else:
        return np.array(range(0,n))

def incl2b(r, i):
    return r*np.cos(np.deg2rad(i))

#########
#Plots
def rad_xi(*args):
    i = parse_args(args)
    plt.scatter(prad[i], xi5[i], alpha=0.5)
    plt.scatter(prad[i], xi30[i], alpha=0.5, color='r')
    plt.legend(['5 min','30 min'])
    plt.show()

def rad_zeta(*args):
    i = parse_args(args)
    plt.scatter(prad[i], zeta5[i], alpha=0.5, color='r')
    plt.scatter(prad[i], zeta10[i], alpha=0.5, color='g')
    plt.scatter(prad[i], zeta30[i], alpha=0.5, color='b')
    plt.legend(['5 min','10 min','30 min'])
    plt.show()

def hist_xi(*args):
    i = parse_args(args)
    low = int(np.log10(min(xi5[i][np.where(xi5[i] > 0)[0]])))
    high = int(np.log10(max(xi30[i])))+1
    bins = np.logspace(low,high,100)
    plt.hist(xi5[i], bins=bins, alpha=0.5)
    plt.hist(xi30[i], bins=bins, alpha=0.5, color='r')
    plt.legend(['5 min','30 min'])
    plt.xscale('log')
    plt.show()

def hist_zeta(*args):
    i = parse_args(args)
    low = int(np.log10(min(xi5[i][np.where(xi5[i] > 0)[0]])))
    high = int(np.log10(max(xi30[i])))+1
    bins = np.logspace(low,high,100)
    gd = np.where(-np.isinf(zeta5[i]))[0]
    plt.hist(zeta5[i][gd], bins=200, alpha=0.5, color='r')
    gd = np.where(-np.isinf(zeta10[i]))[0]
    plt.hist(zeta10[i][gd], bins=200, alpha=0.5, color='g')
    gd = np.where(-np.isinf(zeta30[i]))[0]
    plt.hist(zeta30[i][gd], bins=200, alpha=0.5, color='b')
    plt.legend(['5 min', '10 min', '30 min'])
    plt.xlabel('log(Photometric precision)')
    plt.show()

def hist_prad(*args):
    i = parse_args(args)
    high = max(np.concatenate((prad, nea_prad, dv_prad)))
    bins=np.linspace(0,2,400)
    plt.hist(prad[i], bins=bins, alpha=0.5)
    plt.hist(nea_prad[i], bins=bins, alpha=0.5, color='r')
    plt.hist(dv_prad[i], bins=bins, alpha=0.5, color='g')
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.xlabel('Radius')
    plt.show()

def hist_dor(*args):
    i = parse_args(args)
    high = max(np.concatenate((dor, nea_dor, dv_dor)))
    bins=np.linspace(0,high,500)
    plt.hist(dor[i], bins=bins, alpha=0.5)
    plt.hist(nea_dor[i], bins=bins, alpha=0.5, color='r')
    plt.hist(dv_dor[i], bins=bins, alpha=0.5, color='g')
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.xlabel('Radius')
    plt.show()

def hist_b(*args):
    i = parse_args(args)
    b = incl2b(dor[i], inclin[i])
    nea_b = incl2b(nea_dor[i], nea_inclin[i])
    dv_b = incl2b(dv_dor[i], dv_inclin[i])
    high = max(np.concatenate((b, nea_b, dv_b)))
    bins=np.linspace(0,2,40)
    plt.hist(b, bins=bins, alpha=0.5)
    plt.hist(nea_b, bins=bins, alpha=0.5, color='r')
    plt.hist(dv_b, bins=bins, alpha=0.5, color='g')
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.xlabel('Impact parameter')
    plt.show()

def hist_inclin():
    bins=np.linspace(0,90,501)
    plt.hist(inclin, bins=bins, alpha=0.5)
    plt.hist(nea_inclin, bins=bins, alpha=0.5, color='r')
    plt.hist(dv_inclin, bins=bins, alpha=0.5, color='g')
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.show()

def nea_dv():
    plt.loglog(nea_prad,dv_prad, '.', alpha = 0.3)
    plt.loglog(nea_inclin,dv_inclin, '.r', alpha = 0.3)
    plt.loglog(nea_dor,dv_dor, '.g', alpha = 0.3)
    plt.legend(['Radius', 'Inclination', 'Distance'])
    plt.show()

def rad_b(*args):
    i = parse_args(args)
    b = incl2b(dor[i], inclin[i])
    nea_b = incl2b(nea_dor[i], nea_inclin[i])
    dv_b = incl2b(dv_dor[i], dv_inclin[i])
    plt.scatter(prad[i], b, alpha = 0.3)
    plt.scatter(nea_prad[i], nea_b, color='r', alpha = 0.3)
    plt.scatter(dv_prad[i], dv_b, color='g', alpha = 0.3)
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.xlabel('Radius')
    plt.ylabel('Impact parameter')
    plt.show()

def smass(*args):
    i = parse_args(args)
    mind = get_mind()

    r = dor * meta.srad[mind] * Rs
    t = meta.tperiod[mind]*24*3600
    v = 2*np.pi*r/t
    M = v**2/(2.*G)*r

    r = nea_dor * meta.srad[mind] * Rs
    v = 2*np.pi*r/t
    M_nea = v**2/(2.*G)*r
    M_check = meta.smass[mind]*Ms
    
    r = dv_dor * meta.srad[mind] * Rs
    v = 2*np.pi*r/t
    M_dv = v**2/(2.*G)*r

    plt.scatter(M_check[i], M[i], alpha = 0.3)
    plt.scatter(M_check[i], M_nea[i], color='r', alpha = 0.3)
    plt.scatter(M_check[i], M_dv[i], color='g', alpha = 0.3)
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.plot([1e29, 1e31], [1.00001e29,0.9999e31], 'k')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('NEA Stellar mass')
    plt.ylabel('Fitted mass')
    plt.show()    

    plt.hist(np.log10(M), bins=100, alpha=0.5)
    plt.hist(np.log10(M_nea), bins=100, alpha=0.5, color='r')
    plt.hist(np.log10(M_dv), bins=100, alpha=0.5, color='g')
    plt.legend(['Fit', 'NASA', 'DV'])
    plt.xlabel('Fitted mass')
    plt.show()

def ldc(*args):
    i = parse_args(args)
    plt.scatter(g1,g2,color='r')    
    plt.scatter(g1[i],g2[i])
    plt.show()
    print np.mean(g1[i]), np.mean(g1)
    print np.mean(g2[i]), np.mean(g2)

def goodness(*args):
    i = parse_args(args)
    global zeta30 
    zeta30 = np.where(abs(zeta30) == np.inf, 0, zeta30)
    plt.hist(my_heur[i], bins=100)
    plt.show()

def snr(*args):
    i = parse_args(args)
    x = np.logspace(-1,5,100)
    caustic = np.sqrt(2*x/np.pi)
    
    plt.loglog(R0H, B,'.')
#    plt.loglog(R0H[i], B[i],'.r', markersize=10)
    plt.loglog(x, caustic, 'k', linewidth=2)
    plt.show()

def snrc(*args):
    i = parse_args(args)
    bis = np.logspace(-6,np.log10(np.amax(snr_correct[i])),60)
    plt.hist(snr_correct[i],bins=bis)
    plt.gca().set_xscale('log')
    plt.xlabel('Signal to noise')
    plt.show()


#########
#Some theory
mind = get_mind()
pd = dor * meta.srad[mind] * Rs #Orbital distance
pr = prad * meta.srad[mind] * Rs #Planet radius
sT = meta.steff[mind] #Stellar temperature
sr = meta.srad[mind] * Rs #Stellar radius
sP = SIG*sT**4*4*np.pi*sr**2 #Emitted power
pP = sP/(4*np.pi*pd**2)*np.pi*pr**2 #Received power (Also emitted)
pT = (pP/(SIG*4*np.pi*pr**2))**0.25 #Planet temperature
prho = 10**np.polyval(np.array([-0.59336486,  7.77852929]), np.log10(pr)) #Earth radius logarithmic interpolation
pM = 4*np.pi*pr**3/3.*prho #Planet mass
pg = G*pM/pr**2 #Planet gravity
mu = 0.24*MH2/MHe*MHe+(1-0.24*MH2/MHe)*MH2 #24% Helium by mass, rest H2
H = KB*pT/(pg*mu*Mu)
a = 0.001214 #Refractive coefficient at 6700 A
B = 2*a*3e-3/H*pd #Assuming observer-lens distance to be large compared to source-lens
#3e-3 kg/m3 is density of atmosphere at optical depth 1, pulled out of a hat but close to some planets.
R0H = pr/H

tingr = 2*prad*tdur/(1+2*prad)
my_heur = prad**2*10**zeta30*dor*tdur
caustics = B*np.sqrt(0.5*np.pi/R0H)
cheur = caustics*10**zeta30*tdur
#plt.hist(tingr)
#plt.show()
snr_correct = caustics*1e-5*10**zeta30*np.where(tingr/(29.4243814323648/60./24.) > 1, 1, tingr/(29.4243814323648/60./24.))*prad**2/0.01

#########
#Main
idx = get_good()
print np.c_[koi[idx],snr_correct[idx]]
mind = get_mind()

#print np.mean(g1[idx]), np.mean(g2[idx])
gen_list(idx)
#goodness()
#goodness(idx)
#snr()
snrc()
#snr(idx)
##ldc(idx)
##rad_xi(idx)
#rad_zeta(idx)
#hist_prad(idx)
#hist_dor(idx)
#hist_b(idx)


#rad_zeta()

##hist_prad()
#hist_prad()
#hist_dor()
#hist_b()

#hist_xi()
#hist_inclin()
#nea_dv()
#rad_b(idx)
#smass()
#hist_zeta()

#########
#For comparing chi2 distributions
#files=glob(WD + '/analysis_?_unpurged/values_?.dat') #Find all files. 
#meta = Meta(WD + '/aux/meta_data.txt')
#dat = []
#for i in range(0,len(files)):
#    dat.append(np.loadtxt(files[i], delimiter=','))
#dat = np.vstack(dat)
#n = dat.shape[0]
#chi2 = dat[:,15]
#bins = np.logspace(-5,0.3,1001)
#plt.hist(1/chi, bins=bins, alpha=0.5)
#plt.hist(1/chi2,bins=bins, alpha=0.5)
#plt.show()
#print np.mean(chi), np.mean(chi2)
