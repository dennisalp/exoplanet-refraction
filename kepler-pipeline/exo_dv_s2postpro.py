import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.stats import sem

dat = []
for i in range(1,len(sys.argv)):
    dat.append(np.loadtxt(sys.argv[i], delimiter=','))

dat = np.vstack(dat)
n = dat.shape[0]

#dat = np.loadtxt(sys.argv[1], delimiter=',')

t    = dat[:,0]
a    = dat[:,1]
delT = abs(dat[:,2])
delA = abs(dat[:,3])
std  = dat[:,4]
chi2nor      = dat[:,5]
chi2ref      = dat[:,6]
delchi2      = dat[:,7]
del_chi2ref  = dat[:,8]  
del_delchi2  = dat[:,9]  
chi2nor_red  = dat[:,10]  
chi2ref_red  = dat[:,11]  
del2chi2_red = dat[:,12] 
del_chi2ref_red = dat[:,13]
del_delchi2_red = dat[:,14]
pppv_delT_nor   = abs(dat[:,15])
pppv_delA_nor   = abs(dat[:,16])
pppv_delT       = abs(dat[:,17])
pppv_delA       = abs(dat[:,18])
pppv_chi2nor    = dat[:,19]
pppv_chi2ref    = dat[:,20]
pppv_delchi2    = dat[:,21]
pppv_chi2nor_red= dat[:,22]
pppv_chi2ref_red= dat[:,23]
pppv_delchi2_red= dat[:,24]
nedge             = dat[:,25]
pppv_nedge        = dat[:,26]
zeta30            = dat[:,27]
koi               = dat[:,28]
gauss_chi2ref     = dat[:,29]
gauss_delchi2     = dat[:,30]
gauss_chi2ref_red = dat[:,31]
gauss_delchi2_red = dat[:,32]
gauss_delT        = dat[:,33]
gauss_delA        = dat[:,34]
prad              = dat[:,35]
inclin            = dat[:,36]
dor               = dat[:,37]

def purge(x):
    return np.where(np.isnan(x), 0., x)

def delta_refraction_data_noise():
    bis = np.linspace(-0.1,0.6,200)
#    bis = np.linspace(-1,1,500)
    plt.hist(del_delchi2_red, bins=bis, alpha=0.5)
    plt.hist(purge(pppv_delchi2_red), bins=bis, alpha=0.5)
    plt.hist(purge(gauss_delchi2_red), bins=bis, alpha=0.5)
    plt.show()

def hist_delchi2():
    plt.hist(delchi2,200)
    plt.show()
    gen_list(np.where(delchi2 > 10)[0])

def delta_refraction_gauss_data():
    bis = np.linspace(-0.1,0.6,200)
#    bis = np.linspace(-1,1,500)
    plt.hist(del_delchi2_red, bins=bis, alpha=0.5)
    plt.hist(purge(gauss_delchi2_red), bins=bis, alpha=0.5)
    plt.show()

def A_data_noise():
    bis = np.linspace(0.,0.03,200)
    bis = np.logspace(-8,np.log10(0.2),200)
    plt.hist(delA, bins=bis, alpha=0.5)
    plt.hist(pppv_delA, bins=bis, alpha=0.5)
    plt.gca().set_xscale('log')
    plt.show()

def A_signal():
    bis = np.linspace(-0.01,0.01,200)
#    bis = np.logspace(-8,np.log10(0.2),200)
    signal = delA-pppv_delA
    g = np.where(abs(signal) <0.01)
    plt.hist(signal[g], 100, alpha=0.5)
    print np.mean(signal[g]), sem(signal[g])
#    plt.gca().set_xscale('log')
    plt.show()

def BIC_refraction():
    BIC = chi2nor + 5*np.log(nedge)
    BIC_delta = chi2ref + 7*np.log(nedge)
    plt.hist(BIC-BIC_delta,400)
    plt.xlabel('Delta BIC (with/without refraction)')
    plt.show()
    gd = np.where(BIC-BIC_delta > 2)[0]
    print np.c_[koi[gd], (BIC-BIC_delta)[gd]]

def gen_list(idx):
    good =  koi[idx]
    print good.size
    for g in good:
        print "ln -s /home/dalp/data/kepler/koi/koi_?/*" + str(g)[:-4] +"* /home/dalp/data/kepler/koi/koi_chi2/"


#delta_refraction_data_noise()
hist_delchi2()
#delta_refraction_gauss_data()
#A_data_noise()
#A_signal()
#BIC_refraction()

#plt.hist(np.where(abs(delT) > 5, 5, abs(delT)),100, alpha=0.5)
#plt.hist(np.where(abs(pppv_delT) > 5, 5, abs(pppv_delT)),100, alpha=0.5)
#plt.show()
