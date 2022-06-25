import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from glob import glob
from scipy.interpolate import griddata

from exo_dv_helper import *

#########
#Classes
#Holds all data on a planet candidate
class Data():
    meta_ind = None #Row index in the meta file
    kid = None #Kepler ID
    koi = None #KOI ID, only unique identifier!
    tdur = None #Transit duration
    prad = None #Planet radius in star radii
    tperiod = None #Orbital period
    tepoch = None #Transit epoch in BJD-2454833 [days]
    inclin = None #Inclination
    dor = None #Star to planet distance over star radius
    ntrans = None #Number of transits

    #Stellar data are from meta file, conveniently prefixed with 's'
    steff = None #Host star effective temperature
    slogg = None #Logarithm of surface gravity at host star surface
    smet = None #Metallicity 

    #Data from dv .fits files prefixed with dv for comparison
    dvtdur = None 
    dvprad = None
    dvinclin = None
    dvdor = None
    dvntrans = None
    gamma1 = None #The polynomial coefficients at points, see e.g. Mandel & Agol (2002)
    gamma2 = None
    fit_par = None #Fitted parameter values

    #Data straight from meta file, will be renormalized after chopping
    time = None #Time in BJD-2454833 [days]
    phase = None #Phase in days
    flux = None #Flux with stellar flux normalized to 0 
    ferr = None #Error in flux

    #After stack-convolution of data
    conv_phase = None #Phase of data
    conv_res = None #Residuals of data

    #"Exact" in the sense that they are calculated from theory
    z = None #z defined following Mandel & Agol (2002)
    ex_pha = None #Phase for model
    ex_mod = None #Modelled flux values
    instr_mod = None #Modelled flux after fake instrument binning

    #For analysis of convergence
    binsize = None #Width of flux bins in minutes
    std = None #Standard deviation of data points when binned with binsize

    def __init__(self, meta, system, tce, meta_ind):
        def is_ok(val):
            return not (np.isnan(val) or val < 1)

        self.meta_ind = meta_ind
        self.kid = int(system[-33:-24].lstrip('0'))
        self.koi = str(self.kid) + '_' + str(meta.koi[self.meta_ind])[-2:]
        self.tdur = meta.tdur[self.meta_ind]
        self.prad = meta.prad[self.meta_ind]
        self.tperiod = meta.tperiod[self.meta_ind]
        self.tepoch = meta.tepoch[self.meta_ind]
        self.inclin = meta.inclin[self.meta_ind]
        self.dor = meta.dor[self.meta_ind]
        self.ntrans = meta.ntrans[self.meta_ind] if is_ok(meta.ntrans[self.meta_ind]) else float(fits.getval(system,'NTRANS',tce)) if is_ok(fits.getval(system,'NTRANS',tce)) else 1
        self.steff = meta.steff[self.meta_ind]
        self.slogg = meta.slogg[self.meta_ind]
        self.smet = meta.smet[self.meta_ind]

        self.dvtdur = fits.getval(system,'TDUR',tce)/24.
        self.dvprad = fits.getval(system,'RADRATIO',tce)
        self.dvprad = self.dvprad if type(self.dvprad) == type(0.) else self.prad
        self.dvinclin = fits.getval(system,'INCLIN',tce)
        self.dvinclin = self.dvinclin if type(self.dvinclin) == type(0.) else self.inclin
        self.dvdor = fits.getval(system,'DRRATIO',tce)
        self.dvdor = self.dvdor if type(self.dvdor) == type(0.) else self.dor
        self.dvntrans = float(fits.getval(system,'NTRANS',tce)) if not np.isnan(float(fits.getval(system,'NTRANS',tce))) else 1
        self.gamma1, self.gamma2 = meta.gamma1[self.meta_ind], meta.gamma2[self.meta_ind] #Claret & Bloemen (2011)

        tab = Table.read(system, hdu=tce)
        self.time = np.array(tab['TIME'])
        self.phase = np.array(tab['PHASE'])
        self.flux = np.array(tab['LC_INIT'])
        self.ferr = np.array(tab['LC_INIT_ERR'])
        
        self.binsize = []
        self.std = []
            
    def upt_lists(self, ind):
        self.time = self.time[ind]
        self.phase = self.phase[ind]
        self.flux = self.flux[ind]
        self.ferr = self.ferr[ind]

    def set_conv(self, conv_phase, conv_res):
        self.conv_phase = conv_phase
        self.conv_res = conv_res

#Contains meta data on the candidates
class Meta():
    kid = None #Kepler ID of all KOI candidates
    koi = None #KOI ID of all KOI candidates, only unique identifier!
    tperiod = None #Orbital period
    tepoch = None #Transit epoch in BJD-2454833 [days]
    tdur = None #Transit duration, hours converted to days
    prad = None #Planet radius in star radii
    inclin = None #Inclination
    dor = None #Star to planet distance over star radius
    ntrans = None #Number of transits
    steff = None #Host star effective temperature
    slogg = None #Logarithm of surface gravity at host star surface
    smet = None #Metallicity of host star
    srad = None #Radius of star in solar radii
    smass = None #Mass of star in solar masses
    gamma1 = None #The polynomial coefficients at points, see e.g. Mandel & Agol (2002)
    gamma2 = None

    def __init__(self, f):
        meta_table = np.loadtxt(f, delimiter=',')
        self.kid = meta_table[:,1] #
        self.koi = meta_table[:,2] #
        self.tperiod = meta_table[:,3] #
        self.tepoch = meta_table[:,6] #
        self.impact = meta_table[:,9] #
        self.tdur = meta_table[:,12]/24. #
        self.prad = meta_table[:,15] #
        self.earthrads = meta_table[:,18] #
        self.inclin = meta_table[:,21] #
        self.eqtemp = meta_table[:,24] #
        self.insol = meta_table[:,27] #
        self.dor = meta_table[:,30] #
        self.gamma2 = meta_table[:,33] #
        self.gamma1 = meta_table[:,34] #
        self.ntrans = meta_table[:,35] #
        self.tcenum = meta_table[:,36] #
        self.steff = meta_table[:,37] #
        self.slogg = meta_table[:,40] #
        self.smet = meta_table[:,43] #
        self.srad = meta_table[:,46] #
        self.smass = meta_table[:,49] #

    def get_ind(self, system, tce, TOL):
        kid = int(system[-33:-24].lstrip('0'))
        tperiod = fits.getval(system,'TPERIOD',tce)
        tepoch = fits.getval(system,'TEPOCH',tce)
        rows = np.where(self.kid==kid)
        same_period = abs(self.tperiod[rows[0]]/tperiod-1) < TOL
        same_epoch0 = abs(self.tepoch[rows[0]]/tepoch-1) < TOL
        same_epoch1 = np.mod((self.tepoch[rows[0]]-tepoch)/tperiod,1) < TOL
        same_epoch2 = abs(np.mod((self.tepoch[rows[0]]-tepoch)/tperiod,1)-1) < TOL
        rows_ind = np.where(same_period & (same_epoch0 | same_epoch1 | same_epoch2))
        return rows[0][rows_ind[0]][0] if rows_ind[0].size > 0 else -1

class Results():
    def __init__(self, BATCH_ID):
        self.res_file = open('results_' + BATCH_ID + '.dat', 'w')
        self.par_file = open('parameters_' + BATCH_ID + '.dat', 'w')
        self.residuals = np.array([])
        self.chi = np.array([])
        self.angles = np.array([])
        self.phase_trans = np.array([]) #Phase in units of transits
        self.radii = np.array([])
        self.inclins = np.array([])
        self.distances = np.array([])
        self.weights = np.array([])
        
    #This is not beautiful.
    def save(self, dat):
        atemp = 360*dat.conv_phase/dat.tperiod
        wtemp = 1/(len(dat.conv_phase)*dat.conv_ferr)
        rtemp = (dat.conv_flux-dat.conv_mod)*wtemp/dat.fit_par[0]**2
        ptemp = dat.conv_phase/dat.tdur
        mtemp = dat.meta_ind*np.ones(len(atemp))
        ctemp = np.tile(chi2red(dat.conv_mod, dat.conv_flux, dat.conv_ferr),len(atemp))

        self.chi = np.concatenate((self.chi, ctemp))
        self.angles = np.concatenate((self.angles, atemp))
        self.residuals = np.concatenate((self.residuals, rtemp))
        self.weights = np.concatenate((self.weights, wtemp))
        self.phase_trans = np.concatenate((self.phase_trans, ptemp))
        np.savetxt(self.res_file,np.transpose(np.vstack((mtemp, ctemp, atemp, rtemp, wtemp, ptemp))), delimiter=',')
        self.res_file.flush()

        rtemp = np.array([dat.fit_par[0], dat.prad, dat.dvprad])
        itemp = np.array([dat.fit_par[1], dat.inclin, dat.dvinclin])
        dtemp = np.array([dat.fit_par[2], dat.dor, dat.dvdor])
        self.radii = np.vstack((self.radii, rtemp)) if self.radii.size > 0 else rtemp #Seems to be a bad solution
        self.inclins = np.vstack((self.inclins, itemp)) if self.inclins.size > 0 else itemp
        self.distances = np.vstack((self.distances, dtemp)) if self.distances.size > 0 else dtemp
        np.savetxt(self.par_file,np.reshape(np.concatenate((rtemp,itemp,dtemp)),(1,9)), delimiter=',')
        self.par_file.flush()

#Saves all interesting fitted values
class Values():
    def __init__(self, BATCH_ID):
        self.val_file = open('values_' + BATCH_ID + '.dat', 'w')
        self.koi = None
        self.alpha = None
        self.prad = None
        self.nea_prad = None
        self.dv_prad = None
        self.inclin = None
        self.nea_inclin = None
        self.dv_inclin = None
        self.dor = None
        self.nea_dor = None
        self.dv_dor = None
        self.g1 = None
        self.h1 = None
        self.g2 = None
        self.h2 = None
        self.chi = None
        self.xi5 = None
        self.xi30 = None
        self.zeta5 = None
        self.zeta30 = None
        self.gfit = None
        self.f1 = None
        self.f2 = None
        self.err = None
        
    #This is not beautiful.
    def save(self, dat, GFIT):
        coefs = np.polyfit(np.log(dat.binsize[0:dat.binsize.size/2]),np.log(dat.std[0:dat.binsize.size/2]),1)
        sig5 = griddata(dat.binsize, dat.std, 5) if 5 > min(dat.binsize) and 5 < max(dat.binsize) else np.infty
        sig10 = griddata(dat.binsize, dat.std, 10) if 10 > min(dat.binsize) and 10 < max(dat.binsize) else np.infty
        sig30 = griddata(dat.binsize, dat.std, 30) if 30 > min(dat.binsize) and 30 < max(dat.binsize) else np.infty
        self.koi = dat.koi
        self.alpha = coefs[0]
        self.prad = dat.fit_par[0]
        self.nea_prad = dat.prad
        self.dv_prad = dat.dvprad
        self.inclin = dat.fit_par[1]
        self.nea_inclin = dat.inclin
        self.dv_inclin = dat.dvinclin
        self.dor = dat.fit_par[2]
        self.nea_dor = dat.dor
        self.dv_dor = dat.dvdor
        self.g1 = dat.gamma1
        self.h1 = poor_mans_bound(dat.fit_par[3], GFIT)
        self.g2 = dat.gamma2
        self.h2 = poor_mans_bound(dat.fit_par[4], GFIT)
        self.chi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        self.xi5 = abs(min(dat.ex_mod))/sig5
        self.xi30 = abs(min(dat.ex_mod))/sig30
        self.zeta5 = sig5
        self.zeta10 = sig10
        self.zeta30 = sig30
        self.gfit = GFIT
        self.f1 = dat.fit_par[3]
        self.f2 = dat.fit_par[4]
        self.err = np.std(dat.flux-dat.instr_mod)
        
        #As Shaggy woulda said, it wasnt me.
        helper = ','.join(str(item) for item in [self.alpha, self.prad, self.nea_prad,\
                                                 self.dv_prad, self.inclin, self.nea_inclin,\
                                                 self.dv_inclin, self.dor, self.nea_dor,\
                                                 self.dv_dor, self.g1, self.h1, self.g2,\
                                                 self.h2, self.chi, self.xi5, self.xi30,\
                                                 self.zeta5, self.zeta10, self.zeta30, self.gfit,\
                                                 self.f1, self.f2, self.err])

        helper = ','.join([self.koi.replace("_",""), helper])
        self.val_file.write(helper + "\n")
        self.val_file.flush()

class S1vals():
    def __init__(self, path):
        files=glob(path) #Find all files. 
        dat = []
        for i in range(0,len(files)):
            dat.append(np.loadtxt(files[i], delimiter=','))
            
        dat = np.vstack(dat)
        n = dat.shape[0]
            
        self.koi = dat[:,0]
        self.alpha = dat[:,1]
        self.prad = dat[:,2]
        self.nea_prad = dat[:,3]
        self.dv_prad = dat[:,4]
        self.inclin = dat[:,5]
        self.nea_inclin = dat[:,6]
        self.dv_inclin = dat[:,7]
        self.dor = dat[:,8]
        self.nea_dor = dat[:,9]
        self.dv_dor = dat[:,10]
        self.g1 = dat[:,11]
        self.h1 = dat[:,12]
        self.g2 = dat[:,13]
        self.h2 = dat[:,14]
        self.chi = dat[:,15]
        self.xi5 = dat[:,16]
        self.xi30 = dat[:,17]
        self.zeta5 = dat[:,18]
        self.zeta10 = dat[:,19]
        self.zeta30 = dat[:,20]
        self.gfit = dat[:,21]
        self.f1 = dat[:,22]
        self.f2 = dat[:,23]
        self.err = dat[:,24]

class S2vals():
    def __init__(self, BATCH_ID):
        self.f_handle = open('s2vals_' + BATCH_ID + '.dat','w')
        self.residuals =  open('residuals_' + BATCH_ID + '.dat','w')
#        self.ex_pha = None
#        self.ex_mod = None
        self.pm_mod = None
        self.delta_mod = None
        self.fit_par = None
        self.pppv_mod = None
        self.pppv_par = None
        self.pppv_edge = None
        self.pppv_delT = None
        self.pppv_delA = None
        self.t = None
        self.a = None
        self.std = None
        self.chi2nor = None
        self.chi2ref = None
        self.delchi2 = None
        self.chi2nor_red = None
        self.chi2ref_red = None
        self.delchi2_red = None
        self.edge = None
        self.del_chi2ref = None
        self.del_delchi2 = None
        self.del_chi2ref_red = None
        self.del_delchi2_red = None

    def save(self, dat, s1vals, s1ind):
        vals = np.array([self.t, self.a, self.delT, self.delA, self.std, self.chi2nor, self.chi2ref, \
                         self.delchi2, self.del_chi2ref, self.del_delchi2, self.chi2nor_red, self.chi2ref_red, self.delchi2_red, \
                         self.del_chi2ref_red, self.del_delchi2_red, self.pppv_delT_nor, self.pppv_delA_nor, \
                         self.pppv_delT, self.pppv_delA, self.pppv_chi2nor, self.pppv_chi2ref, \
                         self.pppv_delchi2, self.pppv_chi2nor_red, self.pppv_chi2ref_red, self.pppv_delchi2_red, \
                         self.edge.size, self.pppv_edge.size, s1vals.zeta30[s1ind], s1vals.koi[s1ind], self.gauss_chi2ref, self.gauss_delchi2, \
                         self.gauss_chi2ref_red, self.gauss_delchi2_red, self.gauss_delT, self.gauss_delA, \
                         s1vals.prad[s1ind], s1vals.inclin[s1ind], s1vals.dor[s1ind]])
        np.savetxt(self.f_handle, vals[None], delimiter = ',')
        self.f_handle.flush()

        tdur = calc_tdur(dat)
        tingr = calc_tingr(dat)
#        if self.delchi2_red >= 0.1:
#            plt.plot(dat.phase, dat.flux-dat.instr_mod)
#            plt.plot(dat.phase, dat.flux-self.pm_mod)
#            plt.show()
        if tdur > 1e-12 and self.delchi2_red < 0.1:
            e = self.edge
            pe = self.pppv_edge
            norm_pha = dat.phase[e]/tingr-np.sign(dat.phase[e])*(tdur/(2*tingr)-5)
#            plt.plot(norm_pha)
#            plt.show()
            np.savetxt(self.residuals, np.c_[norm_pha, dat.ferr, dat.flux[e]-dat.instr_mod[e], dat.flux[e]-self.pm_mod, \
                                             dat.flux[e]-self.delta_mod, dat.flux[e]-self.gauss_mod], delimiter = ',')
            self.residuals.flush()
