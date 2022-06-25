import numpy as np
import pdb

from astropy.table import Table
from scipy.interpolate import griddata

from exo_sap_hlp import *

#########
#Classes
#Holds all data on a planet candidate
class Data():
    def __init__(self, meta, meta_ind):
        self.meta_ind = meta_ind
        self.kid = meta.kid[self.meta_ind]
        self.koi = meta.koi[self.meta_ind]
        self.tperiod = meta.tperiod[self.meta_ind]
        self.tepoch = meta.tepoch[self.meta_ind]
        self.impact = meta.impact[self.meta_ind]
        self.tdur = meta.tdur[self.meta_ind]
        self.prad = meta.prad[self.meta_ind]
        self.earthrads = meta.earthrads[self.meta_ind]
        self.inclin = meta.inclin[self.meta_ind]
        self.eqtemp = meta.eqtemp[self.meta_ind]
        self.insol = meta.insol[self.meta_ind]
        self.dor = meta.dor[self.meta_ind]
        self.gamma2 = meta.gamma2[self.meta_ind]
        self.gamma1 = meta.gamma1[self.meta_ind]
        self.ntrans = meta.ntrans[self.meta_ind]
        self.steff = meta.steff[self.meta_ind]
        self.slogg = meta.slogg[self.meta_ind]
        self.smet = meta.smet[self.meta_ind]
        self.srad = meta.srad[self.meta_ind]
        self.smass = meta.smass[self.meta_ind]

        self.binsize = []
        self.std = []
        self.time = np.array([])
        self.flux = np.array([])
        self.ferr = np.array([])
        self.phas = None
        self.ttv = None
        self.ttv_lst = None
        self.fit_par = None
        self.conv_phase = None
        self.conv_res = None
        self.instr_mod = None
        self.calc_ntrans = None
        self.npro = 0
        self.nslo = 0
        self.ngof = 0
    
    def add(self, path):
        tab = Table.read(path, hdu=1)
        self.time = np.concatenate((self.time, np.array(tab['TIME'])))
#        self.flux = np.concatenate((self.flux, np.array(tab['SAP_FLUX'])))
#        self.ferr = np.concatenate((self.ferr, np.array(tab['SAP_FLUX_ERR'])))
        tmp = np.where(np.array(tab['SAP_QUALITY']) == 0, np.array(tab['SAP_FLUX']), np.nan)
        self.flux = np.concatenate((self.flux, tmp))
        tmp = np.where(np.array(tab['SAP_QUALITY']) == 0, np.array(tab['SAP_FLUX_ERR']), np.nan)
        self.ferr = np.concatenate((self.ferr, tmp))
#        self.flux = np.concatenate((self.flux, np.array(tab['SAP_FLUX'])-np.array(tab['SAP_BKG'])))
#        self.ferr = np.concatenate((self.ferr, np.sqrt(np.array(tab['SAP_FLUX_ERR'])**2+np.array(tab['SAP_BKG_ERR'])**2)))

        self.orig_flux = self.flux.copy()
        self.orig_ferr = self.ferr.copy()

    def upt_lists(self, ind):
        self.time = self.time[ind]
        self.flux = self.flux[ind]
        self.ferr = self.ferr[ind]
        self.orig_flux = self.orig_flux[ind]
        self.orig_ferr = self.orig_ferr[ind]

        if not self.phas is None:
            self.phas = self.phas[ind]
        if not self.instr_mod is None:
            self.instr_mod = self.instr_mod[ind]

    def set_conv(self, conv_phase, conv_res):
        self.conv_phase = conv_phase
        self.conv_res = conv_res

    def save_res(self):
        zz = get_z(self.phas, self.tperiod, self.fit_par[1], self.fit_par[2])
        if self.ttv is None:
            out_arr = np.vstack((self.time, self.phas, zz, self.flux, self.ferr, self.instr_mod))
        else:
            out_arr = np.vstack((self.time, self.phas, zz, self.flux, self.ferr, self.instr_mod, self.ttv))
        np.save(str(self.kid) + "_" + str(self.koi), out_arr.T)

#Contains meta data on the candidates
class Meta():
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
        self.steff = meta_table[:,36] #
        self.slogg = meta_table[:,39] #
        self.smet = meta_table[:,42] #
        self.srad = meta_table[:,45] #
        self.smass = meta_table[:,48] #

#Saves all interesting fitted values
class Values():
    def __init__(self, BATCH_ID):
        self.val_file = open('values_' + BATCH_ID + '.dat', 'w')
        
    #This is not beautiful.
    def save(self, dat, GFIT):
        coefs = np.polyfit(np.log(dat.binsize[0:dat.binsize.size/2]),np.log(dat.std[0:dat.binsize.size/2]),1)
        sig5 = griddata(dat.binsize, dat.std, 5) if 5 > min(dat.binsize) and 5 < max(dat.binsize) else np.infty
        sig10 = griddata(dat.binsize, dat.std, 10) if 10 > min(dat.binsize) and 10 < max(dat.binsize) else np.infty
        sig30 = griddata(dat.binsize, dat.std, 30) if 30 > min(dat.binsize) and 30 < max(dat.binsize) else np.infty
        self.kid = dat.kid
        self.koi = dat.koi
        self.alpha = coefs[0]
        self.prad = dat.fit_par[0]
        self.inclin = dat.fit_par[1]
        self.dor = dat.fit_par[2]
        self.g1 = dat.gamma1
        self.h1 = poor_mans_bound(dat.fit_par[3], GFIT)
        self.g2 = dat.gamma2
        self.h2 = poor_mans_bound(dat.fit_par[4], GFIT)
        self.chi = chi2red(dat.instr_mod, dat.flux, dat.ferr)
        self.xi5 = abs(min(dat.ex_mod)+1)/sig5
        self.xi10 = abs(min(dat.ex_mod)+1)/sig10
        self.xi30 = abs(min(dat.ex_mod)+1)/sig30
        self.zeta5 = sig5
        self.zeta10 = sig10
        self.zeta30 = sig30
        self.gfit = GFIT
        self.f1 = dat.fit_par[3]
        self.f2 = dat.fit_par[4]
        self.err = np.std(dat.flux-dat.instr_mod)
        self.init_ntrans = dat.init_ntrans
        self.calc_ntrans = dat.calc_ntrans
        self.npro = dat.npro
        self.nslo = dat.nslo
        self.ngof = dat.ngof
        self.tdur = calc_tdur(dat)

        if dat.ttv is None:
            helper = ','.join(str(item) for item in [self.kid, self.koi, self.alpha, self.prad, self.inclin, self.dor,\
            self.g1, self.h1, self.g2,self.h2, self.chi, self.xi5, self.xi10, self.xi30,\
            self.zeta5, self.zeta10, self.zeta30, self.gfit, self.f1, self.f2, self.err,\
            self.init_ntrans, self.calc_ntrans, self.npro, self.nslo, self.ngof, self.tdur])
        else:
            self.ttv_max = np.amax(np.abs(dat.ttv))
            helper = ','.join(str(item) for item in [self.kid, self.koi, self.alpha, self.prad, self.inclin, self.dor,\
            self.g1, self.h1, self.g2,self.h2, self.chi, self.xi5, self.xi10, self.xi30,\
            self.zeta5, self.zeta10, self.zeta30, self.gfit, self.f1, self.f2, self.err,\
            self.init_ntrans, self.calc_ntrans, self.npro, self.nslo, self.ngof, self.tdur, self.ttv_max])
#        else:
#            ttv_par = ttv_bounds(dat.fit_par[5], dat.fit_par[6], dat.fit_par[7])
#            self.ttv_aa = ttv_par[0]
#            self.ttv_ww = ttv_par[1]
#            self.ttv_phi = ttv_par[2]
#        
#            helper = ','.join(str(item) for item in [self.kid, self.koi, self.alpha, self.prad, self.inclin, self.dor,\
#                self.g1, self.h1, self.g2,self.h2, self.chi, self.xi5, self.xi10, self.xi30,\
#                self.zeta5, self.zeta10, self.zeta30, self.gfit, self.f1, self.f2, self.err,\
#                self.init_ntrans, self.calc_ntrans, self.npro, self.nslo, self.ngof, self.tdur,\
#                self.ttv_aa, self.ttv_ww, self.ttv_phi])

        self.val_file.write(helper + "\n")
        self.val_file.flush()

# The object that stores all residuals and aux for analysis
class Residuals():
    def __init__(self, NX, NY, vals):
        self.first = True
        self.NX = NX
        self.NY = NY
        self.ntot = 0
        self.ncap = 0
        self.ngood = 0
        self.hmap = np.zeros((NX,NY))
        self.mean = np.zeros(NX)
        self.std = []
        [self.std.append([]) for i in range(0,NX)]
        self.stdw = []
        [self.stdw.append([]) for i in range(0,NX)]
        self.count = np.zeros(NX)
        self.cwght = np.zeros(NX)
        self.err = np.zeros(self.NX)
        self.sem = np.zeros(self.NX)

        self.tingr = np.zeros(vals.shape[0])
        self.bb = np.zeros(vals.shape[0])

        self.nprad = NX/2
        self.uni = self.NX/self.nprad
        self.midpt1 = self.nprad/4*self.uni
        self.midpt2 = self.nprad/2*self.uni+self.midpt1

    def set_ids(self, vals, meta, kic, koi):
        self.kid = kic
        self.koi = koi            
        self.val_id = np.where(vals[:,1] == koi)[0][0]
        self.met_id = np.where(meta.koi == koi)[0][0]

    def add(self, idx, path, vals, meta):
        def fix_z(z, scal, pha, tperiod, p0):
            backside = np.where(np.abs(2*np.pi*pha/tperiod) > np.pi/2)[0]
            if np.any(backside):
                z[backside] = 100+p0
                scal[backside] = 100+p0

        def get_tingr(pha, zz, p0):
            s1 = np.where(zz < 1+p0)[0]
            if s1.size == 0:
                return 0.
            s1 = s1[0]
        
            e1 = np.where(zz < 1-p0)[0]
            if e1.size == 0:
                return 0.
            e1  = e1[0]
        
            s0 = s1 - 1
            e0 = e1 - 1
            if s1 == 0:
                return 0.
            s = griddata(zz[s0:s1+1], pha[s0:s1+1], 1+p0, method='linear')
            e = griddata(zz[e0:e1+1], pha[e0:e1+1], 1-p0, method='linear')
            return e-s

        def alloc():
            old_cap = self.ncap
            self.ncap *= 4
            ttime = np.zeros(self.ncap)
            tphase= np.zeros(self.ncap)
            tzz   = np.zeros(self.ncap)
            tflux = np.zeros(self.ncap)
            tferr = np.zeros(self.ncap)
            tmod  = np.zeros(self.ncap)
            tscal = np.zeros(self.ncap)
            twght = np.zeros(self.ncap)
            ttime [:old_cap] = self.time
            tphase[:old_cap] = self.phase
            tzz   [:old_cap] = self.zz
            tflux [:old_cap] = self.flux
            tferr [:old_cap] = self.ferr
            tmod  [:old_cap] = self.mod
            tscal [:old_cap] = self.scal
            twght [:old_cap] = self.wght
            self.time  = ttime
            self.phase = tphase
            self.zz    = tzz
            self.flux  = tflux
            self.ferr  = tferr
            self.mod   = tmod
            self.scal  = tscal
            self.wght  = twght
        
        dat = np.load(path)
        self.pts = dat.shape[0]
        prad = vals[self.val_id, 3]
        tdur = meta.tdur[self.met_id]
        tperiod = meta.tperiod[self.met_id]

        def get_inside(dat, prad):
            scal = (dat[:,2]-1)/prad
            phas = dat[:,1]
            negative = -self.uni*scal+self.midpt1
            positive =  self.uni*scal+self.midpt2
            xvals = np.where(phas < 0, negative, positive)
            return (xvals >= 0) & (xvals < self.NX)

        inside = get_inside(dat, prad)
        ninside = np.sum(inside)

        if ninside < 15:
            semp = dat[:,4]
        else:
            semp = np.std(dat[inside,3]-dat[inside,5])*np.ones(dat.shape[0])
        if semp.shape[0] == 0:
            semp = np.ones(dat.shape[0])
        
        if self.first:
            self.time = dat[:,0]
            self.phase = dat[:,1]
            self.zz = dat[:,2]
            self.flux = dat[:,3]
            self.ferr = dat[:,4]
            self.mod = dat[:,5]
            self.scal = (dat[:,2]-1)/prad
#            self.wght = 1/(ninside*semp**2)
#            self.wght = 1/semp**2
            self.wght = 1/dat[:,4]**2
            
            self.ncap = self.pts
            self.first = False
        else:
            while self.ntot + self.pts > self.ncap:
                alloc()

            self.time[self.ntot:self.ntot+self.pts] = dat[:,0]
            self.phase[self.ntot:self.ntot+self.pts] = dat[:,1]
            self.zz[self.ntot:self.ntot+self.pts] = dat[:,2]
            self.flux[self.ntot:self.ntot+self.pts] = dat[:,3]
            self.ferr[self.ntot:self.ntot+self.pts] = dat[:,4]
            self.mod[self.ntot:self.ntot+self.pts] = dat[:,5]
            self.scal[self.ntot:self.ntot+self.pts] = (dat[:,2]-1)/prad
#            self.wght[self.ntot:self.ntot+self.pts] = 1/(ninside*semp**2)
#            self.wght[self.ntot:self.ntot+self.pts] = 1/semp**2
            self.wght[self.ntot:self.ntot+self.pts] = 1/dat[:,4]**2

        aa = self.ntot
        self.ntot += self.pts
        bb = self.ntot
        
        fix_z(self.zz[aa:bb], self.scal[aa:bb], self.phase[aa:bb], tperiod, prad)
        self.tingr[idx] = get_tingr(self.phase[aa:bb], self.zz[aa:bb], prad)
        self.bb[idx] = np.cos(np.deg2rad(vals[self.val_id,4]))*vals[self.val_id,5]

    def rm_last(self):
        aa = self.ntot-self.pts
        bb = self.ntot
        self.time [aa:bb]= 0
        self.phase[aa:bb]= 0
        self.zz   [aa:bb]= 0
        self.flux [aa:bb]= 0
        self.ferr [aa:bb]= 0
        self.mod  [aa:bb]= 0
        self.scal [aa:bb]= 0
        self.wght [aa:bb]= 0
        self.ntot -= self.pts

    def finalize(self):
        self.time = self.time[:self.ntot]
        self.phase= self.phase[:self.ntot]
        self.zz   = self.zz[:self.ntot]
        self.flux = self.flux[:self.ntot]
        self.ferr = self.ferr[:self.ntot]
        self.mod  = self.mod[:self.ntot]
        self.scal = self.scal[:self.ntot]
        self.wght = self.wght[:self.ntot]
        self.ncap = self.ntot

    def mk_map(self):
        def get_xcoord():
            negative = -self.uni*self.scal+self.midpt1
            positive =  self.uni*self.scal+self.midpt2
            self.mapX = np.where(self.phase < 0, negative, positive)
            # Exterminate negative and positive steppinginto each others regions
            self.mapX = np.where(negative > self.NX/2, -1, self.mapX)
            self.mapX = np.where(positive < self.NX/2, -1, self.mapX)

        def get_yppm():
            resi = (self.flux-self.mod)
            self.mapY = resi/1e-6

#        def add2map():
#            inplt = np.where((self.mapX >= 0) & (self.mapX < self.NX) & (self.mapY >= 0) & (self.mapY < self.NY))
#            xx = np.floor(self.mapX[inplt]).astype('int')
#            yy = np.floor(self.mapY[inplt]).astype('int')
#            tmp = xx.size
#            for ii in range(0,xx.size):
#                self.hmap[xx[ii], yy[ii]] += 1

        def get_mva(): # Move average
            mvn = 5000
            self.mvx = np.linspace(0, self.NX, mvn)
            self.mva = np.zeros(mvn)
            tmp_zz = np.where(self.phase > 0, self.scal, -self.scal)
            for ii, mv_pos in enumerate(self.mvx):
                inside = np.abs(tmp_zz - mv_pos) < 0.5
                self.mva[ii] = np.sum(self.wght[inside]*(self.flux[inside]-self.mod[inside]))/np.sum(self.wght[inside])

            self.mva /= 1e-6
            
        def add2mean():
            inx = np.where((self.mapX >= 0) & (self.mapX < self.NX))
            xx = np.floor(self.mapX[inx]).astype('int')
            for ii in range(0,xx.size):
                self.mean[xx[ii]] += self.mapY[inx[0][ii]]*self.wght[inx[0][ii]]
                self.count[xx[ii]] += 1
                self.cwght[xx[ii]] += self.wght[inx[0][ii]]
                self.std[xx[ii]].append(self.mapY[inx[0][ii]])
                self.stdw[xx[ii]].append(self.wght[inx[0][ii]])

#            pdb.set_trace()
            self.mean = self.mean/self.cwght
#            self.hmap = self.hmap/np.repeat(self.count,self.NY).reshape(self.hmap.shape)
    
        def calc_sem():
#            for xii, xpos in enumerate(self.std):
#                self.err[xii] = np.sqrt(np.sum((np.array(xpos)-self.mean[xii])**2)/len(xpos))
#                self.sem[xii] = self.err[xii]/np.sqrt(len(xpos))
            for xx, vals in enumerate(self.std):
                vals = np.array(vals)
                valsw = np.array(self.stdw[xx])
                variance = np.sum(valsw*(vals-self.mean[xx])**2)/np.sum(valsw)
                self.err[xx] = np.sqrt(variance)
                self.sem[xx] = self.err[xx]/np.sqrt(vals.shape[0])
       

        get_xcoord()
        get_yppm()
#        add2map()
        get_mva()
        add2mean()
        calc_sem()

    def print_diag(self):
        mean = np.mean(self.mean)
        print 'Mean:', mean
        err = np.mean(self.sem)
        print 'Error:', err
        chi2 = np.sum((self.mean/self.sem)**2)/self.mean.shape[0]
        print 'chi2:', chi2

# The object that stores all residuals and aux for analysis
class Wings():
    def __init__(self, NX, resolution, vals):
        self.first = True
        self.NX = NX
        self.resolution = resolution
        self.ntot = 0
        self.ncap = 0
        self.ngood = 0
        self.mean = np.zeros(NX)
        self.sqs = np.zeros(NX)
        self.std = []
        [self.std.append([]) for i in range(0,NX)]
        self.stdw = []
        [self.stdw.append([]) for i in range(0,NX)]
        self.count = np.zeros(NX)
        self.cwght = np.zeros(NX)
        self.err = np.zeros(self.NX)
        self.sem = np.zeros(self.NX)

        self.tingr = np.zeros(vals.shape[0])
        self.bb = np.zeros(vals.shape[0])

    def set_ids(self, vals, meta, kic, koi):
        self.kid = kic
        self.koi = koi
        self.val_id = np.where(vals[:,1] == koi)[0][0]
        self.met_id = np.where(meta.koi == koi)[0][0]

    def add(self, idx, path, vals, meta):
        def fix_z(zz, pha, tperiod, p0):
            backside = np.where(np.abs(2*np.pi*pha/tperiod) > np.pi/2)[0]
            if np.any(backside):
                zz[backside] = 100+p0
            return zz

        def get_tingr(pha, zz, p0):
            s1 = np.where(zz < 1+p0)[0]
            if s1.size == 0:
                return 0.
            s1 = s1[0]
        
            e1 = np.where(zz < 1-p0)[0]
            if e1.size == 0:
                return 0.
            e1  = e1[0]
        
            s0 = s1 - 1
            e0 = e1 - 1
            if s1 == 0:
                return 0.
            s = griddata(zz[s0:s1+1], pha[s0:s1+1], 1+p0, method='linear')
            e = griddata(zz[e0:e1+1], pha[e0:e1+1], 1-p0, method='linear')
            return e-s

        def alloc():
            old_cap = self.ncap
            self.ncap *= 4
            ttime = np.zeros(self.ncap)
            tphase= np.zeros(self.ncap)
            tzz   = np.zeros(self.ncap)
            tflux = np.zeros(self.ncap)
            tferr = np.zeros(self.ncap)
            tmod  = np.zeros(self.ncap)
            tscal = np.zeros(self.ncap)
            twght = np.zeros(self.ncap)
            ttime [:old_cap] = self.time
            tphase[:old_cap] = self.phase
            tzz   [:old_cap] = self.zz
            tflux [:old_cap] = self.flux
            tferr [:old_cap] = self.ferr
            tmod  [:old_cap] = self.mod
            tscal [:old_cap] = self.scal
            twght [:old_cap] = self.wght
            self.time  = ttime
            self.phase = tphase
            self.zz    = tzz
            self.flux  = tflux
            self.ferr  = tferr
            self.mod   = tmod
            self.scal  = tscal
            self.wght  = twght
        
        dat = np.load(path)
        self.pts = dat.shape[0]
        prad = vals[self.val_id, 3]
        tdur = meta.tdur[self.met_id]
        tperiod = meta.tperiod[self.met_id]
        zz = fix_z(dat[:,2], dat[:,1], tperiod, prad)

        def get_inside(zz):
            return zz < self.NX/2.*self.resolution

        inside = get_inside(zz)
        ninside = np.sum(inside)
        if ninside < 15:
            semp = dat[:,4]
        else:
            semp = np.std(dat[inside,3]-dat[inside,5])*np.ones(dat.shape[0])
        if semp.shape[0] == 0:
            semp = np.ones(dat.shape[0])
        
        if self.first:
            self.time = dat[:,0]
            self.phase = dat[:,1]
            self.zz = zz
            self.flux = dat[:,3]
            self.ferr = dat[:,4]
            self.mod = dat[:,5]
            self.scal = (dat[:,2]-1)/prad
            self.wght = 1/semp**2
            
            self.ncap = self.pts
            self.first = False
        else:
            while self.ntot + self.pts > self.ncap:
                alloc()

            self.time[self.ntot:self.ntot+self.pts] = dat[:,0]
            self.phase[self.ntot:self.ntot+self.pts] = dat[:,1]
            self.zz[self.ntot:self.ntot+self.pts] = zz
            self.flux[self.ntot:self.ntot+self.pts] = dat[:,3]
            self.ferr[self.ntot:self.ntot+self.pts] = dat[:,4]
            self.mod[self.ntot:self.ntot+self.pts] = dat[:,5]
            self.scal[self.ntot:self.ntot+self.pts] = (dat[:,2]-1)/prad
            self.wght[self.ntot:self.ntot+self.pts] = 1/semp**2

        aa = self.ntot
        self.ntot += self.pts
        bb = self.ntot
        
        self.tingr[idx] = get_tingr(self.phase[aa:bb], self.zz[aa:bb], prad)
        self.bb[idx] = np.cos(np.deg2rad(vals[self.val_id,4]))*vals[self.val_id,5]

    def rm_last(self):
        aa = self.ntot-self.pts
        bb = self.ntot
        self.time [aa:bb]= 0
        self.phase[aa:bb]= 0
        self.zz   [aa:bb]= 0
        self.flux [aa:bb]= 0
        self.ferr [aa:bb]= 0
        self.mod  [aa:bb]= 0
        self.scal [aa:bb]= 0
        self.wght [aa:bb]= 0
        self.ntot -= self.pts

    def finalize(self):
        self.time = self.time[:self.ntot]
        self.phase= self.phase[:self.ntot]
        self.zz   = self.zz[:self.ntot]
        self.flux = self.flux[:self.ntot]
        self.ferr = self.ferr[:self.ntot]
        self.mod  = self.mod[:self.ntot]
        self.scal = self.scal[:self.ntot]
        self.wght = self.wght[:self.ntot]
        self.ncap = self.ntot
        
    def mk_map(self):
        def get_xcoord():
            sgn = np.where(self.phase >= 0, 1, -1)
            self.mapX = sgn*self.zz/self.resolution+self.NX/2.

        def get_yppm():
            resi = (self.flux-self.mod)
            self.mapY = resi/1e-6

        def get_mva(): # Move average
            mvn = 1000
            self.mvx = np.linspace(-self.NX/2.*self.resolution, self.NX/2.*self.resolution, mvn)
            self.mva = np.zeros(mvn)
            tmp_zz = np.where(self.phase > 0, self.zz, -self.zz)
            for ii, mv_pos in enumerate(self.mvx):
                inside = np.abs(tmp_zz - mv_pos) < self.resolution/2.
                self.mva[ii] = np.sum(self.wght[inside]*(self.flux[inside]-self.mod[inside]))/np.sum(self.wght[inside])

            self.mva /= 1e-6
            
        def add2mean():
            inx = np.where((self.mapX >= 0) & (self.mapX < self.NX))
            xx = np.floor(self.mapX[inx]).astype('int')
            for ii in range(0,xx.size):
                self.mean[xx[ii]] += self.mapY[inx[0][ii]]*self.wght[inx[0][ii]]
                self.sqs[xx[ii]]  += (self.ferr[inx[0][ii]]/1e-6)**2*self.wght[inx[0][ii]]
                self.count[xx[ii]] += 1
                self.cwght[xx[ii]] += self.wght[inx[0][ii]]
                self.std[xx[ii]].append(self.mapY[inx[0][ii]])
                self.stdw[xx[ii]].append(self.wght[inx[0][ii]])
        
            self.mean = self.mean/self.cwght

        def calc_sem():
            for xx, vals in enumerate(self.std):
                vals = np.array(vals)
                valsw = np.array(self.stdw[xx])
                variance = np.sum(valsw*(vals-self.mean[xx])**2)/np.sum(valsw)
                self.err[xx] = np.sqrt(variance)
                self.sem[xx] = self.err[xx]/np.sqrt(vals.shape[0])

            self.sqs = np.sqrt(self.sqs/self.cwght)/np.sqrt(self.count)
            # The following line is the switch between Kepler and sample error
            self.sem = self.sqs.copy()
       
        get_xcoord()
        get_yppm()
        get_mva()
        add2mean()
        calc_sem()

    def print_diag(self):
        mean = np.mean(self.mean)
        print 'Mean:', mean
        err = np.mean(self.sem), np.median(self.sem), np.amin(self.sem), np.amax(self.sem)
        print 'Error:', err
        chi2 = np.sum((self.mean/self.sem)**2)/self.mean.shape[0]
        print 'chi2:', chi2
