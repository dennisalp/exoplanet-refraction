import numpy as np
import pdb

def sane(res, vals, meta):
    p0 = vals[res.val_id,3]
    return p0 < 0.5
    
def shallow(res, vals, meta):
    aa = res.ntot-res.pts
    bb = res.ntot
    return np.min(res.scal) > 0

def deep(res, vals, meta):
    aa = res.ntot-res.pts
    bb = res.ntot
    return np.min(res.scal[aa:bb]) < 0

def noise(res, vals, meta):
    alpha = vals[res.val_id,2]
#    return alpha > -999999
#    return (alpha > -0.80) & (alpha < -0.2)
#    return (alpha > -1.00) & (alpha <  0.2)
    return (alpha > -0.8) & (alpha < -0.3)
    return (alpha > -0.65) & (alpha < -0.425)


def chi(res, vals, meta):
    return vals[res.val_id,10] < 999992

def acceptance(res, vals, meta):
    return vals[res.val_id,22]/vals[res.val_id,21] > 0.

def far(res, vals, meta):
    return (meta.tperiod[res.met_id] < 999999) & (meta.tperiod[res.met_id] > 40)
    return vals[res.val_id,5] > 12

def jovian(res, vals, meta):
    dor = vals[res.val_id,5]
    p0  = vals[res.val_id,3]
    tper= meta.tperiod[res.met_id]
    return (p0 > 0.06)# & (dor > 60) & (tper > 80)

def earthian(res, vals, meta):
    dor = vals[res.val_id,5]
    p0  = vals[res.val_id,3]
    tper= meta.tperiod[res.met_id]
    return (p0 < 0.02)# & (dor > 60) & (tper > 80)

def neptunian(res, vals, meta):
    dor = vals[res.val_id,5]
    p0  = vals[res.val_id,3]
    tper= meta.tperiod[res.met_id]
    return (p0 > 0.02) & (p0 < 0.06)# & (dor > 60) & (tper > 80)
    
def sample(res, vals, meta):
    p0  = vals[res.val_id,3]
    return (p0 > 0.025)# & (p0 < 0.02)

def ttv(res, vals, meta):
    ttv = vals[res.val_id,-1]
    return ttv*24*60 > -1

def test(res, vals, meta):
    return vals[res.val_id,1]==1894.01
