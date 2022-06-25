from scipy.interpolate import griddata
import numpy as np
#NASA EA IS ATLAS LEAST SQUARES FROM CLARET 2011

def get_ldc(logg, t, met, mturb):
    print 'sad'
    args = np.reshape(np.array([logg,t,met,mturb]),(1,4))
    af=np.loadtxt('/home/dalp/Dropbox/astrophysics/lib/exoplanets/claret11/atlas_flux.dat')
    alsq=np.loadtxt('/home/dalp/Dropbox/astrophysics/lib/exoplanets/claret11/atlas_lsq.dat')
    pf=np.loadtxt('/home/dalp/Dropbox/astrophysics/lib/exoplanets/claret11/phoenix_flux.dat')
    plsq=np.loadtxt('/home/dalp/Dropbox/astrophysics/lib/exoplanets/claret11/phoenix_lsq.dat')
    
    print 'asd'
    ldc_af=griddata(af[:,0:4],af[:,4:],args, method='linear')
    print ldc_af
    ldc_alsq=griddata(alsq[:,0:4],alsq[:,4:],args, method='linear')
    print ldc_alsq
    ldc_pf=griddata(pf[:,0:4],pf[:,4:],args, method='linear')
    print ldc_pf
    ldc_plsq=griddata(plsq[:,0:4],plsq[:,4:],args, method='linear')
    print ldc_plsq
    
    return ldc_af, ldc_alsq, ldc_pf, ldc_plsq

print get_ldc(4.609, 5126, -0.12,  2.0)

#>>> meta.steff[0:10]
#np.array([ 5850.,  5850.,  6031.,  6046.,  6046.,  6046.,  5126.,  5850., 6350.,  6225.])
#np.>>> meta.slogg[0:10]
#np.array([ 4.426,  4.426,  4.438,  4.486,  4.486,  4.486,  4.609,  4.455, 4.021,  4.169])
#np.>>> meta.smet[0:10]
#np.array([ 0.14,  0.14,  0.07, -0.08, -0.08, -0.08, -0.12, -0.15,  0.26, -0.04])
#np.>>> meta.gamma1[0:10]
#np.array([ 0.3959,  0.3959,  0.3586,  0.3481,  0.3481,  0.3481,  0.5227, 0.3731,  0.3197,  0.3227])
#np.>>> meta.gamma2[0:10]
#np.array([ 0.2694,  0.2694,  0.2882,  0.2924,  0.2924,  0.2924,  0.1846, 0.2779,  0.309 ,  0.3015])

