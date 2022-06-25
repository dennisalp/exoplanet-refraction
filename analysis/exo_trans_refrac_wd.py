#Calculates the non-linear limb darkening transit using simps
from numpy import pi, sqrt, reshape, tile, repeat, arccos, linspace, \
    zeros, ones, exp, arange, cos, sin, where, log, array, savetxt, \
    concatenate, c_, isnan, isinf
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

nr = 800
nt = 400

def refraction_transit(zin, g1, g2, B, H, p0):
#########
#Stuff
    def calc_flux(z0, g1, g2, BB, HH, p0):
        def I(r):
            mu = sqrt(1-r**2)
            return 1  - g1*(1-mu) - g2*(1-mu)**2

        def newrap(func, dfunc, x0):
            x1 = x0 + 2*TOL
            func1 = zeros(r.size)
            while (max(abs(x0 - x1)) > TOL): #Could perhaps throw away solved equations
                left = where(abs(x0-x1) > TOL)[0] #20% time saved, 99% of calculations, because Python?
                func1[left] = func(x1[left], left)
    #            print max(abs(func1)), left.size, 'hey'
                x0[left] = x1[left]
                x1[left] = x0[left] - func1[left]/dphi(x0[left], left)
            return x1

        def uti1(ti1, left):
            shift = ti1-z0
            temp = shift*ts2[left]/(ts1[left]-z0)
            return sqrt(shift**2+temp**2)

        def phi(ti1, left):
            uhelp = uti1(ti1, left)
            res = ts1[left]-z0-(ti1-z0)*(1-B*sqrt(pi*H/(2*uhelp))*exp(-(uhelp-p0)/H))
            return res

        def dphi(ti1, left):
            uhelp = uti1(ti1, left)
            shifti = ti1-z0
            shifts = ts1[left]-z0
            dudti1 = 0.5*(2*shifti+2*shifti*ts2[left]**2/shifts**2)/uhelp
            expu = exp(-(uhelp-p0)/H)
            temp = sqrt(pi/2)*B*(H+2*uhelp)*expu/(2*sqrt(H*uhelp**3))*dudti1
            res = -1+B*sqrt(pi*H/(2*uhelp))*expu-(ti1-z0)*temp
            return res

        def gen_guess(first):
            guess = first*(ts1-z0)/sqrt((ts1-z0)**2+ts2**2)*(p0+TOL)+z0 #Adding TOL here makes phi run 20x faster
            signs = where(ts1-z0 > 0, 1*first, -1*first)
            func = phi(guess, array(range(0,r.size)))
            left = where(func*signs < 0)[0]
            while left.size > 0: #Find good guess by stepping images into the lens
#            print min(func*signs), left.size, 'guess'
                guess[left] = (guess[left]-z0)*0.99+z0
                func[left] = phi(guess[left], left)
                left = where(func*signs < 0)[0]
            return guess
        #z0 = abs(zin)
        p0 = abs(p0)
        TOL = 1e-12

#########
#Unocculted flux
        r = linspace(1e-10,1,nr)
        norm = simps(2*pi*r*I(r), r)

#########
#Prepare
        theta = linspace(0., pi, nt)#arange(0., pi, pi/nt)
        r = tile(r, nt)
        theta = repeat(theta, nr)
        cost = cos(theta)

        ts1 = r*cost
        ts2 = r*sin(theta)

#########
#Images
        guess = gen_guess(1.)

        ti1 = newrap(phi, dphi, guess.copy())
        shifti = ti1-z0
        shifts = ts1-z0
        ti2 = shifti*ts2/shifts
    
        Au= sqrt(shifti**2+ti2**2)
        expu = exp(-(Au-p0)/H)
        Aphi = -B*sqrt(pi*H/(2*Au))*expu
        u2phi= B*sqrt(pi/2)*sqrt(H/Au)*(1/2.+Au/H)*expu
           
        A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
        A   = 1/A
        W   = where(Au > p0, 1., 0.)
        integrand = 2*r*I(r)*abs(A)*W #Magnification has to be positive, right?

#########
#Caustics
        if 1-sqrt(pi/2)*sqrt(H/p0)*B < 0:
            guess = gen_guess(-1.)
            ti1b = newrap(phi, dphi, guess.copy())
            shifti = ti1b-z0
            shifts = ts1-z0
            ti2b = shifti*ts2/shifts
        
            Au= sqrt(shifti**2+ti2b**2)
            expu = exp(-(Au-p0)/H)
            Aphi = -B*sqrt(pi*H/(2*Au))*expu
            u2phi= B*sqrt(pi/2)*sqrt(H/Au)*(1/2.+Au/H)*expu

            A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
            A   = 1/A
            W   = where(Au > p0, 1., 0.)
            integrand += 2*r*I(r)*abs(A)*W #DON'T KNOW WHY THIS IS PLUS, MINUS MAKES MORE SENSE FOR DIAMOND RING

#########
#Integrate
        integrand = where(isnan(integrand) | isinf(integrand), 0, integrand) #Should be fine
        integrand = integrand.reshape((nt, nr))
        theta = theta.reshape((nt, nr))
        r = r.reshape((nt, nr))
        integrand = simps(integrand,theta,axis=0)
        F = simps(integrand,r[0],axis=0)
        F = F/norm
        return F

    FF = zeros(zin.size)
    for idx, z0 in enumerate(zin):
        print(idx,z0)
        FF[idx] += calc_flux(z0, g1, g2, B, H, p0)

    return FF