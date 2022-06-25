#Calculates the non-linear limb darkening transit using simps
from numpy import pi, sqrt, reshape, tile, repeat, arccos, linspace, \
    zeros, ones, exp, arange, cos, sin, where, log, array, savetxt, \
    concatenate, c_, isnan, isinf
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def refract_simps(zin, g1, g2, B, H, p0, nr, nt):
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
        shift = ti1-z2
        temp = shift*ts2[left]/(ts1[left]-z2)
        return sqrt(shift**2+temp**2)

    def phi(ti1, left):
        uhelp = uti1(ti1, left)
        res = ts1[left]-z2-(ti1-z2)*(1-B*sqrt(pi*H/(2*uhelp))*exp(-(uhelp-p0)/H))
        return res

    def dphi(ti1, left):
        uhelp = uti1(ti1, left)
        shifti = ti1-z2
        shifts = ts1[left]-z2
        dudti1 = 0.5*(2*shifti+2*shifti*ts2[left]**2/shifts**2)/uhelp
        expu = exp(-(uhelp-p0)/H)
        temp = sqrt(pi/2)*B*(H+2*uhelp)*expu/(2*sqrt(H*uhelp**3))*dudti1
        res = -1+B*sqrt(pi*H/(2*uhelp))*expu-(ti1-z2)*temp
        return res

    def gen_guess(first):
        guess = first*(ts1-z2)/sqrt((ts1-z2)**2+ts2**2)*(p0+TOL)+z2 #Adding TOL here makes phi run 20x faster
        signs = where(ts1-z2 > 0, 1*first, -1*first)
        func = phi(guess, array(range(0,r.size)))
        left = where(func*signs < 0)[0]
        while left.size > 0: #Find good guess by stepping images into the lens
#            print min(func*signs), left.size, 'guess'
            guess[left] = (guess[left]-z2)*0.99+z2
            func[left] = phi(guess[left], left)
            left = where(func*signs < 0)[0]
        return guess

#########
#Stuff
    z2 = abs(zin)
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
    shifti = ti1-z2
    shifts = ts1-z2
    ti2 = shifti*ts2/shifts
    
    Au= sqrt(shifti**2+ti2**2)
    expu = exp(-(Au-p0)/H)
    Aphi = -B*sqrt(pi*H/(2*Au))*expu
    u2phi= B*sqrt(pi/2)*sqrt(H/Au)*(1/2.+Au/H)*expu
           
    A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
    A   = 1/A
    W   = where(Au > p0, 1., 0.)
    integrand = 2*r*I(r)*abs(A)*W #Magnification has to be positive, right?
#    plt.plot(A)
#    plt.plot(Au)
#    plt.plot(W)
#    plt.show()
#    plt.plot(ts1, ts2, '.')
#    plt.plot(ti1, ti2, '.')

#    Aold = A

#########
#Caustics
    if 1-sqrt(pi/2)*sqrt(H/p0)*B < 0:
        guess = gen_guess(-1.)
        ti1b = newrap(phi, dphi, guess.copy())
        shifti = ti1b-z2
        shifts = ts1-z2
        ti2b = shifti*ts2/shifts
        
        Au= sqrt(shifti**2+ti2b**2)
        expu = exp(-(Au-p0)/H)
        Aphi = -B*sqrt(pi*H/(2*Au))*expu
        u2phi= B*sqrt(pi/2)*sqrt(H/Au)*(1/2.+Au/H)*expu

        A   = 1 + 2*Aphi + Aphi**2 + u2phi*(1+Aphi)
        A   = 1/A
        W   = where(Au > p0, 1., 0.)
        integrand += 2*r*I(r)*abs(A)*W #DON'T KNOW WHY THIS IS PLUS, MINUS MAKES MORE SENSE FOR DIAMOND RING
#        plt.plot(Aold)
#        plt.plot(A)
#        plt.plot(Aold+A)
#        plt.plot(W)
#        plt.plot(Au)
#        plt.plot(where(sqrt((ts1-z2)**2+ts2**2)>p0,1.,0.))
#        plt.scatter(r*cos(theta), r*sin(theta), s=abs(A)+abs(Aold))
#        plt.show()
#        plt.plot(ti1b, ti2b, '.')
#        temp = linspace(0,2*pi,10000)
#        plt.plot(p0*cos(temp)+z2,p0*sin(temp))
#        plt.plot((p0+H)*cos(temp)+z2,(p0+H)*sin(temp))
#        plt.show()
#        plt.plot(2*r*I(r)*A*W)
#        plt.show()

#########
#Integrate
    integrand = where(isnan(integrand) | isinf(integrand), 0, integrand) #Should be fine
    integrand = integrand.reshape((nt, nr))
    theta = theta.reshape((nt, nr))
#    plt.plot(theta, integrand)
#    plt.show()
    r = r.reshape((nt, nr))
    integrand = simps(integrand,theta,axis=0)
    F = simps(integrand,r[0],axis=0)
    F = F/norm

    return F

#########
# Constants, cgs
kB = 1.380658e-16 # Boltzmann constant
mu = 2 # Mean molecular wight, probably H2 for Jovians
mH = 1.6733e-24 # Hydrogen mass

#########
# Beta Pictoris
Rstar = 1.253e11
# Beta Pictoris b
# Observables
TT = 1600 # Temperature
rho0 = 3e-6 # Density at optical depth 1. Pretty much taken this out of nowhere.
alpha = 1.2 # Refractive index, shouldn't vary too much
gg = 10**3.8 # Surface gravity
MM = 11*1.898e30 # Mass
RR = 1.65*6.9911e+9/Rstar # Radius
Dls = 9*1.496e13/Rstar # Orbital distance

# Parameters
HH = kB*TT/(gg*mu*mH)/Rstar
BB = 2*alpha*rho0*Dls/HH
print HH, BB, RR/HH, HH*Rstar



##########
## Sun
#Rstar = 6.957e10
## Jupiter
## Observables
#TT = 128 # Temperature
#rho0 = 3.5e-5 # Density at optical depth 1. Pretty much taken this out of nowhere.
#alpha = 1.2 # Refractive index, shouldn't vary too much
#gg = 2479 # Surface gravity
#MM = 1.898e30 # Mass
#RR = 7.1e+9/Rstar # Radius
#Dls = 5.2*1.496e13/Rstar # Orbital distance
#
## Parameters
#HH = kB*TT/(gg*mu*mH)/Rstar
#BB = 2*alpha*rho0*Dls/HH
#print HH, BB, RR/HH, HH*Rstar


#########
#For testing
t=linspace(0.,1.3,200)
z=sqrt(t**2+(2*0.084)**2) # hui02
#z=t

x=zeros(z.size)
for i in range(0,z.size):
    print i # g1, g2, B, H, p0, nr, nt
    # g1 0.381445454545, c2 = g1+2*g2, g1 = c2+2*c4
    # g2 0.269718181818, c4 = -g2,     g2 = -c4
    x[i] = refract_simps(z[i], 0.35, 0.225, 40.3, 0.084/117.3, 0.084, 800, 800) # hui02
#    x[i] = refract_simps(z[i], 0.35, 0.225, BB, HH, RR, 800, 800) # hui02

x=concatenate((x[1:][::-1],x))
z=concatenate((-z[1:][::-1],z))
t=concatenate((-t[1:][::-1],t))
savetxt('/Users/silver/Dropbox/sci/project_exo/models/hui02_srcrf.dat', c_[t,x])
#plt.plot(t,x,'ok')
plt.plot(t,x)
plt.show()
