#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:43:55 2025

@author: gianni
"""

#here in investigate why I got inconsistent results when using a certain
#function for x2. There are two problems:
# 1) different inclinations give different integrated fluxes, althought the disk
# is optically thin
#2) for a low inclination disk, the integrated flux depends on the grid
#strangly, a finer and finer grid gives lower and lower fluxes for low-inclination
#disks, while for edge-on disks, no dependence on grid is seen, and the flux is
#much *larger* than in the face-on case.
#also, if I use a Boltzmann distribution for x1, no issues are seen
#it turns out that x2 (and thus also the product n*x2) is dependent on the grid,
#but I was unable to understand why


import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
import raytracing
import numpy as np
from scipy import constants
from pythonradex import molecule
import matplotlib.pyplot as plt


M = 2e30
width_v = 1.5*constants.kilo
mole = molecule.EmittingMolecule(
                 datafilepath='c+.dat',line_profile_type='Gaussian',
                 width_v=width_v)
trans = mole.rad_transitions[0]


def T_Boltzmann(x,y,z):
    T0 = 50
    r0 = 50*constants.au
    r = np.sqrt(x**2+y**2)*np.ones_like(x*y*z)
    return (r/r0)**-0.5 * T0

def x1(x,y,z):
    T = T_Boltzmann(x=x,y=y,z=z)
    return T/np.max(T)*0.2
    # return np.ones_like(x*y*z)*0.2

def x2(x,y,z):
    lower_level = x1(x=x,y=y,z=z)
    return lower_level**1.5
    # return np.ones_like(x*y*z)*0.1

# def x1(x,y,z):
#     T = T_Boltzmann(x=x,y=y,z=z)
#     Z = mole.Z(T)
#     low = trans.low
#     return low.g*np.exp(-low.E/(constants.k*T))/Z * T/np.max(T)

# def x2(x,y,z):
#     T = T_Boltzmann(x=x,y=y,z=z)
#     Z = mole.Z(T)
#     up = trans.up
#     return up.g*np.exp(-up.E/(constants.k*T))/Z


def number_density(x,y,z):
    r = np.sqrt(x**2+y**2)
    H = 15*constants.au
    n0 = 1000/constants.centi**3
    r0 = 50*constants.au
    sigma = 10*constants.au
    n = n0*np.exp(-(r-r0)**2/(2*sigma**2))*np.exp(-z**2/(2*H**2))
    return n

def thin_n(x,y,z):
    return number_density(x=x,y=y,z=z)/1e6

def v_y(x,y,z):
    r = np.sqrt(x**2+y**2)
    v0 = np.sqrt(constants.G*M/r)
    phi = np.arctan2(y,x)
    return v0*np.cos(phi)

distance = 1*constants.parsec
inclination = 0
print(f'inclination: {np.degrees(inclination):.3g} deg')

fluxes = []
factors = np.linspace(1,20,20)
for factor in factors:
    print(factor)
    x = np.linspace(-100,100,int(20*factor))*constants.au
    y = x.copy()
    z = x.copy()
    v = np.linspace(-10,10,60)*constants.kilo
    grid = {'x':x,'y':y,'z':z,'v':v}
    
    x3D,y3D,z3D = x[:,None,None],y[None,:,None],z[None,None,:]
    # x3D_p = x3D
    # y3D_p = y3D*np.sin(inclination) + z3D*np.cos(inclination)
    # z3D_p = -y3D*np.cos(inclination) + z3D*np.sin(inclination)
    upper_level_pop = x2(x=x3D,y=y3D,z=z3D)
    print(f'x2 stats: mean = {np.mean(upper_level_pop):.3g}, std = {np.std(upper_level_pop):.3g}')
    n3D = thin_n(x=x3D,y=y3D,z=z3D)
    print(f'n stats: mean = {np.mean(n3D):.3g}, std = {np.std(n3D):.3g}')
    product = upper_level_pop*n3D
    N2 = np.trapezoid(np.trapezoid(np.trapezoid(n3D*upper_level_pop,
                                                z,axis=-1),y,axis=-1),x)
    print(f'n*x2 stats: mean = {np.mean(product):.3g}, std = {np.std(product):.3g}, int={N2:.3g}')
    expected_flux = N2*trans.A21*trans.Delta_E/(4*np.pi*distance**2)
    print(f'expected flux: {expected_flux:.3g} W/m2')
    
    # fig,axes = plt.subplots(2,2)
    # axes[0,0].plot(x/constants.au,x2(x=x,y=0,z=0))
    # axes[0,1].plot(y/constants.au,x2(x=0,y=y,z=0))
    # axes[1,0].plot(x/constants.au,x2(x=x,y=0,z=0)*number_density(x=x,y=0,z=0))
    # axes[1,1].plot(z/constants.au,x2(x=50*constants.au,y=50*constants.au,z=z)
    #                *number_density(x=50*constants.au,y=50*constants.au,z=z))
    
    # model = raytracing.Raytracing(
    #            grid=grid,T_Boltzmann=None,number_density=thin_n,x1=x1,x2=x2,
    #            v_y=v_y,atom=mole,transition_index=0,zsym=False,inclination=inclination,
    #            verbose=False)
    # model.raytrace()
    # model.compute_spec()
    # model.plot_mom0()
    # model.plot_max_tau_nu()
    # model.plot_spectrum()
    # total_flux = model.total_flux(distance=distance)
    # fluxes.append(total_flux)
    # print(f'flux = {total_flux:.3g} W/m2')

# fig,ax = plt.subplots()
# ax.plot(factors,fluxes)