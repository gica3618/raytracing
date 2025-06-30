#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:50:18 2019

@author: gianni
"""

import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
import raytracing
from scipy import constants
import sys
sys.path.append('/home/gianni/science/projects/code/pythonradex')
from pythonradex import molecule,atomic_transition
sys.path.append('/home/gianni/science/projects/code/Lime')
import pyLime


x = np.linspace(-120,120,150)*constants.au
y = np.linspace(-120,120,200)*constants.au
z = np.linspace(-70,70,100)*constants.au
v = np.linspace(-15,15,40)*constants.kilo
grid = {'x_sky':x,'y_sky':y,'z_sky':z,'v':v}
M = 2e30
line_profile_cls = atomic_transition.GaussianLineProfile
width_v = 1.5*constants.kilo
T0 = 50
distance = 10*constants.parsec
n0_range = np.logspace(0,6,10)/constants.centi**3
def T_Boltzmann(x,y,z):
    return np.ones_like(x*y*z)*T0
def velocity_yz(x,y,z):
    r = np.sqrt(x**2+y**2)
    v0 = np.sqrt(constants.G*M/r)
    phi = np.arctan2(y,x)
    return [v0*np.cos(phi),np.zeros_like(r)]
data_filepath = 'c+.dat'
mole = molecule.EmittingMolecule(
                 datafilepath=data_filepath,line_profile_type='Gaussian',
                 width_v=width_v)
transition_index = 0
inclination_xyz = np.radians(40)
verbose = False


flux_ratios = []
for n0 in n0_range:
    print('considering n0={:g} cm-3'.format(n0*constants.centi**3))
    def number_density(x,y,z):
        r = np.sqrt(x**2+y**2)
        H = 10*constants.au
        r0 = 50*constants.au
        sigma = 10*constants.au
        x0 = 50*constants.au
        y0 = 0*constants.au
        r_blob = np.sqrt((x-x0)**2+(y-y0)**2)
        sigma_blob = 10*constants.au
        n = n0*np.exp(-(r-r0)**2/(2*sigma**2))*np.exp(-z**2/(2*H**2))\
            + 4*n0*np.exp(-r_blob**2/(2*sigma_blob**2))*np.exp(-z**2/(2*H**2))
        return n
    kwargs = {'grid':grid,'T_Boltzmann':T_Boltzmann,'number_density':number_density,
              'velocity_yz':velocity_yz,'atom':mole,
              'transition_index':transition_index,
              'inclination_xyz':inclination_xyz,'verbose':verbose}
    raytrace = raytracing.Raytracing(**kwargs)
    raytrace.raytrace()
    raytrace_flux = raytrace.total_flux(distance=distance)
    raytrace_title = 'n0={:g}cm-3, raytrace'.format(n0*constants.centi**3)
    raytrace.plot_max_tau_nu(title=raytrace_title)
    raytrace.plot_pv(title=raytrace_title)
    raytrace.plot_mom0(title=raytrace_title)
    raytrace.plot_spectrum(title=raytrace_title)
    print('max opt depth raytrace: {:g}'.format(np.max(raytrace.tau_nu)))
    print('total flux raytrace: {:g} W/m2'.format(raytrace_flux))

    axes = axes = {'x':x,'y':y,'z':z}
    T = np.ones((x.size,y.size,z.size))*T0
    n = number_density(x=x[:,None,None],y=y[None,:,None],z=z[None,None,:])
    colliders = [pyLime.Collider(name='e',density=n),]
    radiating_species = [pyLime.RadiatingSpecie(moldatfile=data_filepath,density=n),]
    x3D,y3D,z3D = np.meshgrid(x,y,z,indexing='ij')
    r = np.sqrt(x3D**2+y3D**2)
    vkep = np.sqrt(constants.G*M/r)
    velocity = {'x':-y3D/r*vkep,'y':x3D/r*vkep}
    radius = 200*constants.au
    broadening_param = width_v / (2*np.sqrt(np.log(2)))
    img_res = np.min((np.diff(x)[0],np.diff(z)[0]))
    general_img_kwargs = {'nchan':v.size,'velres':np.diff(v)[0],'trans':0,'pxls':200,
                          'imgres':img_res/distance,'distance':distance,
                          'phi':0,'units':'2 4'}
    images = [pyLime.LimeImage(theta=inclination_xyz,filename='test_c+.fits',molI=0,
                               **general_img_kwargs),]
    n_threads = 10
    n_solve_iters = 14
    level_population_filename = None
    lte_only = True
    test_lime = pyLime.Lime(axes=axes,T=T,colliders=colliders,
                            radiating_species=radiating_species,velocity=velocity,
                            radius=radius,broadening_param=broadening_param,
                            images=images,n_threads=n_threads,n_solve_iters=n_solve_iters,
                            level_population_filename=level_population_filename,
                            lte_only=lte_only)
    test_lime.run()
    lime_flux_image = pyLime.LimeFitsOutputFluxSI('test_c+_SI.fits')
    lime_flux_image.compute_projections()
    lime_title = 'n0={:g}cm-3, lime'.format(n0*constants.centi**3)
    lime_flux_image.plot_mom0(title=lime_title)
    lime_flux_image.plot_pv(title=lime_title)
    lime_flux = lime_flux_image.total_flux()
    print('flux from LIME (LTE mode): {:g} W/m2'.format(lime_flux))
    flux_ratio = raytrace_flux/lime_flux
    flux_ratios.append(flux_ratio)
    print('flux ratio: {:g}'.format(flux_ratio))
    print('\n')
    lime_tau_image = pyLime.LimeFitsOutputTau('test_c+_Tau.fits')
    lime_tau_image.compute_max_map()
    lime_tau_image.plot_max_map(title=lime_title)
print(flux_ratios)