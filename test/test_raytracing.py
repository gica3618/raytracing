#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:04:47 2019

@author: gianni
"""

import unittest
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
import raytracing
from scipy import constants

import sys
sys.path.append('/home/cataldi/Applications/pythonradex')
from pythonradex import molecule,atomic_transition
x = np.linspace(-100,100,50)*constants.au
y = x.copy()
z = np.linspace(-30,30,50)*constants.au
v = np.linspace(-10,10,60)*constants.kilo
grid = {'x':x,'y':y,'z':z,'v':v}
M = 2e30
line_profile_cls = atomic_transition.GaussianLineProfile
width_v = 1.5*constants.kilo
T0 = 50
def Tex(x,y,z):
    return np.ones_like(x*y*z)*T0
def number_density(x,y,z):
    r = np.sqrt(x**2+y**2)
    H = 10*constants.au
    n0 = 1000/constants.centi**3
    r0 = 50*constants.au
    sigma = 10*constants.au
    n = n0*np.exp(-(r-r0)**2/(2*sigma**2))*np.exp(-z**2/(2*H**2))
    return n
def velocity_field(x,y,z):
    r = np.sqrt(x**2+y**2)
    v0 = np.sqrt(constants.G*M/r)
    phi = np.arctan2(y,x)
    return [-v0*np.sin(phi),v0*np.cos(phi),np.zeros_like(r)]
mole = molecule.EmittingMolecule.from_LAMDA_datafile(
                 datafilepath='c+.dat',line_profile_cls=line_profile_cls,
                 width_v=width_v)
transition_name = '1-0'
zsym = False
inclination = np.pi/2
verbose = False
kwargs = {'grid':grid,'Tex':Tex,'number_density':number_density,
          'velocity_field':velocity_field,'atom':mole,
          'transition_name':transition_name,'zsym':zsym,
          'inclination':inclination,'verbose':verbose}


class TestRaytracing(unittest.TestCase):
    
    d = 10*constants.parsec
    
    def test_compute_spatial_interval(self):
        a = np.array((-2,0,1,1.5))
        expected_da = np.array((2,1.5,0.75,0.5))
        da = raytracing.compute_spatial_interval(a)
        self.assertTrue(np.all(da==expected_da))
        bad_arrays = [np.array((-2,3,3,4)),np.array((3,2,4,5,6,7)),np.array((2,1,0,-2))]
        for bad_array in bad_arrays:
             self.assertRaises(AssertionError,raytracing.compute_spatial_interval,bad_array)

    def test_inclination_optically_thin(self):
        test_inclinations = [0,np.pi/8,np.pi/4,np.pi/2]
        total_fluxes = np.empty(len(test_inclinations))
        test_kwargs = kwargs.copy()
        for i,inc in enumerate(test_inclinations):
            test_kwargs['inclination'] = inc
            raytrace = raytracing.Raytracing(**test_kwargs)
            raytrace.raytrace()
            raytrace.compute_spec()
            total_fluxes[i] = raytrace.total_flux(distance=self.d)
        self.assertTrue(np.allclose(total_fluxes,total_fluxes[0]))

    def test_optically_thin_flux(self):
        test_kwargs = kwargs.copy()
        def thin_n(x,y,z):
            return number_density(x=x,y=y,z=z)/1e4
        test_kwargs['number_density'] = thin_n
        raytrace = raytracing.Raytracing(**test_kwargs)
        raytrace.raytrace()
        raytrace.compute_spec()
        flux = raytrace.total_flux(distance=self.d)
        n = thin_n(x=x[:,None,None],y=y[None,:,None],z=z[None,None,:])
        N = np.trapz(np.trapz(np.trapz(n,z,axis=-1),y,axis=-1),x)
        Z = mole.Z(T0)
        trans = raytrace.transition
        up = trans.up
        upper_level_pop = up.g*np.exp(-up.E/(constants.k*T0))/Z
        expected_flux = upper_level_pop*N*trans.A21*trans.Delta_E/(4*np.pi*self.d**2)
        self.assertTrue(np.isclose(flux,expected_flux,rtol=1e-2,atol=0))

    def test_transition_selection(self):
        raytrace = raytracing.Raytracing(**kwargs)
        self.assertEqual(raytrace.transition.name,transition_name)