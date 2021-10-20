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
sys.path.append('/home/gianni/science/projects/code/pythonradex')
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
def T_LTE(x,y,z):
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
std_kwargs = {'grid':grid,'T_LTE':T_LTE,'number_density':number_density,
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
              self.assertRaises(AssertionError,raytracing.compute_spatial_interval,
                                bad_array)

    def test_init_T_x1_x2(self):
        T_None_kwargs = std_kwargs.copy()
        T_None_kwargs['T_LTE'] = None
        x1_x2_not_None_kwargs = std_kwargs.copy()
        x1_x2_not_None_kwargs['x1'] = 1
        x1_x2_not_None_kwargs['x2'] = 1
        for k in (T_None_kwargs,x1_x2_not_None_kwargs):
            self.assertRaises(AssertionError,raytracing.Raytracing,**k)

    def test_inclination_optically_thin(self):
        test_inclinations = [0,np.pi/8,np.pi/4,np.pi/2]
        total_fluxes = np.empty(len(test_inclinations))
        test_kwargs = std_kwargs.copy()
        for i,inc in enumerate(test_inclinations):
            test_kwargs['inclination'] = inc
            raytrace = raytracing.Raytracing(**test_kwargs)
            raytrace.raytrace()
            raytrace.compute_spec()
            total_fluxes[i] = raytrace.total_flux(distance=self.d)
        self.assertTrue(np.allclose(total_fluxes,total_fluxes[0]))

    def test_optically_thin_flux(self):
        test_kwargs = std_kwargs.copy()
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
        print(flux,expected_flux)
        self.assertTrue(np.isclose(flux,expected_flux,rtol=1e-2,atol=0))

    def test_transition_selection(self):
        raytrace = raytracing.Raytracing(**std_kwargs)
        self.assertEqual(raytrace.transition.name,transition_name)

    def get_x1_x2_kwargs(self,const_x1,const_x2):
        def x1(x,y,z):
            return np.ones_like(x*y*z)*const_x1
        def x2(x,y,z):
            return np.ones_like(x*y*z)*const_x2
        x1_x2_kwargs = std_kwargs.copy()
        x1_x2_kwargs['T_LTE'] = None
        x1_x2_kwargs['x1'] = x1
        x1_x2_kwargs['x2'] = x2
        return x1_x2_kwargs

    def test_level_population_input(self):
        T_raytracing = raytracing.Raytracing(**std_kwargs)
        partition_func = mole.Z(T0)
        upper_level_fraction = T_raytracing.transition.up.LTE_level_pop(
                                                    Z=partition_func,T=T0)
        lower_level_fraction = T_raytracing.transition.low.LTE_level_pop(
                                                    Z=partition_func,T=T0)
        x1_x2_kwargs = self.get_x1_x2_kwargs(const_x1=lower_level_fraction,
                                              const_x2=upper_level_fraction)
        x1_x2_raytracing = raytracing.Raytracing(**x1_x2_kwargs)
        fluxes = []
        for raytrace in (T_raytracing,x1_x2_raytracing):
            raytrace.raytrace()
            raytrace.compute_spec()
            fluxes.append(raytrace.total_flux(distance=self.d))
        self.assertTrue(np.allclose(fluxes,fluxes[0],rtol=1e-5,atol=0))

    def test_negative_tau(self):
        std_raytracing = raytracing.Raytracing(**std_kwargs)
        partition_func = mole.Z(T0)
        negative_Tex = -20
        x1 = std_raytracing.transition.low.LTE_level_pop(Z=partition_func,T=negative_Tex)
        x2 = std_raytracing.transition.up.LTE_level_pop(Z=partition_func,T=negative_Tex)
        x1_x2_kwargs = self.get_x1_x2_kwargs(const_x1=x1,const_x2=x2)
        neg_tau_not_allowed = raytracing.Raytracing(**x1_x2_kwargs,allow_negative_tau=False)
        neg_tau_allowed = raytracing.Raytracing(**x1_x2_kwargs,allow_negative_tau=True)
        neg_tau_allowed.raytrace()
        self.assertRaises(AssertionError,neg_tau_not_allowed.raytrace)
        