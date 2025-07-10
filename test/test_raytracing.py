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
from scipy import constants,optimize
from pythonradex import molecule


x_sky = np.linspace(-100,100,50)*constants.au
y_sky = x_sky.copy()
z_sky = x_sky.copy()
v = np.linspace(-10,10,60)*constants.kilo
grid = {'x_sky':x_sky,'y_sky':y_sky,'z_sky':z_sky,'v':v}
M = 2e30
u = constants.physical_constants['atomic mass constant'][0]

mole = molecule.Molecule(datafilepath='c+.dat')
m_Cplus = 12*u
trans_index = 0
trans = mole.rad_transitions[trans_index]


def T_Boltzmann(x,y,z):
    T0 = 50
    r0 = 50*constants.au
    r = np.sqrt(x**2+y**2)
    return (r/r0)**-0.5 * T0

def FWHM_v(x,y,z):
    sigma_v = np.sqrt(constants.k*T_Boltzmann(x=x,y=y,z=z)/m_Cplus)
    return 2*np.sqrt(2*np.log(2))*sigma_v

constant_FWHM = 1*constants.kilo
def FWHM_v_constant(x,y,z):
    shape = np.broadcast(x,y,z).shape
    return np.ones(shape)*constant_FWHM

def x1(x,y,z):
    T = T_Boltzmann(x=x,y=y,z=z)
    Z = mole.Z(T)
    low = trans.low
    return low.g*np.exp(-low.E/(constants.k*T))/Z

def x2(x,y,z):
    T = T_Boltzmann(x=x,y=y,z=z)
    Z = mole.Z(T)
    up = trans.up
    return up.g*np.exp(-up.E/(constants.k*T))/Z

#following x1 and x2 are dependent on the grid, so I don't use them
# def x1(x,y,z):
#     T = T_Boltzmann(x=x,y=y,z=z)
#     return T/np.max(T)*0.2

# def x2(x,y,z):
#     lower_level = x1(x=x,y=y,z=z)
#     return lower_level**1.5

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

# def v_y(x,y,z):
#     r = np.sqrt(x**2+y**2)
#     v0 = np.sqrt(constants.G*M/r)
#     phi = np.arctan2(y,x)
#     return v0*np.cos(phi)

velocity_yz = raytracing.Raytracing.generate_Keplerian_velocity_yz(Mstar=M)

std_kwargs_T = {'grid':grid,'T_Boltzmann':T_Boltzmann,'number_density':number_density,
                'velocity_yz':velocity_yz,'FWHM_v':FWHM_v,'mole':mole,
                'transition_index':0,'inclination_xyz':np.pi/3,'verbose':False}
std_kwargs_x1x2 = std_kwargs_T.copy()
std_kwargs_x1x2['T_Boltzmann'] = None
std_kwargs_x1x2['x1'] = x1
std_kwargs_x1x2['x2'] = x2
std_kwargs = {'T':std_kwargs_T,'x1x2':std_kwargs_x1x2}


class TestRaytracing(unittest.TestCase):
    
    d = 1*constants.parsec

    def test_compute_spatial_interval(self):
        a = np.array((-2,0,1,1.5))
        expected_da = np.array((2,1.5,0.75,0.5))
        da = raytracing.Raytracing.compute_spatial_interval(a)
        self.assertTrue(np.all(da==expected_da))
        bad_arrays = [np.array((-2,3,3,4)),np.array((3,2,4,5,6,7)),np.array((2,1,0,-2))]
        for bad_array in bad_arrays:
              self.assertRaises(AssertionError,raytracing.Raytracing.compute_spatial_interval,
                                bad_array)

    def test_init_T_x1_x2(self):
        T_None_kwargs = std_kwargs_T.copy()
        T_None_kwargs['T_Boltzmann'] = None
        x1_x2_not_None_kwargs = std_kwargs_T.copy()
        x1_x2_not_None_kwargs['x1'] = 1
        x1_x2_not_None_kwargs['x2'] = 1
        for k in (T_None_kwargs,x1_x2_not_None_kwargs):
            self.assertRaises(AssertionError,raytracing.Raytracing,**k)

    def test_grid_setup(self):
        bad_ys = [np.array((-2,3,3,4)),np.array((3,2,4,5,6,7)),np.array((2,1,0,-2))]
        for y in bad_ys:
            bad_grid = grid.copy()
            bad_grid['y_sky'] = y
            for kwargs in std_kwargs.values():
                k = kwargs.copy()
                k['grid'] = bad_grid
                self.assertRaises(AssertionError,raytracing.Raytracing,**k)

    def test_inclination_optically_thin(self):
        test_inclinations = [0,np.pi/8,np.pi/4,np.pi/2]
        total_fluxes = np.empty(len(test_inclinations))
        for key,kwargs in std_kwargs.items():
            test_kwargs = kwargs.copy()
            test_kwargs['number_density'] = thin_n
            for i,inc in enumerate(test_inclinations):
                test_kwargs['inclination_xyz'] = inc
                raytrace = raytracing.Raytracing(**test_kwargs)
                raytrace.raytrace()
                #print(f'tau_nu: {np.min(raytrace.tau_nu):.3g} - {np.max(raytrace.tau_nu):.3g}')
                total_fluxes[i] = raytrace.total_flux(distance=self.d)
            #print(total_fluxes)
            self.assertTrue(np.allclose(total_fluxes,total_fluxes[0],atol=0,rtol=1e-2))

    def test_optically_thin_flux(self):
        for kwargs in std_kwargs.values():
            test_kwargs = kwargs.copy()
            test_kwargs['number_density'] = thin_n
            raytrace = raytracing.Raytracing(**test_kwargs)
            raytrace.raytrace()
            flux = raytrace.total_flux(distance=self.d)
            #print(f'tau_nu: {np.min(raytrace.tau_nu):.3g} - {np.max(raytrace.tau_nu):.3g}')
            #I am calculating the analytical solution by putting the disk in the
            #x,y,z (i.e. "edge-on"), but since it is optically thin, it should
            #not matter
            x3D,y3D,z3D = x_sky[:,None,None],y_sky[None,:,None],z_sky[None,None,:]
            n3D = thin_n(x=x3D,y=y3D,z=z3D)
            #N = np.trapezoid(np.trapezoid(np.trapezoid(n,z_sky,axis=-1),y_sky,axis=-1),x_sky)
            if test_kwargs['T_Boltzmann'] is not None:
                T3D = test_kwargs['T_Boltzmann'](x=x3D,y=y3D,z=z3D)
                Z = mole.Z(T3D)
                trans = raytrace.transition
                up = trans.up
                upper_level_pop = up.g*np.exp(-up.E/(constants.k*T3D))/Z
            else:
                                       
                upper_level_pop = x2(x=x3D,y=y3D,z=z3D)
            N2 = np.trapezoid(np.trapezoid(np.trapezoid(n3D*upper_level_pop,
                                                        z_sky,axis=-1),y_sky,axis=-1),x_sky)
            expected_flux = N2*trans.A21*trans.Delta_E/(4*np.pi*self.d**2)
            #print(f'computed flux = {flux:.3g}, analytical flux = {expected_flux:.3g}')
            self.assertTrue(np.isclose(flux,expected_flux,rtol=1e-2,atol=0))

    @staticmethod
    def generate_level_pops(T):
        def x1(x,y,z):
            Txyz = T(x=x,y=y,z=z)
            Z = mole.Z(Txyz)
            low = trans.low
            return low.g*np.exp(-low.E/(constants.k*Txyz))/Z
        def x2(x,y,z):
            Txyz = T(x=x,y=y,z=z)
            Z = mole.Z(Txyz)
            up = trans.up
            return up.g*np.exp(-up.E/(constants.k*Txyz))/Z
        return x1,x2

    def test_level_population_input(self):
        T_raytracing = raytracing.Raytracing(**std_kwargs['T'])
        
        x1_x2_kwargs = std_kwargs['T'].copy()
        x1_x2_kwargs['T_Boltzmann'] = None
        x1_x2_kwargs['x1'] = x1
        x1_x2_kwargs['x2'] = x2
        x1_x2_raytracing = raytracing.Raytracing(**x1_x2_kwargs)
        cubes = []
        for raytrace in (T_raytracing,x1_x2_raytracing):
            raytrace.raytrace()
            cubes.append(raytrace.cube)
        self.assertTrue(np.allclose(*cubes,rtol=1e-3,atol=0))

    def test_negative_tau(self):
        def neg_T(x,y,z):
            out_shape = np.broadcast(x,y,z).shape
            return -20*np.ones(out_shape)
        x1,x2 = self.generate_level_pops(neg_T)
        x1_x2_kwargs = std_kwargs['x1x2'].copy()
        x1_x2_kwargs['x1'] = x1
        x1_x2_kwargs['x2'] = x2
        neg_tau_not_allowed = raytracing.Raytracing(
                                 **x1_x2_kwargs,allow_negative_tau=False)
        neg_tau_allowed = raytracing.Raytracing(
                                  **x1_x2_kwargs,allow_negative_tau=True)
        neg_tau_allowed.raytrace()
        self.assertRaises(AssertionError,neg_tau_not_allowed.raytrace)

    def test_optically_thick(self):
        def n_thick(x,y,z):
            return number_density(x=x,y=y,z=z)*1e3
        T0 = 45
        def T_const(x,y,z):
            out_shape = np.broadcast(x,y,z).shape
            return T0*np.ones(out_shape)
        kwargs = std_kwargs['T'].copy()
        kwargs['T_Boltzmann'] = T_const
        kwargs['number_density'] = n_thick
        kwargs['inclination_xyz'] = 0
        model = raytracing.Raytracing(**kwargs)
        model.raytrace()
        expected_peak = raytracing.Raytracing.B_nu(T=T0,nu=model.transition.nu0)#W/m2/Hz/sr
        expected_peak *= model.transition.nu0/constants.c
        peak = np.max(model.cube[x_sky.size//2,z_sky.size//2,:])
        #print(np.max(model.tau_nu))
        #print(peak,expected_peak)
        self.assertTrue(np.isclose(peak,expected_peak,rtol=1e-3,atol=0))

    @staticmethod
    def change_velocity_yz(kwargs,velocity_yz):
        out = kwargs.copy()
        out['velocity_yz'] = velocity_yz
        return out

    def test_velocity_field(self):
        def fit_Gaussian(v,data):
            scaling = np.max(data)
            def model(params):
                return params[0]*np.exp(-(v-params[1])**2/(2*params[2]**2))
            def residual(params):
                return np.sum((model(params=params)-data/scaling)**2)
            x0 = [1,0,constant_FWHM/(2*np.sqrt(2*np.log(2)))]
            fit = optimize.minimize(fun=residual,x0=x0)
            assert fit.success
            return [fit.x[0]*scaling,fit.x[1],fit.x[2]]
        vz = 1*constants.kilo
        Kep_field = raytracing.Raytracing.generate_Keplerian_velocity_yz(Mstar=M)
        Kep_kwargs = std_kwargs['T'].copy()
        Kep_kwargs['velocity_yz'] = Kep_field
        Kep_kwargs['inclination_xyz'] = 0
        Kep_kwargs['FWHM_v'] = FWHM_v_constant
        Kep_kwargs['number_density'] = thin_n
        def Kep_field_plus_outflow(x,y,z):
            Kep = Kep_field(x=x,y=y,z=z)
            out_shape = np.broadcast(x,y,z).shape
            return [Kep[0],vz*np.ones(out_shape)]
        def only_outflow(x,y,z):
            shape = np.broadcast(x,y,z).shape
            return [np.zeros(shape),vz*np.ones(shape)]
        Kep_vz_kwargs = self.change_velocity_yz(
                             kwargs=Kep_kwargs,velocity_yz=Kep_field_plus_outflow)
        vz_kwargs = self.change_velocity_yz(
                              kwargs=Kep_kwargs,velocity_yz=only_outflow)
        Kep_model = raytracing.Raytracing(**Kep_kwargs)
        Kep_vz_model = raytracing.Raytracing(**Kep_vz_kwargs)
        vz_model = raytracing.Raytracing(**vz_kwargs)
        for m in (Kep_model,Kep_vz_model,vz_model):
            m.raytrace()
            assert np.max(m.tau_nu) < 1e-2
            #print(np.max(m.tau_nu))
        #since the disk is edge-on, vz should not have any effect
        self.assertTrue(np.all(Kep_model.cube==Kep_vz_model.cube))
        #for the disk with only outflow, the spectrum should be a Gaussian
        #with width equal to the line width
        #vz_model.plot_spectrum()
        vz_spec_fit = fit_Gaussian(v=v,data=vz_model.spectrum)
        #print(vz_spec_fit)
        dv = np.diff(v)
        assert np.allclose(dv[0],dv,atol=0,rtol=1e-6)
        dv = dv[0]
        #I expect a Gaussian centred at 0 and known FWHM
        self.assertTrue(np.abs(vz_spec_fit[1])<dv/10)
        self.assertTrue(np.isclose(vz_spec_fit[2]*2*np.sqrt(2*np.log(2)),constant_FWHM,
                                   atol=0,rtol=1e-3))
        #let's incline the vz model to face-on so that the flow comes towards the observer
        for inclination in np.radians([20,50,90]):
            vz_inclined_kwargs = vz_kwargs.copy()
            vz_inclined_kwargs['inclination_xyz'] = inclination
            vz_inclined_model = raytracing.Raytracing(**vz_inclined_kwargs)
            vz_inclined_model.raytrace()
            vz_inclined_fit = fit_Gaussian(v=v,data=vz_inclined_model.spectrum)
            #the spectrum should be centered at -vz*sin(inclination) and have
            #width
            #print(vz_edgeon_fit[1], -vz*np.sin(inclination))
            self.assertTrue(np.abs(vz_inclined_fit[1] - -vz*np.sin(inclination))<dv/10)
            #check also FWHM:
            self.assertTrue(np.isclose(vz_inclined_fit[2]*2*np.sqrt(2*np.log(2)),
                                       constant_FWHM,atol=0,rtol=1e-3))


if __name__ == '__main__':
    unittest.main(verbosity=2)