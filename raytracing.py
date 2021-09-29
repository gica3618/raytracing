#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:28:47 2019

@author: gianni
"""

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

def compute_spatial_interval(a):
    '''calculation of the spatial interval associated with every point of array a'''
    assert np.all(np.diff(a)>0),'input array not strictly monotonically increasing'
    intervals = (a[1]-a[0], (a[2:]-a[1:-1])/2+(a[1:-1]-a[:-2])/2, a[-1]-a[-2])
    return np.hstack(intervals)


class Raytracing():

    '''convention: line of sight is along y-axis,i.e. the observers looks along the
    y-axis in the positive direction; the disk is lying in the x-y-plane
    convention for array dimensions: first axis: x; second axis: z; third axis: v
    Previously, the calculation was aborted when the increase in optical depth
    was too large in a single iteration step. However, I found that this does not
    really change the results by comparing to LIME, so I made it optional
    '''
    max_optical_depth_increase = 0.2
    optical_depth_epsilon = 1e-10

    def __init__(self,grid,Tex,number_density,velocity_field,atom,transition_name,
                 zsym=False,inclination=np.pi/2,verbose=False,
                 check_max_optical_depth_increase=False):
        self.grid = grid
        self.Tex = Tex
        self.number_density = number_density
        self.velocity_field = velocity_field
        self.atom = atom
        self.transition_name = transition_name
        transition_number = self.atom.get_rad_transition_number(self.transition_name)
        self.transition = self.atom.rad_transitions[transition_number]
        self.zsym = zsym
        self.inclination = inclination
        if self.zsym:
            assert self.inclination == np.pi/2, 'zsym not allowed for inclined disk'
        self.verbose = verbose
        self.check_max_optical_depth_increase = check_max_optical_depth_increase
        self.grid_setup()

    def compute_nu(self,v):
        return self.transition.nu0*(1-v/constants.c)

    def grid_setup(self):
        self.v = self.grid['v']
        self.V = self.v[np.newaxis,np.newaxis,:]
        self.nu = self.compute_nu(self.v)
        self.NU = self.compute_nu(self.V)
        self.x = self.grid['x']
        self.X = self.x[:,np.newaxis,np.newaxis]
        self.dx = compute_spatial_interval(self.x)
        self.DX = self.dx[:,np.newaxis,np.newaxis]
        self.y = self.grid['y']
        self.dy = compute_spatial_interval(self.y)
        self.z = self.grid['z']
        self.Z = self.z[np.newaxis,:,np.newaxis]
        self.dz = compute_spatial_interval(self.z)
        self.DZ = self.dz[np.newaxis,:,np.newaxis]
        if self.zsym:
            assert (0 in self.z) and np.all(self.z>=0),\
                     'Error: z axis not appropriate for symmetric z calculation'
        assert np.all(np.diff(self.y)>0),\
                          'y axis not appropriate for optical depth calculation'

    def raytrace(self):
        self.cube = np.zeros((self.x.size,self.z.size,self.v.size))
        self.tau_nu = np.zeros_like(self.cube)
        if self.check_max_optical_depth_increase:
            #following array is used to alert if the resolution in y-direction is not
            #sufficient, i.e. tau is increasing too fastly
            previous_tau_nu = np.zeros_like(self.cube)
        # NOTE: it is very important that the iteration goes along the line of
        #sight, away from the telescope! (to sum up optical depth)
        for i,y_i in enumerate(self.y):
            if self.verbose:
                print("Step %i of %i: y=%g AU" %(i+1,self.y.size,y_i/constants.au))
            # prime coordinates to mimick an inclined disk, i.e. from given x,y,z, compute
            # the coordinates x',y',z' in the inclined coordinate system
            X_prime = self.X
            Y_prime = y_i*np.sin(self.inclination) + self.Z*np.cos(self.inclination)
            Z_prime = -y_i*np.cos(self.inclination) + self.Z*np.sin(self.inclination)
            n_x_yi_z = self.number_density(x=X_prime,y=Y_prime,z=Z_prime)
            T_ex_x_yi_z = self.Tex(x=X_prime,y=Y_prime,z=Z_prime)
            partition_func = self.atom.Z(T_ex_x_yi_z)
            upper_level_fraction = self.transition.up.LTE_level_pop(
                                                Z=partition_func,T=T_ex_x_yi_z)
            lower_level_fraction = self.transition.low.LTE_level_pop(
                                                Z=partition_func,T=T_ex_x_yi_z)
            #spontaneous transitions/s/m3:
            transition_density = n_x_yi_z*upper_level_fraction*self.transition.A21
            #the radial velocity, again evaluated at the prime coordinates
            V_R = self.velocity_field(x=X_prime,y=Y_prime,z=Z_prime)[1]\
                   * np.sin(self.inclination)
            NU_R = self.compute_nu(V_R)
            shifted_phi_v = self.transition.line_profile.phi_v(self.V-V_R)
            #intensity in W/m**2/(m/s)/sr at position x,z:
            intensity = (transition_density * constants.h*self.NU * self.dy[i]
                         *shifted_phi_v / (4*np.pi))
            self.cube += intensity*np.exp(-self.tau_nu)
            #column densities in particles/m2:
            N1 = n_x_yi_z * lower_level_fraction * self.dy[i]
            N2 = n_x_yi_z * upper_level_fraction * self.dy[i]
            Delta_NU = self.NU - NU_R
            tau_nu_i = self.transition.tau_nu(N1=N1,N2=N2,
                                              nu=self.transition.nu0+Delta_NU)
            assert np.all(tau_nu_i >= -self.optical_depth_epsilon),\
                'something strage is going on with the optical depth calculation:'+\
                ' min tau_nu: {:g}'.format(np.min(tau_nu_i))
            self.tau_nu += tau_nu_i
            if self.check_max_optical_depth_increase:
                tau_nu_change = self.tau_nu-previous_tau_nu
                assert np.all(tau_nu_change < self.max_optical_depth_increase),\
                    'resolution of y-axis not sufficient; optical depth increasing too rapidly'
                previous_tau_nu = self.tau_nu.copy()
            if self.verbose:
                print('current maximum optical depth: {:g}'.format(np.max(self.tau_nu)))
        if self.zsym:
            self.apply_zsym()

    def apply_zsym(self):
        self.z = np.concatenate((-self.z[:0:-1],self.z),axis=1)
        self.dz = np.concatenate((self.dz[:0:-1],self.dz),axis=1)
        self.cube = np.concatenate((self.cube[:,:0:-1,:],self.cube),axis=1)
        self.tau_nu = np.concatenate((self.tau_nu[:,:0:-1,:],self.tau_nu),axis=1)

    def compute_spec(self):
        self.spec = (self.DX*self.DZ*self.cube).sum(axis=0).sum(axis=0)#W/(m/s)/sr

    def plot_X_Z_data(self,data,colorbar_label='',title=None):
        XX,ZZ = np.meshgrid(self.x,self.z,indexing='ij')
        fig,ax = plt.subplots()
        ax.set_title(title)
        im = ax.pcolormesh(XX/constants.au,ZZ/constants.au,data,shading='auto')
        ax.set_xlabel('x [au]')
        ax.set_ylabel('z [au]')
        cbar = fig.colorbar(im,ax=ax,label=colorbar_label)
        cbar.ax.set_ylabel(ylabel=colorbar_label)
        ax.set_aspect('equal')
        
    def plot_mom0(self,title=None):
        mom0 = np.trapz(self.cube,self.v,axis=2) #W/m2/sr
        self.plot_X_Z_data(data=mom0,colorbar_label='W/m2/sr',title=title)

    def plot_max_tau_nu(self,title=None):
        tau_nu_max = np.max(self.tau_nu,axis=-1)
        self.plot_X_Z_data(data=tau_nu_max,colorbar_label='max optical depth',
                           title=title)

    def plot_pv(self,title=None):
        P,V = np.meshgrid(self.x,self.v,indexing='ij')
        pv = np.trapz(self.cube,self.z,axis=1) #W/m/(m/s)/sr
        plt.figure()
        plt.title(title)
        plt.pcolormesh(P/constants.au,V/constants.kilo,pv,shading='auto')
        plt.xlabel('x [au]')
        plt.ylabel('v [km/s]')
        plt.colorbar(label='W/m/(m/s)/sr')

    def plot_spectrum(self,title=None):
        plt.figure()
        plt.title(title)
        plt.plot(self.v/constants.kilo,self.spec,'.-')
        plt.xlabel('v [km/s]')
        plt.ylabel('flux [W/(m/s)/sr]')

    def total_flux(self,distance):
        #transform from /sr to /m2: multiply by sr=surface/distance**2 divide by m2=surface
        return np.trapz(self.spec,self.v)/distance**2

    def save(self,filepath):
        np.savez(file=filepath,x=self.x,z=self.z,v=self.v,cube=self.cube,
                 tau_nu=self.tau_nu)


if __name__ == '__main__':
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
    def Tex(x,y,z):
        return np.ones_like(x*y*z)*50
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
    data_filepath = '/home/gianni/Desktop/Applications/LAMDA_database_files/c+.dat'
    mole = molecule.EmittingMolecule.from_LAMDA_datafile(
                     data_filepath=data_filepath,line_profile_cls=line_profile_cls,
                     width_v=width_v)
    transition_name = '1-0'
    zsym = False
    inclination = np.pi/2
    verbose = True
    raytrace = Raytracing(grid=grid,Tex=Tex,number_density=number_density,
                          velocity_field=velocity_field,atom=mole,
                          transition_name=transition_name,zsym=zsym,
                          inclination=inclination,verbose=verbose)
    raytrace.raytrace()
    raytrace.compute_spec()
    raytrace.plot_mom0()
    raytrace.plot_pv()
    raytrace.plot_spectrum()
    raytrace.plot_max_tau_nu()
    total_flux = raytrace.total_flux(distance=10*constants.parsec)
    print('total flux: {:g} W/m2'.format(total_flux))