#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:28:47 2019

@author: gianni
"""

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


class Raytracing():
    """A simple class for raytracing.

    Arguments:
        grid (dict): A dictionary that contains four 1D numpy arrays. These
            arrays define the sky grid over which the model is calculated. The
            dictionary keys are:
                - x_sky: the grid in the horizontal direction on the sky.
                - y_sky: the grid along the line of sight
                - z_sky: the grid in the vertical direction on the sky.
                - v: the velocity grid.
            The units of x_sky, y_sky and z_sky are [m]. The units of v are
            [m/s].
        NOTE: Below, several functions of x, y and z are described. Here, x,
            y and z describe a coordinate system that can be inclined with
            respect to the sky coordinate system. The inclination is achieved
            by rotating around the x_sky axis. The inclination angle
            is defined as the angle between the y and y_sky. For inclination=0,
            x=x_sky, y=y_sky, z=z_sky. For inclination=90 deg, x=x_sky,
            y=z_sky, z=-y_sky
        number_density (func): A function of x, y and z. It should return
            the number density in units of [1/m3].
        velocity_yz (func): A function of x, y and z. It should return
            the instrinsic FWHM of the emission line in units of [m/s].
        FWHM_v (func):  A function of x, y and z. It should return
            a list with two elements. The elements of the list are the
            velocity (in units of [m/s]) in the y and z direction.
        mole (pythonradex.molecule.Molecule): Object containing
            atomic/molecular data. The object is initialised using a datafile
            from the LAMDA database.
        transition_index (int): The index of the transition to be considered,
            based on the list of transitions in the LAMDA file. The first
            transition has index 0.
        inclination_xyz (float): Inclination angle in un
                 its of [rad]. It is the
            angle between y and y_sky. Defaults to 0, i.e. x=x_sky, y=y_sky,
            z=z_sky.
        T_Boltzmann (func):  A function of x, y and z. It should return
            the temperature that is used to define the level populations,
            assuming a Boltzmann distribution. Defaults to None. The user
            needs either to define T_Boltzmann, or x1 and x2 (see below).
        x1 (func): A function of x, y and z. It should return the fractional
            population of the lower level of the transition. Defaults to None.
            The user needs either to define T_Boltzmann, or x1 and x2.
        x2 (func): A function of x, y and z. It should return the fractional
            population of the upper level of the transition. Defaults to None.
            The user needs either to define T_Boltzmann, or x1 and x2.
        verbose (bool): Whether or not to print additional information.
            Defaults to False
        check_max_optical_depth_increase (bool): whether or not to check
            the increase of the optical depth when taking one grid step
            along the line of sight. If True and the optical depth increases
            by more than 0.2, the caculation is aborted. Defaults to False.
        allow_negative_tau (bool): If False and negative optical depth occurs,
            the calculation is aborted. Defaults to False.

    Methods:
        generate_Keplerian_velocity_yz(Mstar):
            helper function to generate velocity_yz function of a Keplerian disk
        raytrace():
            raytrace the model
        plot_mom0(title=None):
            Plot the moment 0 map of the model.
        plot_max_tau_nu(title=None):
            Plot a map of the peak optical depth.
        plot_pv(title=None):
            Integrate the cube along z_sky and plot the resulting map.
        plot_spectrum(title=None):
            Plot the total spectrum of the model.
        total_flux(distance):
            Calculate the total flux of the model in units of [W/m2]. The argument
            distance defines the distance to the target in units of [m].
        save(filepath):
            Save x_sky, z_sky, v, cube and tau_nu in npz format into a file defined
            by the filepath argument.

    Attributes available after raytracing:
        cube (numpy array): A 3D array giving the result of the raytracing, i.e.
            the emission in units of [W/m2/(m/s)/sr] over the grid x_sky, z_sky, v.
        tau_nu (numpy array): A 3D array giving the optical depth over the grid
            x_sky, z_sky, v
        spectrum (numpy array): A 1D array giving the total spectrum in units of
            [W/(m/s)/sr] over the velocity axis v.
    """

    # convention: line of sight is along y_sky-axis, i.e. the observers looks
    # along the y_sky-axis in the positive direction
    # convention for array dimensions: first axis: x_sky; second axis: z_sky;
    # third axis: v_sky
    # Previously, the calculation was aborted when the increase in optical depth
    # was too large in a single iteration step. However, I found that this does not
    # really change the results by comparing to LIME, so I made it optional

    max_optical_depth_increase = 0.2

    def __init__(self,grid,number_density,velocity_yz,FWHM_v,mole,transition_index,
                 inclination_xyz=0,T_Boltzmann=None,x1=None,x2=None,
                 verbose=False,check_max_optical_depth_increase=False,
                 allow_negative_tau=False):
        self.grid = grid
        self.T_Boltzmann = T_Boltzmann
        self.x1 = x1
        self.x2 = x2
        if self.T_Boltzmann is None:
            assert self.x1 is not None and self.x2 is not None
        else:
            assert self.x1 is None and self.x2 is None
        self.number_density = number_density
        self.velocity_yz = velocity_yz
        self.FWHM_v = FWHM_v
        self.mole = mole
        self.transition = self.mole.rad_transitions[transition_index]
        self.inclination_xyz = inclination_xyz
        self.verbose = verbose
        self.check_max_optical_depth_increase = check_max_optical_depth_increase
        self.allow_negative_tau = allow_negative_tau
        self.grid_setup()

    @staticmethod
    def compute_spatial_interval(a):
        '''calculation of the spatial interval associated with every point of array a'''
        assert np.all(np.diff(a)>0),'input array not strictly monotonically increasing'
        intervals = (a[1]-a[0], (a[2:]-a[1:-1])/2+(a[1:-1]-a[:-2])/2, a[-1]-a[-2])
        return np.hstack(intervals)
    
    @staticmethod
    def B_nu(T,nu):
        return 2*constants.h*nu**3/constants.c**2\
                     * (np.exp(constants.h*nu/(constants.k*T))-1)**-1
    
    @staticmethod
    def generate_Keplerian_velocity_yz(Mstar):
        '''Generate the velocity_yz function for a Keplerian disk lying in the
        x-y plane. Mstar is the mass of the central star in units of [kg]. The
        velocity is positive for x>0, negative for x<0'''
        def v(x,y,z):
            r = np.sqrt(x**2+y**2)
            v0 = np.sqrt(constants.G*Mstar/r)
            phi = np.arctan2(y,x)
            return [v0*np.cos(phi),np.zeros_like(r)]
        return v

    def compute_nu(self,v):
        return self.transition.nu0*(1-v/constants.c)

    def phi_nu(self,FWHM_nu,nu):
        #width interpreted as FWHM
        sigma = FWHM_nu/np.sqrt(8*np.log(2))
        norm = 1/(np.sqrt(2*np.pi)*sigma)
        return norm*np.exp(-(nu-self.transition.nu0)**2/(2*sigma**2))

    def compute_tau_nu(self,N1,N2,nu,FWHM_nu):
        A21 = self.transition.A21
        g_up = self.transition.up.g
        g_low = self.transition.low.g
        phi_nu = self.phi_nu(FWHM_nu=FWHM_nu,nu=nu)
        return constants.c**2/(8*np.pi*nu**2)*A21*phi_nu*(g_up/g_low*N1-N2)

    def grid_setup(self):
        self.v = self.grid['v']
        self.V = self.v[np.newaxis,np.newaxis,:]
        self.nu = self.compute_nu(self.v)
        self.NU = self.compute_nu(self.V)
        self.x_sky = self.grid['x_sky']
        self.X_sky = self.x_sky[:,np.newaxis,np.newaxis]
        self.dx_sky = self.compute_spatial_interval(self.x_sky)
        self.dX_sky = self.dx_sky[:,np.newaxis,np.newaxis]
        self.y_sky = self.grid['y_sky']
        assert np.all(np.diff(self.y_sky)>0),\
                          'y axis not appropriate for optical depth calculation'
        self.dy_sky = self.compute_spatial_interval(self.y_sky)
        self.z_sky = self.grid['z_sky']
        self.Z = self.z_sky[np.newaxis,:,np.newaxis]
        self.dz_sky = self.compute_spatial_interval(self.z_sky)
        self.dZ_sky = self.dz_sky[np.newaxis,:,np.newaxis]

    def raytrace(self):
        '''Raytrace the model.'''
        self.cube = np.zeros((self.x_sky.size,self.z_sky.size,self.v.size)) #W/m2/(m/s)/sr
        self.tau_nu = np.zeros_like(self.cube)
        if self.check_max_optical_depth_increase:
            #following array is used to alert if the resolution in y-direction is not
            #sufficient, i.e. tau is increasing too fastly
            previous_tau_nu = np.zeros_like(self.cube)
        # NOTE: it is very important that the iteration goes along the line of
        #sight, away from the telescope! (to sum up optical depth)
        for i,y_i in enumerate(self.y_sky):
            if self.verbose:
                print(f"Step {i+1} of {self.y_sky.size}: y={y_i/constants.au:.2g} au")
            #to take into account inclination, for each point in the sky grid,
            #compute the coordinates in the x,y,z coordinate system and evaluate
            #functions using those coordinates
            #a point with coordinates (x_sky,y_sky,z_sky) in the sky coordinate system
            #has coordinates (x_sky,cosi*y_sky+sini*z_sky,-sini*y_sky+cosi*z_sky)
            #in the x,y,z coordinate system
            X = self.X_sky
            Y = y_i*np.cos(self.inclination_xyz)\
                                 +self.Z*np.sin(self.inclination_xyz)
            Z = -y_i*np.sin(self.inclination_xyz)\
                        +self.Z*np.cos(self.inclination_xyz)
            n_x_yi_z = self.number_density(x=X,y=Y,z=Z)
            if self.T_Boltzmann is not None:
                T_x_yi_z = self.T_Boltzmann(x=X,y=Y,z=Z)
                partition_func = self.mole.Z(T_x_yi_z)
                upper_level_fraction = self.transition.up.LTE_level_pop(
                                                    Z=partition_func,T=T_x_yi_z)
                lower_level_fraction = self.transition.low.LTE_level_pop(
                                                   Z=partition_func, T=T_x_yi_z)
                Tex = T_x_yi_z
            else:
                upper_level_fraction = self.x2(x=X,y=Y,z=Z)
                lower_level_fraction = self.x1(x=X,y=Y,z=Z)
                Tex = self.transition.Tex(x1=lower_level_fraction,
                                          x2=upper_level_fraction)
            #for velocity field, calculate it using x,y,z coordinate system,
            #then project onto the line of sight (i.e. y_sky)
            #the unit vector in direction y_sky is (0,cosi,-sini)
            v_yz = self.velocity_yz(x=X,y=Y,z=Z)
            assert len(v_yz) == 2
            V_los = v_yz[0]*np.cos(self.inclination_xyz)\
                                    -v_yz[1]*np.sin(self.inclination_xyz)
            NU_los = self.compute_nu(V_los)
            #column densities in particles/m2:
            N1 = n_x_yi_z * lower_level_fraction * self.dy_sky[i]
            N2 = n_x_yi_z * upper_level_fraction * self.dy_sky[i]
            Delta_NU = self.NU - NU_los
            FWHM_nu = self.FWHM_v(x=X,y=Y,z=Z)*self.transition.nu0/constants.c
            tau_nu_i = self.compute_tau_nu(N1=N1,N2=N2,nu=self.transition.nu0+Delta_NU,
                                           FWHM_nu=FWHM_nu)
            # tau_nu_i = self.transition.tau_nu(
            #                    N1=N1,N2=N2,nu=self.transition.nu0+Delta_NU)
            if not self.allow_negative_tau:
                assert np.all(tau_nu_i >= 0),\
                    'something strage is going on with the optical depth calculation:'+\
                    ' min tau_nu: {:g}'.format(np.min(tau_nu_i))
            #intensity in W/m**2/(m/s)/sr at position x,z:
            intensity = self.B_nu(nu=self.transition.nu0,T=Tex)\
                                            *(1-np.exp(-tau_nu_i)) #W/m2/Hz/sr
            #if emission from a single cell is optically thin, we can calculate as follows:
            # dV = self.dX_sky*self.dZ_sky*self.dy_sky[i]
            # intensity = n_x_yi_z*dV*upper_level_fraction*self.transition.Delta_E*self.transition.A21\
            #             *self.transition.line_profile.phi_nu(nu=self.transition.nu0+Delta_NU)\
            #             /(4*np.pi)/(self.dX_sky*self.dZ_sky)
            intensity *= self.transition.nu0/constants.c #W/m**2/(m/s)/sr
            self.cube += intensity*np.exp(-self.tau_nu)
            self.tau_nu += tau_nu_i
            if self.check_max_optical_depth_increase:
                tau_nu_change = self.tau_nu-previous_tau_nu
                max_tau_nu_change = np.max(tau_nu_change)
                assert np.all(tau_nu_change < self.max_optical_depth_increase),\
                    f'resolution of y-axis not sufficient; optical depth increasing too rapidly (max tau nu change: {max_tau_nu_change})'
                previous_tau_nu = self.tau_nu.copy()
            if self.verbose:
                print('current maximum optical depth: {:g}'.format(np.max(self.tau_nu)))
        self.spectrum = np.sum(self.dX_sky*self.dZ_sky*self.cube,axis=(0,1))#W/(m/s)/sr

    def plot_X_Z_data(self,data,colorbar_label='',title=None):
        XX,ZZ = np.meshgrid(self.x_sky,self.z_sky,indexing='ij')
        fig,ax = plt.subplots()
        ax.set_title(title)
        im = ax.pcolormesh(XX/constants.au,ZZ/constants.au,data,shading='auto')
        ax.set_xlabel('x [au]')
        ax.set_ylabel('z [au]')
        cbar = fig.colorbar(im,ax=ax,label=colorbar_label)
        cbar.ax.set_ylabel(ylabel=colorbar_label)
        ax.set_aspect('equal')

    def plot_mom0(self,title=None):
        '''Plot the moment 0 map of the model.'''
        mom0 = np.trapezoid(self.cube,self.v,axis=2) #W/m2/sr
        self.plot_X_Z_data(data=mom0,colorbar_label='W/m2/sr',title=title)

    def plot_max_tau_nu(self,title=None):
        '''Plot a map of the peak optical depth.'''
        tau_nu_max = np.max(self.tau_nu,axis=-1)
        self.plot_X_Z_data(data=tau_nu_max,colorbar_label='max optical depth',
                           title=title)

    def plot_pv(self,title=None):
        '''Integrate the cube along z_sky and plot the resulting map.'''
        P,V = np.meshgrid(self.x_sky,self.v,indexing='ij')
        pv = np.trapezoid(self.cube,self.z_sky,axis=1) #W/m/(m/s)/sr
        plt.figure()
        plt.title(title)
        plt.pcolormesh(P/constants.au,V/constants.kilo,pv,shading='auto')
        plt.xlabel('x [au]')
        plt.ylabel('v [km/s]')
        plt.colorbar(label='W/m/(m/s)/sr')

    def plot_spectrum(self,title=None):
        '''Plot the total spectrum of the model.'''
        plt.figure()
        plt.title(title)
        plt.plot(self.v/constants.kilo,self.spectrum,'.-')
        plt.xlabel('v [km/s]')
        plt.ylabel('flux [W/(m/s)/sr]')

    def total_flux(self,distance):
        '''Calculate the total flux of the model in units of [W/m2]'''
        #transform from /sr to /m2: multiply by sr=surface/distance**2 divide by m2=surface
        return np.trapezoid(self.spectrum,self.v)/distance**2 #W/m2

    def save(self,filepath):
        '''Save x_sky, z_sky, v, cube and tau_nu in npz format into a file defined
        by the filepath argument.'''
        np.savez(file=filepath,x_sky=self.x_sky,z=self.z_sky,v=self.v,cube=self.cube,
                 tau_nu=self.tau_nu)