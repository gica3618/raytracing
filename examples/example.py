#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:31:44 2025

@author: gianni
"""

#In this example, we consider the raytracing of a Keplerian disk
#all inputs have to be in SI units


from pythonradex import molecule
import numpy as np
from scipy import constants
#add the raytracing module:
import sys
#path to the folder that contains the raytracing.py file:
sys.path.append('/home/gianni/science/projects/code/raytracing')
import raytracing
import matplotlib.pyplot as plt

#define the arrays used as sky grid
#x_sky is in the horizontal direction, y_sky along the line of sight, and z_sky
#in the vertical direction
x_sky = np.linspace(-100,100,50)*constants.au
y_sky = x_sky.copy()
# we adjust z according to the inclination of the disk:
z_sky = np.linspace(-50,50,60)*constants.au
#velocity grid:
v = np.linspace(-10,10,60)*constants.kilo
grid = {'x_sky':x_sky,'y_sky':y_sky,'z_sky':z_sky,'v':v}

#define the velocity field. We consider a Keplerian velocity field of a disk
#that is placed in the x-y plane. The function needs to return the velocity
#in the y and z direction.
Mstar = 2e30 #stellar mass in kg
def velocity_yz(x,y,z):
    output_shape = np.broadcast(x,y,z).shape
    r = np.sqrt(x**2+y**2)
    v0 = np.sqrt(constants.G*Mstar/r)
    phi = np.arctan2(y,x)
    return [v0*np.cos(phi),np.zeros(output_shape)]

#consider a temperature profile scaling as 1/sqrt(r)
#this temperature defines the level population assuming a Boltzmann distribution
#for example, when assuming LTE, this temperature is equal to the kinetic
#temperature of the gas
T0 = 50
r0 = 50*constants.au
def T_Boltzmann(x,y,z):
    r = np.sqrt(x**2+y**2)
    return T0*(r/r0)**-0.5

#for the number density, consider a Gaussian ring centered at r0. The vertical
#profile is also Gaussian
def number_density(x,y,z):
    r = np.sqrt(x**2+y**2)
    H = 10*constants.au
    n0 = 500/constants.centi**3
    sigma = 10*constants.au
    n = n0*np.exp(-(r-r0)**2/(2*sigma**2))*np.exp(-z**2/(2*H**2))
    return n
    
#define the species that is emitting
#FWHM of the intrinsic Gaussian emission profile:
line_profile_type = 'Gaussian' #'rectangular' is also available, see pythonradex documentation
width_v = 1.5*constants.kilo
#filepath to the LAMDA file that describes the atomic properties of the species
#here we consider CO
data_filepath = 'co.dat'
atom = molecule.EmittingMolecule(
                 datafilepath=data_filepath,line_profile_type=line_profile_type,
                 width_v=width_v)
#define which transition we want to consider. Here we want the third transition
#listed in the LAMDA file (CO 3-2), so the index is 2
transition_index = 2
#inclination of the x,y,z coordinates system with respect to the sky coordinate system
#we set the inclination to 30 degrees. A zero inclination would correspond to an
#edge-on disk (i.e. disk inclination 90 deg), because our disk is defined to lay
#in the x-y plane
inclination_xyz = np.radians(30)
#extra parameters:
#activate extra information to be printed to the terminal
verbose = True
#allow large optical depth increase when raytracing along the line of sight:
check_max_optical_depth_increase = False 
# do not allow negative optical depth:
allow_negative_tau = False
#define the model:
model = raytracing.Raytracing(
            grid=grid,T_Boltzmann=T_Boltzmann,number_density=number_density,
            velocity_yz=velocity_yz,atom=atom,transition_index=transition_index,
            inclination_xyz=inclination_xyz,verbose=verbose)
#execute the raytracing:
model.raytrace()

#now the model has additional attributes that contain the results:
#model.cube is the emission cube in units of W/m2/(m/s)/sr
print(f'peak emission: {np.max(model.cube):.3g} W/m2/(m/s)/sr')
#plot the spectrum from a specific pixel:
ix,iz = 10,20
spec = model.cube[ix,iz,:]
fig,ax = plt.subplots()
ax.set_title(f'spectrum at x={x_sky[ix]/constants.au:.3g}'
             +f' au, z={z_sky[iz]/constants.au:.3g} au')
ax.plot(v/constants.kilo,spec)
ax.set_xlabel('v [km/s]')
ax.set_ylabel('intensity [W/m2/(m/s)/Hz]')

#model.tau_nu is the optical depth
#let's look at the optical depth at the same pixel:
tau = model.tau_nu[ix,iz,:]
fig,ax = plt.subplots()
ax.set_title(f'optical depth at x={x_sky[ix]/constants.au:.3g}'
             +f' au, z={z_sky[iz]/constants.au:.3g} au')
ax.plot(v/constants.kilo,tau)
ax.set_xlabel('v [km/s]')
ax.set_ylabel('optical depth')

#model.spectrum is the total spectrum
fig,ax = plt.subplots()
ax.set_title('spectrum')
ax.plot(v/constants.kilo,model.spectrum)
ax.set_xlabel('v [km/s]')
ax.set_ylabel('flux [W/(m/s)/sr]')

#Plots:
#plot the moment 0:
model.plot_mom0(title='moment 0')
#plot the integration of the cube along the z_sky axis
model.plot_pv()
#plot the disk-integrated spectrum (this is the same as we already plotted
#above using model.spectrum)
model.plot_spectrum(title='spectrum')
#plot a map of the peak optical depth
model.plot_max_tau_nu(title='peak optical depth')
#calculate the disk-integrated flux at a distance of 10 parsec
total_flux = model.total_flux(distance=10*constants.parsec)
print('total flux: {:g} W/m2'.format(total_flux))
model.save(filepath='my_model.npz')