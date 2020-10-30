# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:40:11 2018

@author: hector
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as pth

from pyPro4Sail import ProspectD, FourSAIL

from SALib.sample import saltelli
from SALib.analyze import sobol

from scipy.ndimage.filters import gaussian_filter1d

# Number of simulations
N = 5000

# Half-Width Full Maximum
fwhm = 10

calc_second_order = False
# Define the model inputs
cpus = 7

parallel = True

# Define the wavelenght range to calculate the SA
wls_SA=np.arange(400,1000,10)

soil_folder = pth.join(pth.expanduser('~'),
                       'Data',
                       'Scripts', 
                       'Python', 
                       'PROSPECT4SAIL', 
                       'pyPro4Sail', 
                       'SoilSpectralLibrary')
                       
problem_leaf = {'num_vars': 7,
            'names': ['N_leaf',
                      'Cab',
                      'Car',
                      'Cbrown',
                      'Cw',
                      'Cm',
                      'Ant'],
            'bounds': [(1.0,3.0),
                      (10.0,100.0),
                      (0.0,40.0),
                      (0.0,1.0),
                      (0.000063,0.040000),
                      (0.001900,0.016500),
                      (0.00,40.)]}


problem_canopy = {'num_vars': 9,
            'names': ['N_leaf',
                      'Cab',
                      'Car',
                      'Cbrown',
                      'Cw',
                      'Cm',
                      'Ant',
                      'LAI',
                      'leaf_angle'],
            'bounds': [(1.0,3.0),
                      (0.0,100.0),
                      (0.0,40.0),
                      (0.0,1.0),
                      (0.000063,0.040000),
                      (0.001900,0.016500),
                      (0.00,40.),
                      (0.10,6.),
                      (30.0,80.0)]}

print('Generating %s simulations for ProspectD'%(N*(problem_leaf['num_vars']+2)))
samples = saltelli.sample(problem_leaf, N, calc_second_order=calc_second_order)

print('Running ProspectD')
wls, rho, tau = ProspectD.ProspectD_vec(samples[:,0],
                                     samples[:,1],
                                     samples[:,2],
                                     samples[:,3],
                                     samples[:,4],
                                     samples[:,5],
                                     samples[:,6])

N_samples_canopy = N*(problem_canopy['num_vars']+2)
print('Generating %s simulations for SAIL'%N_samples_canopy)
samples = saltelli.sample(problem_canopy, N, calc_second_order=calc_second_order)

print('Running ProspectD+4SAIL')
l, r, t = ProspectD.ProspectD_vec(samples[:,0],
                                     samples[:,1],
                                     samples[:,2],
                                     samples[:,3],
                                     samples[:,4],
                                     samples[:,5],
                                     samples[:,6])


lidf=FourSAIL.CalcLIDF_Campbell_vec(samples[:,8])

# Read the soil reflectance        
rsoil=np.genfromtxt(pth.join(
                    soil_folder,
                    'ipgp.jussieu.soil.prosail.dry.coarse.1.spectrum.txt'))

#wl_soil=rsoil[:,0]
rsoil=np.array(rsoil[:,1])
rsoil=np.repeat(rsoil[:,np.newaxis], N_samples_canopy, axis=1)

[_,_,_,_,_,_,_,_,_,_,_,_,_,_,rdot,_,_,rsot,_,_,_]=FourSAIL.FourSAIL_vec(samples[:,7],
                                                                       np.ones(N_samples_canopy)*0.01, 
                                                                       lidf,
                                                                       np.ones(N_samples_canopy)*37.,
                                                                       np.zeros(N_samples_canopy), 
                                                                       np.zeros(N_samples_canopy),
                                                                       r.T,
                                                                       t.T,
                                                                       rsoil)


rho_canopy = rdot*0.2 + rsot * (1.0 - 0.2)
if fwhm:
    #Convolve spectra by full width half maximum
    sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
    rho_canopy=gaussian_filter1d(rho_canopy,sigma,axis=0)


rho_canopy=rho_canopy.T

del r, t, lidf, _, rdot, rsot, rsoil


print('Running Sensitivity Analysis')

S1_rho = []
S1_tau = []
S1_abs = []
S1_rho_canopy = []

ST_rho = []
ST_tau = []
ST_abs = []
ST_rho_canopy = []

wl_index = [int(np.where(wl == wls)[0]) for wl in wls_SA]

for i,wl in zip(wl_index,wls_SA):
    print('Getting SA indices for wavelength %s nm'%wl)
    SA_rho = sobol.analyze(problem_leaf, 
                         rho[:, i], 
                         calc_second_order=calc_second_order, 
                         num_resamples=100, 
                         conf_level=0.95, 
                         print_to_console=False, 
                         parallel=parallel, 
                         n_processors=cpus)
    
    S1_rho.append(SA_rho['S1'])
    ST_rho.append(SA_rho['ST']) 
              
    SA_tau = sobol.analyze(problem_leaf, 
                         tau[:, i], 
                         calc_second_order=calc_second_order, 
                         num_resamples=100, 
                         conf_level=0.95, 
                         print_to_console=False, 
                         parallel=parallel, 
                         n_processors=cpus)
 
    S1_tau.append(SA_tau['S1'])
    ST_tau.append(SA_tau['ST']) 

   
    SA_abs = sobol.analyze(problem_leaf, 
                         1. - (rho[:, i] + tau[:, i]), 
                         calc_second_order=calc_second_order, 
                         num_resamples=100, 
                         conf_level=0.95, 
                         print_to_console=False, 
                         parallel=parallel, 
                         n_processors=cpus)

    S1_abs.append(SA_abs['S1'])
    ST_abs.append(SA_abs['ST']) 

    SA_canopy = sobol.analyze(problem_canopy, 
                         rho_canopy[:, i], 
                         calc_second_order=calc_second_order, 
                         num_resamples=100, 
                         conf_level=0.95, 
                         print_to_console=False, 
                         parallel=parallel, 
                         n_processors=cpus)

    S1_rho_canopy.append(SA_canopy['S1'])
    ST_rho_canopy.append(SA_canopy['ST']) 

S1_rho, ST_rho, S1_tau, ST_tau, S1_abs, ST_abs, S1_rho_canopy, ST_rho_canopy = map(np.asarray,
                                                                 [S1_rho, 
                                                                  ST_rho, 
                                                                  S1_tau, 
                                                                  ST_tau, 
                                                                  S1_abs, 
                                                                  ST_abs,
                                                                  S1_rho_canopy,
                                                                  ST_rho_canopy])

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

rho_cum = np.zeros(wls_SA.shape)
tau_cum = np.zeros(wls_SA.shape)
abs_cum = np.zeros(wls_SA.shape)
rho_canopy_cum = np.zeros(wls_SA.shape)

rho_sum = np.sum(S1_rho, axis = 1)
tau_sum = np.sum(S1_tau, axis = 1)
abs_sum = np.sum(S1_abs, axis = 1)
rho_canopy_sum = np.sum(S1_rho_canopy, axis = 1)

colors=['black', 'green', 'orange', 'brown', 'blue', 'yellow', 'red', 'purple', 'cyan', 'white']



for i, param in enumerate(problem_leaf['names']):
    Y_value = rho_cum + S1_rho[:, i] / rho_sum
    ax1.fill_between(wls_SA, rho_cum, Y_value, color=colors[i], label = param)
    rho_cum = np.copy(Y_value)

    Y_value = tau_cum + S1_tau[:, i] / tau_sum
    ax2.fill_between(wls_SA, tau_cum, Y_value, color=colors[i], label = param)
    tau_cum = np.copy(Y_value)

    Y_value = abs_cum + S1_abs[:, i] / abs_sum
    ax3.fill_between(wls_SA, abs_cum, Y_value, color=colors[i], label = param)
    abs_cum = np.copy(Y_value)

wl_max = dict()
for i, param in enumerate(problem_canopy['names']):

    Y_value = rho_canopy_cum + S1_rho_canopy[:, i] / rho_canopy_sum
    ax4.fill_between(wls_SA, rho_canopy_cum, Y_value, color=colors[i], label = param)
    rho_canopy_cum = np.copy(Y_value)
    wl_max[param] = float(wls_SA[np.where(S1_rho_canopy[:, i]  / rho_canopy_sum 
                            == np.max(S1_rho_canopy[:, i] / rho_canopy_sum))][0])
    ax4.plot([wl_max[param],wl_max[param]],[0,1],color=colors[i])

ax1.set_title('SA leaf reflectance')
ax2.set_title('SA leaf transmittance')
ax3.set_title('SA leaf absorptance')
ax4.set_title('SA canopy reflectance')

ax1.set_xlabel('Wavelength (nm)')
ax2.set_xlabel('Wavelength (nm)')
ax3.set_xlabel('Wavelength (nm)')
ax4.set_xlabel('Wavelength (nm)')

ax1.set_ylabel('Relative S1')
ax2.set_ylabel('Relative S1')
ax3.set_ylabel('Relative S1')  
ax4.set_ylabel('Relative S1') 

ax1.legend(ncol=4, mode='expand') 
ax2.legend(ncol=4, mode='expand') 
ax3.legend(ncol=4, mode='expand') 
ax4.legend(ncol=4, mode='expand') 