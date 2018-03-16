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
from pyPro4Sail.CostFunctionsPROSPECT4SAIL import FCostJac_ProSail as fcost_jac
from pyPro4Sail.CostFunctionsPROSPECT4SAIL import FCost_ProSail as fcost
from pyPro4Sail.invertANN_ProSAIL import build_prosail_database as generate

from SALib.sample import saltelli

import scipy.optimize as op
from scipy.ndimage.filters import gaussian_filter1d
import scipy.stats as st

# Number of simulations
N_samples_canopy =  600
N = 100

ObjParams = ('Cab','Cbrown', 'Ant', 'LAI', 'leaf_angle')

average_param = {'N_leaf' : 1.5,
          'Cab' : 40,
          'Car' : 20,
          'Cbrown' : 0.0,
          'Cw' : 0.005,
          'Cm' : 0.005,
          'Ant' :  0.0,
          'LAI' : 0.5,
          'hotspot': 0.01, 
          'leaf_angle' : 53.7}

param_bounds={'N_leaf':(1.0,3.0),
             'Cab':(0.0,100.0),
             'Car':(0.0,40.0),
             'Cbrown':(0.0,1.0),
             'Cw':(0.000063,0.040000),
             'Cm':(0.001900,0.016500),
             'Ant':(0.00,40.),
             'LAI':(0.0,6.0),
             'hotspot':(0.01, 0.1),
             'leaf_angle':(30.0,80.0)}

moments={'N_leaf':(1.5,0.3),
         'Cab':(45.0,30.0),
         'Car':(20.0,10.0),
         'Cbrown':(0.0,0.3),
         'Cw':(0.005,0.005),
         'Cm':(0.005,0.005),
         'Ant':(0.0,10.0),
         'LAI':(1.5,3.0),
         'hotspot':(0.2,0.5),
         'leaf_angle':(60.0,30.0)} 

UNIFORM_DIST = 1    
GAUSSIAN_DIST = 2         
distribution = {'N_leaf':UNIFORM_DIST,
             'Cab':UNIFORM_DIST,
             'Car':UNIFORM_DIST,
             'Cbrown':UNIFORM_DIST,
             'Cw':UNIFORM_DIST,
             'Cm':UNIFORM_DIST,
             'Ant':UNIFORM_DIST,
             'LAI':UNIFORM_DIST,
             'hotspot':UNIFORM_DIST,
             'leaf_angle':UNIFORM_DIST}         

apply_covariate = {'N_leaf':False,
             'Cab':False,
             'Car':False,
             'Cbrown':False,
             'Cw':False,
             'Cm':False,
             'Ant':False,
             'leaf_angle':False,
             'hotspot':False}
             
covariate = {'N_leaf':((1.0,3),(1.3,1.8)),
             'Cab':((0,100),(45,100)),
             'Car':((0,40),(20,40)),
             'Cbrown':((0,1),(0,0.2)),
             'Cw':((0.000063,0.040000),(0.005,0.011)),
             'Cm':((0.001900,0.016500),(0.005,0.011)),
             'Ant':((0,40),(0,40)),
             'leaf_angle':((30,80),(55,65)),
             'hotspot':((0.01,0.1),(0.01,0.1))}

problem_canopy = {'num_vars': 10,
            'names': ['N_leaf',
                      'Cab',
                      'Car',
                      'Cbrown',
                      'Cw',
                      'Cm',
                      'Ant',
                      'LAI',
                      'hotspot',
                      'leaf_angle'],
            'bounds': [(1.0,3.0),
                      (0.0,100.0),
                      (0.0,40.0),
                      (0.0,1.0),
                      (0.000063,0.040000),
                      (0.001900,0.016500),
                      (0.00,40.),
                      (0.0,6.),
                      (0.01, 0.1),
                      (30.0,80.0)]}


# Half-Width Full Maximum
fwhm = 10

band_selection = {'default': [490, 550, 680, 720, 800, 900],
                  'SA': [460, 560, 680, 710, 770, 870],
                  'Sentinel2': [560, 664, 704, 740, 782, 864]}

calc_second_order = False


soil_folder = pth.join(pth.expanduser('~'),
                       'Data',
                       'Scripts', 
                       'Python', 
                       'PROSPECT4SAIL', 
                       'pyPro4Sail', 
                       'SoilSpectralLibrary')



#==============================================================================
# # Use distribution-based sampling method
# input_param = generate(N_samples_canopy,
#                        param_bounds = param_bounds,
#                        moments = moments,
#                        distribution = distribution,
#                        apply_covariate = apply_covariate,
#                        covariate = covariate)
#==============================================================================

# Use Saltelli sampling method
N_samples_canopy = N*(problem_canopy['num_vars']+2)
print('Generating %s simulations for SAIL'%N_samples_canopy)
samples = saltelli.sample(problem_canopy, N, calc_second_order=False)
input_param = {}
for i, param in enumerate(FourSAIL.paramsPro4SAIL):
    input_param[param]=samples[:,i]

samples_new = np.zeros((N_samples_canopy,len(ObjParams)))
j = 0
for i, param in enumerate(FourSAIL.paramsPro4SAIL):
    if param in ObjParams:
        samples_new[:,j] = input_param[param]
        j += 1

print('Running ProspectD+4SAIL')
l, r, t = ProspectD.ProspectD_vec(input_param['N_leaf'],
                                  input_param['Cab'],
                                  input_param['Car'],
                                    input_param['Cbrown'], 
                                    input_param['Cw'],
                                    input_param['Cm'],
                                    input_param['Ant'])


lidf=FourSAIL.CalcLIDF_Campbell_vec(input_param['leaf_angle'])

# Read the soil reflectance        
rsoil=np.genfromtxt(pth.join(
                    soil_folder,
                    'ipgp.jussieu.soil.prosail.dry.coarse.1.spectrum.txt'))

#wl_soil=rsoil[:,0]
rsoil=np.array(rsoil[:,1])
rsoil=np.repeat(rsoil[:,np.newaxis], N_samples_canopy, axis=1)

# Run nadir observations 
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,
                 rdot,_,_,rsot,_,_,_]=FourSAIL.FourSAIL_vec(input_param['LAI'],
                                                           input_param['hotspot'], 
                                                           lidf,
                                                           np.ones(N_samples_canopy)*37.,
                                                           np.zeros(N_samples_canopy), 
                                                           np.zeros(N_samples_canopy),
                                                           r.T,
                                                           t.T,
                                                           rsoil)


rho_canopy = rdot*0.2 + rsot * (1.0 - 0.2)
#Convolve spectra by full width half maximum
sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
rho_canopy = gaussian_filter1d(rho_canopy, sigma, axis=0)

rho_canopy_nadir = rho_canopy.T

# Run oblique observations
[_,_,_,_,_,_,_,_,_,_,_,_,_,_,
                 rdot,_,_,rsot,_,_,_]=FourSAIL.FourSAIL_vec(input_param['LAI'],
                                                           input_param['hotspot'], 
                                                           lidf,
                                                           np.ones(N_samples_canopy)*37.,
                                                           np.ones(N_samples_canopy)*40., 
                                                           np.zeros(N_samples_canopy),
                                                           r.T,
                                                           t.T,
                                                           rsoil)


rho_canopy = rdot*0.2 + rsot * (1.0 - 0.2)
#Convolve spectra by full width half maximum
sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
rho_canopy = gaussian_filter1d(rho_canopy, sigma, axis=0)

rho_canopy_oblique = rho_canopy.T

del r, t, lidf, _, rdot, rsot, rsoil

# Test inversion
n_obs = 1

scale=[]
guess=[]
FixedValues=[]

for param in FourSAIL.paramsPro4SAIL:
    if param in ObjParams:
        scale.append([param_bounds[param][0],
                      param_bounds[param][1]-param_bounds[param][0]])
        guess.append(average_param[param])
    else:
        FixedValues.append(average_param[param])


x0=[]
for param in ObjParams:
    x0.append((average_param[param] - float(param_bounds[param][0]))
                /float(param_bounds[param][1] - param_bounds[param][0]))

bounds_lbgfs=tuple(len(x0)*[[0,1]])


# Inversion for typical Tetracam wavelengths
for filters, wls_inv in band_selection.items():
    print('# Inversion for %s band selection'%filters)
    
    # Read the soil reflectance        
    rsoil=np.genfromtxt(pth.join(
                        soil_folder,
                        'ipgp.jussieu.soil.prosail.dry.coarse.1.spectrum.txt'))
    
    rsoil =np.array(rsoil[:,1])
    index_wls = []
    for wl in wls_inv:
        index_wls.append(int(np.where(wl == l)[0]))
        
        
    rsoil = rsoil[index_wls]
    dtype = [(param, 'f4') for param in ObjParams]
    estimated = np.zeros((N_samples_canopy,len(ObjParams)))
    
    for i,rho in enumerate(rho_canopy_nadir):
        
        rho_i = [rho[index_wls]]
        args = (ObjParams, 
                FixedValues, 
                n_obs, 
                rho_i, 
                [0.0], 
                [37.0], 
                [0.0], 
                [0.2], 
                rsoil, 
                wls_inv, 
                scale)
        result=op.minimize(fcost_jac,
                          x0,
                          method = 'L-BFGS-B',
                          jac = True, 
                          args = args,
                          bounds = bounds_lbgfs,
                          options = { 'maxiter': 100000}) 
        
        estimates=result.x
                      
        
        var = np.asarray(estimates)*np.asarray(scale)[:,1] + np.asarray(scale)[:,0]
        estimated[i,:] = var
        print('inversion %s/%s'%(i, N_samples_canopy))
    
    
    for i, param in enumerate(ObjParams):
#==============================================================================
#         plt.figure()
#         plt.plot(estimated[:,i], samples_new[:,i], 'b.')
#         plt.title('%s %s'%(filters, param))
#         lims = (scale[i][0],scale[i][0]+scale[i][1])
#         plt.xlim(lims)
#         plt.ylim(lims)
#         plt.xlabel('Inverse model')
#         plt.ylabel('Forward model')
#         plt.plot(lims,lims,'r--')
#         rmse = np.sqrt(np.mean((estimated[:,i] - samples_new[:,i])**2))
#         cor = st.pearsonr(estimated[:,i], samples_new[:,i])[0]
#         bias = np.mean(estimated[:,i] - samples_new[:,i])
#         plt.figtext(0.2,0.7, 
#                     'RMSD:  %s\nbias:  %s\nr:     %s'%(
#                             np.round(rmse,2),
#                             np.round(bias,2),
#                             np.round(cor,3)),
#                     backgroundcolor='white',
#                     linespacing=1.15, 
#                     family='monospace')
#         plt.savefig(pth.join(pth.expanduser('~'),
#                              '%s_%s.png'%(param,filters)),
#                     dpi=100)
#         plt.close()
#==============================================================================


        # Calculate the point density
        xy = np.vstack([estimated[:,i], samples_new[:,i]])
        z = st.gaussian_kde(xy)(xy)
        
        # Sort the points by density, so that the densest points are plotted last
        plt.figure()
        idx = z.argsort()
        x, y, z = estimated[:,i][idx], samples_new[:,i][idx], z[idx]
        
        plt.scatter(x, y, c=z, s=5, edgecolor='')
        plt.title('%s %s'%(filters, param))
        lims = (scale[i][0],scale[i][0]+scale[i][1])
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel('Inverse model')
        plt.ylabel('Forward model')
        plt.plot(lims,lims,'k--')
        rmse = np.sqrt(np.mean((estimated[:,i] - samples_new[:,i])**2))
        cor = st.pearsonr(estimated[:,i], samples_new[:,i])[0]
        bias = np.mean(estimated[:,i] - samples_new[:,i])
        plt.figtext(0.2,0.7, 
                    'RMSD:  %s\nbias:  %s\nr:     %s'%(
                            np.round(rmse,2),
                            np.round(bias,2),
                            np.round(cor,3)),
                    backgroundcolor='white',
                    linespacing=1.15, 
                    family='monospace')
                    
        plt.savefig(pth.join(pth.expanduser('~'),
                             '%s_%s_density.png'%(param, filters)),
                    dpi=100)
        plt.close() 
        
# Inversion for tow observations
wls_inv = band_selection['SA']
n_obs = 2
print('# Inversion for optlimized wavelengths and two observations')

# Read the soil reflectance        
rsoil=np.genfromtxt(pth.join(
                    soil_folder,
                    'ipgp.jussieu.soil.prosail.dry.coarse.1.spectrum.txt'))

rsoil =np.array(rsoil[:,1])
index_wls = []
for wl in wls_inv:
    index_wls.append(int(np.where(wl == l)[0]))
    
    
rsoil = rsoil[index_wls]
dtype = [(param, 'f4') for param in ObjParams]
estimated = np.zeros((N_samples_canopy,len(ObjParams)))
    
for i,rho in enumerate(rho_canopy_nadir):
    
    rho_i = [rho[index_wls]]
    rho_i.append(rho_canopy_oblique[i][index_wls])
    args = (ObjParams, 
            FixedValues, 
            n_obs, 
            rho_i, 
            [0.0, 40.0], 
            [37.0, 37.0], 
            [0.0, 0.0], 
            [0.2, 0.2], 
            rsoil, 
            wls_inv, 
            scale)
    result=op.minimize(fcost_jac,
                      x0,
                      method = 'L-BFGS-B',
                      jac = True, 
                      args = args,
                      bounds = bounds_lbgfs,
                      options = { 'maxiter': 100000}) 
    
    estimates=result.x
                  
    
    var = np.asarray(estimates)*np.asarray(scale)[:,1] + np.asarray(scale)[:,0]
    estimated[i,:] = var
    print('inversion %s/%s'%(i, N_samples_canopy))

for i, param in enumerate(ObjParams):
#==============================================================================
#     plt.figure()
#     plt.plot(estimated[:,i], samples_new[:,i], 'b.')
#     plt.title('dual angle %s'%param)
#     lims = (scale[i][0],scale[i][0]+scale[i][1])
#     plt.xlim(lims)
#     plt.ylim(lims)
#     plt.xlabel('Inverse model')
#     plt.ylabel('Forward model')
#     plt.plot(lims,lims,'r--')
#     rmse = np.sqrt(np.mean((estimated[:,i] - samples_new[:,i])**2))
#     cor = st.pearsonr(estimated[:,i], samples_new[:,i])[0]
#     bias = np.mean(estimated[:,i] - samples_new[:,i])
#     plt.figtext(0.2,0.7, 
#                 'RMSD:  %s\nbias:  %s\nr:     %s'%(
#                         np.round(rmse,2),
#                         np.round(bias,2),
#                         np.round(cor,3)),
#                 backgroundcolor='white',
#                 linespacing=1.15, 
#                 family='monospace')
#     plt.savefig(pth.join(pth.expanduser('~'),
#                          '%s_SA_2angles.png'%param),
#                 dpi=100)
#     plt.close()
#==============================================================================
    
    # Calculate the point density
    xy = np.vstack([estimated[:,i], samples_new[:,i]])
    z = st.gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    plt.figure()
    idx = z.argsort()
    x, y, z = estimated[:,i][idx], samples_new[:,i][idx], z[idx]
    
    plt.scatter(x, y, c=z, s=5, edgecolor='')
    plt.title('dual angle %s'%param)
    lims = (scale[i][0],scale[i][0]+scale[i][1])
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Inverse model')
    plt.ylabel('Forward model')
    plt.plot(lims,lims,'k--')
    rmse = np.sqrt(np.mean((estimated[:,i] - samples_new[:,i])**2))
    cor = st.pearsonr(estimated[:,i], samples_new[:,i])[0]
    bias = np.mean(estimated[:,i] - samples_new[:,i])
    plt.figtext(0.2,0.7, 
                'RMSD:  %s\nbias:  %s\nr:     %s'%(
                        np.round(rmse,2),
                        np.round(bias,2),
                        np.round(cor,3)),
                backgroundcolor='white',
                linespacing=1.15, 
                family='monospace')
    plt.savefig(pth.join(pth.expanduser('~'),
                         '%s_SA_2angles_density.png'%param),
                dpi=100)
    plt.close()
