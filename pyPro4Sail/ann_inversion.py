# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:56:25 2016

@author: hector
"""
import numpy as np
import sklearn.neural_network as ann_sklearn
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from collections import OrderedDict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
import pyPro4Sail.ProspectD as ProspectD 
import pyPro4Sail.FourSAIL as FourSAIL 
import numpy.random as rnd
from scipy.stats import gamma
from scipy.ndimage.filters import gaussian_filter1d

UNIFORM_DIST = 1
GAUSSIAN_DIST = 2
GAMMA_DIST = 3
# Angle step to perform the hemispherical integration
STEP_VZA = 2.5
STEP_PSI = 30

MEAN_N_LEAF = 1.601 # From LOPEX + ANGERS average
MEAN_CAB = 41.07 # From LOPEX + ANGERS average
MEAN_CAR = 9.55 # From LOPEX + ANGERS average
MEAN_CBROWN = 0.2 # from S2 L2B ATBD
MEAN_CM = 0.0114 # From LOPEX + ANGERS average
MEAN_CW = 0.0053 # From LOPEX + ANGERS average
MEAN_ANT = 1.0
MEAN_LAI = 2.0 # from S2 L2B ATBD
MEAN_LEAF_ANGLE = 60.0 # from S2 L2B ATBD
MEAN_HOTSPOT = 0.2 # from S2 L2B ATBD

STD_N_LEAF = 0.305 # From LOPEX + ANGERS average
STD_CAB = 20.55 # From LOPEX + ANGERS average
STD_CAR = 4.69 # From LOPEX + ANGERS average
STD_CBROWN = 0.3 # from S2 L2B ATBD
STD_CM = 0.0031 # From LOPEX + ANGERS average
STD_CW = 0.0061 # From LOPEX + ANGERS average
STD_ANT = 10
STD_LAI = 3.0 # from S2 L2B ATBD
STD_LEAF_ANGLE = 30 # from S2 L2B ATBD
STD_HOTSPOT = 0.2 # from S2 L2B ATBD

MIN_N_LEAF = 1.0 # From LOPEX + ANGERS average
MIN_CAB = 0.0 # From LOPEX + ANGERS average
MIN_CAR = 0.0 # From LOPEX + ANGERS average
MIN_CBROWN = 0.0 # from S2 L2B ATBD
MIN_CM = 0.0017 # From LOPEX + ANGERS average
MIN_CW = 0.000 # From LOPEX + ANGERS average
MIN_ANT = 0.0
MIN_LAI = 0.0
MIN_LEAF_ANGLE = 30.0 # from S2 L2B ATBD
MIN_HOTSPOT = 0.1 # from S2 L2B ATBD

MAX_N_LEAF = 3.0 # From LOPEX + ANGERS average
MAX_CAB = 110.0 # From LOPEX + ANGERS average
MAX_CAR = 30.0 # From LOPEX + ANGERS average
MAX_CBROWN = 2.00 # from S2 L2B ATBD
MAX_CM = 0.0331 # From LOPEX + ANGERS average
MAX_CW = 0.0525 # From LOPEX + ANGERS average
MAX_ANT = 40.0
MAX_LAI = 15.0 # from S2 L2B ATBD
MAX_LEAF_ANGLE = 80.0 # from S2 L2B ATBD
MAX_HOTSPOT = 0.5 # from S2 L2B ATBD

# log Covariance matrix
# 'N', 'C_ab', 'C_car', 'EWT', 'LMA'
LOG_COV = np.array([[0.02992514, 0.03325522, 0.03114539, 0.03282891, 0.04783304],
                    [0.03325522, 0.52976802, 0.35201248, 0.02353788, 0.11989431],
                    [0.03114539, 0.35201248, 0.34761489, 0.00203875, 0.12227829],
                    [0.03282891, 0.02353788, 0.00203875, 0.17976   , 0.08260031],
                    [0.04783304, 0.11989431, 0.12227829, 0.08260031, 0.23379769]])
                    
# log means
# 'N', 'C_ab', 'C_car', 'EWT', 'LMA'
LOG_MEAN = np.array([ 0.45475538,  3.52688334,  2.12818934, -4.60373468, -5.36951247])
 
font = {'family' : 'monospace',
        'size'   : 8}

matplotlib.rc('font', **font)
prospect_bounds = {'N_leaf': (MIN_N_LEAF, MAX_N_LEAF),
                   'Cab': (MIN_CAB, MAX_CAB),
                   'Car': (MIN_CAR, MAX_CAR),
                   'Cbrown': (MIN_CBROWN, MAX_CBROWN),
                   'Cw': (MIN_CW, MAX_CW),
                   'Cm': (MIN_CM, MAX_CM),
                   'Ant': (MIN_ANT, MAX_ANT)}

prospect_moments = {'N_leaf': (MEAN_N_LEAF, STD_N_LEAF),
                    'Cab': (MEAN_CAB, STD_CAB),
                    'Car': (MEAN_CAR, STD_CAR),
                    'Cbrown': (MEAN_CBROWN, STD_CBROWN),
                    'Cw': (MEAN_CW, STD_CW),
                    'Cm': (MEAN_CM, STD_CM),
                    'Ant':  (MEAN_ANT, STD_ANT)}                 

prospect_distribution = {'N_leaf': GAMMA_DIST,
                         'Cab': GAMMA_DIST,
                         'Car': GAMMA_DIST,
                         'Cbrown': GAMMA_DIST,
                         'Cw': GAMMA_DIST,
                         'Cm': GAMMA_DIST,
                         'Ant': GAMMA_DIST}

prospect_covariates = {'N_leaf': ((MIN_N_LEAF, MAX_N_LEAF),
                                  (1.3,1.8)),
                       'Car': ((MIN_CAR, MAX_CAR),
                                (20,40)),
                       'Cbrown': ((MIN_CBROWN, MAX_CBROWN),
                                   (0,0.2)),
                       'Cw': ((MIN_CW, MAX_CW),
                              (0.005,0.011)),
                       'Cm': ((MIN_CM, MAX_CM),
                              (0.005,0.011)),
                       'Ant': ((MIN_ANT, MAX_ANT),
                              (0,40))}

prosail_bounds = {'N_leaf': (MIN_N_LEAF, MAX_N_LEAF),
                  'Cab': (MIN_CAB, MAX_CAB),
                  'Car': (MIN_CAR, MAX_CAR),
                  'Cbrown': (MIN_CBROWN, MAX_CBROWN),
                  'Cw': (MIN_CW, MAX_CW),
                  'Cm': (MIN_CM, MAX_CM),
                  'Ant': (MIN_ANT, MAX_ANT),
                  'LAI': (MIN_LAI, MAX_LAI),
                  'leaf_angle': (MIN_LEAF_ANGLE, MAX_LEAF_ANGLE),
                  'hotspot': (MIN_HOTSPOT, MAX_HOTSPOT)}

prosail_moments = {'N_leaf': (MEAN_N_LEAF, STD_N_LEAF),
                   'Cab': (MEAN_CAB, STD_CAB),
                   'Car': (MEAN_CAR, STD_CAR),
                   'Cbrown': (MEAN_CBROWN, STD_CBROWN),
                   'Cw': (MEAN_CW, STD_CW),
                   'Cm': (MEAN_CM, STD_CM),
                   'Ant': (MEAN_ANT, STD_ANT),
                   'LAI': (MEAN_LAI, STD_LAI),
                   'leaf_angle': (MEAN_LEAF_ANGLE, STD_LEAF_ANGLE),
                   'hotspot': (MEAN_HOTSPOT, STD_HOTSPOT)}

prosail_distribution = {'N_leaf': UNIFORM_DIST,
                        'Cab': GAUSSIAN_DIST,
                        'Car': GAUSSIAN_DIST,
                        'Cbrown': UNIFORM_DIST,
                        'Cw': GAUSSIAN_DIST,
                        'Cm': GAUSSIAN_DIST,
                        'Ant': GAUSSIAN_DIST,
                        'LAI': GAUSSIAN_DIST,
                        'leaf_angle': GAUSSIAN_DIST,
                        'hotspot': GAUSSIAN_DIST}

prosail_covariates = {'N_leaf': ((MIN_N_LEAF, MAX_N_LEAF),
                                 (1.3,1.8)),
                      'Cab': ((MIN_CAB, MAX_CAB),
                              (45,100)),
                      'Car': ((MIN_CAR, MAX_CAR),
                              (20,40)),
                      'Cbrown': ((MIN_CBROWN, MAX_CBROWN),
                                 (0,0.2)),
                      'Cw': ((MIN_CW, MAX_CW),
                             (0.005,0.011)),
                      'Cm': ((MIN_CM, MAX_CM),
                             (0.005,0.011)),
                      'Ant': ((MIN_ANT, MAX_ANT),
                              (0,40)),
                      'leaf_angle': ((MIN_LEAF_ANGLE, MAX_LEAF_ANGLE),
                                     (55,65)),
                      'hotspot': ((MIN_HOTSPOT, MAX_HOTSPOT),
                                  (0.1,0.5))}

def train_ann(X_array,
             Y_array, 
             scaling_input=None, 
             scaling_output=None,
             reduce_pca=False,
             outfile=None,
             regressor_opts={'activation': 'logistic'}):

    print('Fitting Artificial Neural Network')

    Y_array = np.asarray(Y_array)
    X_array = np.asarray(X_array)

    pca = None
    input_scaler = None
    output_scaler = None
    
    # Normalize the input
    if scaling_input == 'minmax':
        input_scaler = MinMaxScaler()
        X_array = input_scaler.fit_transform(X_array)
        if outfile:
            fid = open(outfile + '_scaler_input', 'wb')
            pickle.dump(input_scaler, fid, -1)
            fid.close()
    elif scaling_input == 'maxabs':   
        input_scaler = MaxAbsScaler()
        X_array = input_scaler.fit_transform(X_array)
        if outfile:
            fid = open(outfile + '_scaler_input', 'wb')
            pickle.dump(input_scaler, fid, -1)
            fid.close()
    elif scaling_input == 'normalize':   
        input_scaler = StandardScaler()
        X_array = input_scaler.fit_transform(X_array)
        if outfile:
            fid = open(outfile + '_scaler_input', 'wb')
            pickle.dump(input_scaler, fid, -1)
            fid.close()

    if reduce_pca:
        #Reduce input variables using Principal Component Analisys
        pca = PCA(n_components=10)
        X_array = pca.fit_transform(X_array)
        if outfile:
            fid = open(outfile + '_PCA', 'wb')
            pickle.dump(pca, fid, -1)
            fid.close()

    # Normalize the output
    if scaling_output == 'minmax':
        output_scaler = MinMaxScaler()
        Y_array = output_scaler.fit_transform(Y_array)       
        if outfile:
            fid = open(outfile + '_scaler_output', 'wb')
            pickle.dump(output_scaler, fid, -1)
            fid.close()
    elif scaling_output == 'maxabs':   
        output_scaler = MaxAbsScaler()
        Y_array = output_scaler.fit_transform(Y_array)       
        if outfile:
            fid = open(outfile + '_scaler_output', 'wb')
            pickle.dump(output_scaler, fid, -1)
            fid.close()
    elif scaling_output == 'normalize':   
        output_scaler = StandardScaler()
        Y_array = output_scaler.fit_transform(Y_array)       
        if outfile:
            fid = open(outfile + '_scaler_output', 'wb')
            pickle.dump(output_scaler, fid, -1)
            fid.close()

    # Get the number of bands to set the ANN structure
    ann = ann_sklearn.MLPRegressor(**regressor_opts)
    
    ANN = ann.fit(X_array, Y_array)
    if outfile:    
        fid=open(outfile,'wb')
        pickle.dump(ANN, fid, -1)
        fid.close()
    
    return ANN, input_scaler, output_scaler, pca

def test_ann(X_array,
             Y_array, 
             ann_object, 
             scaling_input=None,
             scaling_output=None,
             reduce_pca=None, 
             outfile=None,
             param_names=None):
    
    print('Testing ANN fit')
    
    X_array = np.asarray(X_array)
    Y_array = np.asarray(Y_array)
    
    # Estimate error measuremenents
    RMSE = []
    bias = []
    cor = []
    
    if scaling_input:
        X_array = scaling_input.transform(X_array)
    if reduce_pca:
        X_array = reduce_pca.transform(X_array)
    
    Y_test = ann_object.predict(X_array)
    
    if scaling_output:
        Y_test = scaling_output.inverse_transform(Y_test)

    Y_test.reshape(-1)    
    
    f = plt.figure()
    RMSE.append(np.sqrt(np.sum((Y_array - Y_test)**2)
                        / np.size(Y_array)))
    bias.append(np.mean(Y_array - Y_test))
    cor.append(pearsonr(Y_array, Y_test)[0])
    plt.scatter(Y_array,
                Y_test,
                color='black', 
                s=5,
                alpha=0.1, 
                marker='.')

    absline=np.asarray([[np.amin(Y_array), np.amax(Y_array)],
                         [np.amin(Y_array), np.amax(Y_array)]])
    
    plt.plot(absline[0], absline[1], color='red')
    plt.title(param_names[0])
    if outfile:        
        f.savefig(outfile)
    plt.close()
    
    return RMSE, bias, cor

def build_prospect_database(n_simulations,
                            param_bounds=prospect_bounds,
                            moments=prospect_moments,
                            distribution=prospect_distribution,
                            apply_covariate={'N_leaf': False,
                                         'Car': False,
                                         'Cbrown': False,
                                         'Cw': False,
                                         'Cm': False,
                                         'Ant': False},
                            covariate=prospect_covariates,
                            outfile=None):
            
    
    print ('Build ProspectD database')
    input_param=dict()
    for param in param_bounds:
        if distribution[param] == UNIFORM_DIST:
            input_param[param] = param_bounds[param][0] \
                                    + rnd.rand(n_simulations) * (param_bounds[param][1] 
                                                                - param_bounds[param][0])

        elif distribution[param] == GAUSSIAN_DIST:
            input_param[param] = moments[param][0]\
                                    + rnd.randn(n_simulations) * moments[param][1]
            
        
        elif distribution[param] == GAMMA_DIST:
            scale = moments[param][1]**2 / (moments[param][0] - param_bounds[param][0])
            shape = (moments[param][0] - param_bounds[param][0]) / scale
            input_param[param] = gamma.rvs(shape,
                                           scale=scale,
                                           loc=param_bounds[param][0],
                                           size=n_simulations)

        input_param[param] = np.clip(input_param[param],
                                     param_bounds[param][0],
                                     param_bounds[param][1])


    # Apply covariates where needed
    Cab_range = param_bounds['Cab'][1] - param_bounds['Cab'][0]
    for param in apply_covariate:
        if apply_covariate:
            Vmin = covariate[param][0][0] \
                    + input_param['Cab'] * (covariate[param][1][0]
                                            - covariate[param][0][0]) / Cab_range
            Vmax = covariate[param][0][1] \
                    + input_param['Cab'] * (covariate[param][1][1]
                                            - covariate[param][0][1]) / Cab_range
            
            input_param[param] = np.clip(input_param[param], Vmin, Vmax)
            
            
    return input_param

def build_prosail_database(n_simulations,
                           param_bounds=prosail_bounds,
                           moments=prosail_moments,
                           distribution=prosail_distribution,
                           apply_covariate={'N_leaf': True,
                                             'Cab': True,
                                             'Car': True,
                                             'Cbrown': True,
                                             'Cw': True,
                                             'Cm': True,
                                             'Ant': True,
                                             'leaf_angle': True,
                                             'hotspot': True},
                           covariate=prosail_covariates,
                           outfile=None):
            
    
    print ('Build ProspectD+4SAIL database')
    input_param=dict()
    for param in param_bounds:
        if distribution[param] == UNIFORM_DIST:
            input_param[param] = param_bounds[param][0] \
                                    + rnd.rand(n_simulations) * (param_bounds[param][1] 
                                                                - param_bounds[param][0])

        elif distribution[param] == GAUSSIAN_DIST:
            input_param[param] = moments[param][0] \
                                    + rnd.randn(n_simulations) * moments[param][1]
            
        
        elif distribution[param] == GAMMA_DIST:
            scale = moments[param][1]**2 / (moments[param][0] - param_bounds[param][0])
            shape = (moments[param][0] - param_bounds[param][0]) / scale
            input_param[param] = gamma.rvs(shape,
                                           scale=scale,
                                           loc=param_bounds[param][0],
                                           size=n_simulations)

        input_param[param] = np.clip(input_param[param],
                                     param_bounds[param][0],
                                     param_bounds[param][1])

            
    # Apply covariates where needed
    LAI_range = param_bounds['LAI'][1] - param_bounds['LAI'][0]
    for param in apply_covariate:
        if apply_covariate:
            Vmin = covariate[param][0][0] \
                    +input_param['LAI'] * (covariate[param][1][0]
                                            - covariate[param][0][0]) / LAI_range
            Vmax = covariate[param][0][1] \
                    +input_param['LAI'] * (covariate[param][1][1]
                                            - covariate[param][0][1]) / LAI_range
            input_param[param]=np.clip(input_param[param], Vmin, Vmax)
            
            
    return input_param


def simulate_prospectD_lut(input_param,
                           wls_sim,
                           srf=None,
                           outfile=None):            
    
    [wls,r,t]=ProspectD.ProspectD_vec(input_param['N_leaf'],
                input_param['Cab'], input_param['Car'],
                input_param['Cbrown'],input_param['Cw'],
                input_param['Cm'],input_param['Ant'])
    #Convolve the simulated spectra to a gaussian filter per band
    rho_leaf=[]
    tau_leaf=[]

    if srf:
        if type(srf)==float or type(srf)==int:
            wls_sim=np.asarray(wls_sim)
            #Convolve spectra by full width half maximum
            sigma=fwhm2sigma(srf)
            r=gaussian_filter1d(r,sigma)
            t=gaussian_filter1d(t,sigma)
            for wl in wls_sim:
                rho_leaf.append(float(r[wls==wl]))
                tau_leaf.append(float(t[wls==wl]))

        elif type(srf)==list or type(srf)==tuple:
            for weight in srf:
                weight = np.tile(weight, (1, r.shape[2])).T
                rho_leaf.append(float(np.sum(weight*r, axis=0)/np.sum(weight, axis=0)))
                tau_leaf.append(float(np.sum(weight*t, axis=0)/np.sum(weight, axis=0)))

    else:
        rho_leaf=np.copy(r)                
        tau_leaf=np.copy(t)
        
    rho_leaf=np.asarray(rho_leaf)
    tau_leaf=np.asarray(tau_leaf)
                        
    
    if outfile:
        fid=open(outfile+'_rho','wb')
        pickle.dump(rho_leaf,fid,-1)
        fid.close()
        fid=open(outfile+'_param','wb')
        pickle.dump(input_param,fid,-1)
        fid.close()

    return rho_leaf,input_param


def simulate_prosail_lut(input_param,
                         wls_sim,
                         rsoil_vec,
                         skyl=0.1,
                         sza=37,
                         vza=0,
                         psi=0,
                         srf=None,
                         outfile=None,
                         calc_FAPAR=False,
                         reduce_4sail=False):            
    
    print ('Starting Simulations')

    # Calculate the lidf
    lidf = FourSAIL.CalcLIDF_Campbell_vec(input_param['leaf_angle'])
    #for i,wl in enumerate(wls_wim):
    [wls,r,t] = ProspectD.ProspectD_vec(input_param['N_leaf'],
                                        input_param['Cab'],
                                        input_param['Car'],
                                        input_param['Cbrown'],
                                        input_param['Cw'],
                                        input_param['Cm'],
                                        input_param['Ant'])
     
    r = r.T
    t = t.T 
    
    if type(skyl) == float:
        skyl = skyl * np.ones(r.shape)

    if calc_FAPAR:
        par_index = wls<=700
        rho_leaf_fapar = np.mean(r[par_index], axis=0).reshape(1, -1)
        tau_leaf_fapar = np.mean(t[par_index], axis=0).reshape(1, -1)
        skyl_rho_fapar = np.mean(skyl[par_index], axis=0).reshape(1, -1)
        rsoil_vec_fapar = np.mean(rsoil_vec[par_index], axis=0).reshape(1, -1)

        par_index = wls_sim<=700
                             
    #Convolve the simulated spectra to a gaussian filter per band
    rho_leaf = []
    tau_leaf = []
    skyl_rho = []
    rsoil = []
    if srf and reduce_4sail:
        if type(srf) == float or type(srf) == int:

            wls_sim = np.asarray(wls_sim)
            #Convolve spectra by full width half maximum
            sigma = fwhm2sigma(srf)
            r = gaussian_filter1d(r, sigma, axis=1)
            t = gaussian_filter1d(t, sigma, axis=1)
            s = gaussian_filter1d(skyl, sigma, axis=1)
            soil = gaussian_filter1d(rsoil_vec, sigma, axis=1)
            for wl in wls_sim:
                rho_leaf.append(r[wls==wl].reshape(-1))
                tau_leaf.append(t[wls==wl].reshape(-1))
                skyl_rho.append(s[wls==wl].reshape(-1))
                rsoil.append(soil[wls==wl].reshape(-1))

        elif type(srf) == list or type(srf) == tuple:
            skyl = np.tile(skyl, (r.shape[1], 1)).T
            for weight in srf:
                weight = np.tile(weight, (r.shape[1], 1)).T
                rho_leaf.append(np.sum(weight * r, axis=0) / np.sum(weight, axis=0))
                tau_leaf.append(np.sum(weight * t, axis=0) / np.sum(weight, axis=0))
                skyl_rho.append(np.sum(weight * skyl, axis=0) / np.sum(weight, axis=0))
                rsoil.append(np.sum(weight * rsoil_vec, axis=0) / np.sum(weight, axis=0))
            skyl_rho = np.asarray(skyl_rho)
            
    elif reduce_4sail:
        wls_sim = np.asarray(wls_sim)
        for wl in wls_sim:
            rho_leaf.append(r[wls==wl].reshape(-1))
            tau_leaf.append(t[wls==wl].reshape(-1))
            skyl_rho.append(skyl[wls==wl].reshape(-1))
            rsoil.append(rsoil_vec[wls==wl].reshape(-1))
    else:
        rho_leaf = r.T
        tau_leaf = t.T
        skyl_rho = np.asarray(skyl.T)
        rsoil = np.asarray(rsoil_vec)
     
    rho_leaf = np.asarray(rho_leaf)
    tau_leaf = np.asarray(tau_leaf)              
    skyl_rho = np.asarray(skyl_rho)
    rsoil = np.asarray(rsoil)
    
    [_, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     _, 
     rdot,
     _, 
     _,
     rsot, 
     _, 
     _, 
     _] = FourSAIL.FourSAIL_vec(input_param['LAI'],
                                input_param['hotspot'],
                                lidf,
                                np.ones(input_param['LAI'].shape) * sza,
                                np.ones(input_param['LAI'].shape) * vza,
                                np.ones(input_param['LAI'].shape) * psi,
                                rho_leaf,
                                tau_leaf,
                                rsoil)
    
                
    r2 = rdot * skyl_rho + rsot * (1 - skyl_rho)
    del rdot, rsot
    
    if calc_FAPAR:
        fAPAR_array, fIPAR_array = calc_fapar_4sail(skyl_rho_fapar,
                                                    input_param['LAI'],
                                                    lidf,
                                                    input_param['hotspot'],
                                                    np.ones(input_param['LAI'].shape) * sza,
                                                    rho_leaf_fapar,
                                                    tau_leaf_fapar,
                                                    rsoil_vec_fapar)
        
        fAPAR_array[np.isfinite(fAPAR_array)] = np.clip(fAPAR_array[np.isfinite(fAPAR_array)],
                                                                    0,
                                                                    1)
        
        fIPAR_array[np.isfinite(fIPAR_array)] = np.clip(fIPAR_array[np.isfinite(fIPAR_array)],
                                                                    0,
                                                                    1)

        input_param['fAPAR'] = fAPAR_array
        input_param['fIPAR'] = fIPAR_array
        del fAPAR_array, fIPAR_array
            
    
    rho_canopy = []      
    if srf and not reduce_4sail:
        if type(srf) == float or type(srf) == int:
            #Convolve spectra by full width half maximum
            sigma = fwhm2sigma(srf)
            r2 = gaussian_filter1d(r2, sigma, axis=1)
            for wl in wls_sim:
                rho_canopy.append(r2[wls==wl].reshape(-1))

        elif type(srf)==list or type(srf)==tuple:
            for weight in srf:
                weight = np.tile(weight, (1, r2.shape[2])).T
                rho_canopy.append(np.sum(weight*r2, axis=0) / np.sum(weight, axis=0))
    elif reduce_4sail:
        rho_canopy = np.asarray(r2)
        
    else:
        for wl in wls_sim:
            rho_canopy.append(r2[wls==wl].reshape(-1))

    rho_canopy=np.asarray(rho_canopy)
    
    if outfile:
        fid = open(outfile + '_rho', 'wb')
        pickle.dump(rho_canopy, fid, -1)
        fid.close()
        fid = open(outfile + '_param', 'wb')
        pickle.dump(input_param, fid, -1)
        fid.close()

    return rho_canopy.T, input_param


def calc_fapar_4sail(skyl,
                     LAI,
                     lidf,
                     hotspot,
                     sza,
                     rho_leaf,
                     tau_leaf,
                     rsoil):
                         
    '''Estimates the fraction of Absorbed and intercepted PAR using the 4SAIL 
         Radiative Transfer Model.
        
    Parameters
    ----------
    skyl : float
        Ratio of diffuse to total PAR radiation
    LAI : float
        Leaf (Plant) Area Index
    lidf : list
        Leaf Inclination Distribution Function, 5 degrees step
    hotspot : float
        hotspot parameters, use 0 to ignore the hotspot effect (turbid medium)
    sza : float
        Sun Zenith Angle (degrees)
    rho_leaf : list
        Narrowband leaf bihemispherical reflectance, 
            it might be simulated with PROSPECT (400-700 @1nm)
    tau_leaf : list
        Narrowband leaf bihemispherical transmittance, 
            it might be simulated with PROSPECT (400-700 @1nm)
    rsoil : list
        Narrowband soil bihemispherical reflectance (400-700 @1nm)
    
    Returns
    -------
    fAPAR : float
        Fraction of Absorbed Photosynthetically Active Radiation
    fIPAR : float
        Fraction of Intercepted Photosynthetically Active Radiation'''
    
    # Diffuse and direct irradiance
    Es = 1.0 - skyl
    Ed = skyl

    #Initialize values
    S_0 = np.zeros(rho_leaf.shape)
    S_1 = np.zeros(rho_leaf.shape)

    # Start the hemispherical integration
    vzas_psis=((vza, psi) for vza in np.arange(0, 90 - STEP_VZA/2., STEP_VZA) 
                          for psi in np.arange(0, 360, STEP_PSI))
    
    step_vza_radians, step_psi_radians = np.radians(STEP_VZA), np.radians(STEP_PSI)
    
    for vza, psi in vzas_psis:
        vza += STEP_VZA / 2.

        # Calculate the reflectance factor and project into the solid angle
        cosvza = np.cos(np.radians(vza))
        sinvza = np.sin(np.radians(vza))
        
        [tss,
         _,
         _,
         rdd,
         tdd,
         _,
         tsd,
         _,
         _,
         _,
         _,
         _,
         _,
         _,
         rdot,
         _,
         _,
         rsot,
         _,
         _,
         _] = FourSAIL.FourSAIL_vec(LAI,
                                    hotspot,
                                    lidf,
                                    sza,
                                    np.ones(LAI.shape) * vza,
                                    np.ones(LAI.shape) * psi,
                                    rho_leaf,
                                    tau_leaf,
                                    rsoil)
        
        # Downwelling solar beam radiation at ground level (beam transmissnion)
        Es_1 = tss * Es
        # Upwelling diffuse shortwave radiation at ground level
        Ed_up_1 = (rsoil * (Es_1 + tsd*Es + tdd*Ed)) / (1. - rsoil*rdd)
        # Downwelling diffuse shortwave radiation at ground level            
        Ed_down_1 = tsd*Es + tdd*Ed + rdd*Ed_up_1
        # Upwelling shortwave (beam and diffuse) radiation towards the observer (at angle psi/vza)
        Eo_0 = rdot * Ed + rsot * Es
        # Spectral flux at the top of the canopy        
        # & add the top of the canopy flux to the integral and continue through the hemisphere
        S_0 += Eo_0 * cosvza * sinvza * step_vza_radians * step_psi_radians 
        # Spectral flus at the bottom of the canopy
        # & add the bottom of the canopy flux to the integral and continue through the hemisphere
        S_1 += (Es_1 + tdd*Ed) * cosvza * sinvza * step_vza_radians * step_psi_radians / np.pi
        # Absorbed flux at ground lnevel
        Sn_soil = (1. - rsoil) * (Es_1 + Ed_down_1) # narrowband net soil shortwave radiation

    # Calculate the vegetation (sw/lw) net radiation (divergence) as residual of top of the canopy and net soil radiation
    Rn_sw_veg_nb = 1.0 - S_0 - Sn_soil  # narrowband net canopy sw radiation
    fAPAR = np.sum(Rn_sw_veg_nb, axis=0)/Rn_sw_veg_nb.shape[0]   # broadband net canopy sw radiation
    fIPAR = np.sum(1.0 - S_1, axis=0)/S_1.shape[0]
    return fAPAR, fIPAR

def inputdict2array(input_param, 
                    obj_param=('N_leaf',
                               'Cab',
                               'Car',
                               'Cbrown',
                               'Cm',
                               'Cw',
                               'Ant',
                               'LAI', 
                               'leaf_angle',
                               'hotspot')):
    
    Y_array = []
    for param in obj_param:
        Y_array.append(input_param[param])

    Y_array = np.asarray(Y_array).T
    
    return Y_array

   
def fwhm2sigma(fwhm):

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma

def alpha2x_LAD(alpha):
    alpha = np.radians(alpha)
    x_LAD = ((alpha / 9.65) ** (1. /-1.65)) - 3.
    
    return x_LAD

def x_LAD2alpha(x_LAD):
    alpha = 9.65 * (3 + x_LAD) ** -1.65
    alpha = np.degrees(alpha)
    return alpha