# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:56:25 2016

@author: hector
"""
from pathlib import Path
import numpy as np
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    print("scikit-learn-intelex not available, skipping the use af Intel "
          "acceleration.\n"
          "Try to consider installing Intel acceleration via "
          "`pip install scikit-learn-intelex`")

from sklearn.neural_network import MLPRegressor as ann_sklearn
from sklearn.ensemble import RandomForestRegressor as rf_sklearn
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn import svm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
import pypro4sail.prospect as prospect
import pypro4sail.four_sail as sail
import numpy.random as rnd
from scipy.stats import gamma, norm
from scipy.ndimage.filters import gaussian_filter1d
from SALib.sample import sobol
import pandas as pd
import multiprocessing as mp
from . import radiation_helpers as rad


UNIFORM_DIST = 1
GAUSSIAN_DIST = 2
GAMMA_DIST = 3
SALTELLI_DIST = 4
# Angle step to perform the hemispherical integration
STEP_VZA = 2.5
STEP_PSI = 30

MEAN_N_LEAF = 1.601  # From LOPEX + ANGERS average
MEAN_CAB = 41.07  # From LOPEX + ANGERS average
MEAN_CAR = 9.55  # From LOPEX + ANGERS average
MEAN_CBROWN = 0.0  # from S2 L2B ATBD
MEAN_CM = 0.0114  # From LOPEX + ANGERS average
MEAN_CW = 0.0053  # From LOPEX + ANGERS average
MEAN_ANT = 1.0
MEAN_LAI = 2.0  # from S2 L2B ATBD
MEAN_LEAF_ANGLE = 60.0  # from S2 L2B ATBD
MEAN_HOTSPOT = 0.2  # from S2 L2B ATBD
MEAN_BS = 1.2  # from S2 L2B ATBD

STD_N_LEAF = 0.305  # From LOPEX + ANGERS average
STD_CAB = 20.55  # From LOPEX + ANGERS average
STD_CAR = 4.69  # From LOPEX + ANGERS average
STD_CBROWN = 0.3  # from S2 L2B ATBD
STD_CM = 0.0031  # From LOPEX + ANGERS average
STD_CW = 0.0061  # From LOPEX + ANGERS average
STD_ANT = 10
STD_LAI = 3.0  # from S2 L2B ATBD
STD_LEAF_ANGLE = 30  # from S2 L2B ATBD
STD_HOTSPOT = 0.2  # from S2 L2B ATBD
STD_BS = 2.00  # from S2 L2B ATBD

MIN_N_LEAF = 1.0  # From LOPEX + ANGERS average
MIN_CAB = 0.0  # From LOPEX + ANGERS average
MIN_CAR = 0.0  # From LOPEX + ANGERS average
MIN_CBROWN = 0.0  # from S2 L2B ATBD
MIN_CM = 0.0017  # From LOPEX + ANGERS average
MIN_CW = 0.000  # From LOPEX + ANGERS average
MIN_ANT = 0.0
MIN_LAI = 0.0
MIN_LEAF_ANGLE = 20.0  # from S2 L2B ATBD
MIN_HOTSPOT = 0.05  # from S2 L2B ATBD
MIN_BS = 0.50  # from S2 L2B ATBD

MAX_N_LEAF = 3.0  # From LOPEX + ANGERS average
MAX_CAB = 110.0  # From LOPEX + ANGERS average
MAX_CAR = 30.0  # From LOPEX + ANGERS average
MAX_CBROWN = 2.00  # from S2 L2B ATBD
MAX_CM = 0.0331  # From LOPEX + ANGERS average
MAX_CW = 0.0525  # From LOPEX + ANGERS average
MAX_ANT = 40.0
MAX_LAI = 8.0  # from effective LAI in Valeri sites
MAX_LEAF_ANGLE = 80.0  # from S2 L2B ATBD
MAX_HOTSPOT = 1  # from S2 L2B ATBD
MAX_BS = 3.5  # from S2 L2B ATBD

# log Covariance matrix
# 'N', 'C_ab', 'C_car', 'EWT', 'LMA'
LOG_COV = np.array(
    [[0.02992514, 0.03325522, 0.03114539, 0.03282891, 0.04783304],
     [0.03325522, 0.52976802, 0.35201248, 0.02353788, 0.11989431],
     [0.03114539, 0.35201248, 0.34761489, 0.00203875, 0.12227829],
     [0.03282891, 0.02353788, 0.00203875, 0.17976, 0.08260031],
     [0.04783304, 0.11989431, 0.12227829, 0.08260031, 0.23379769]])

# log means
# 'N', 'C_ab', 'C_car', 'EWT', 'LMA'
LOG_MEAN = np.array(
    [0.45475538, 3.52688334, 2.12818934, -4.60373468, -5.36951247])

font = {'family': 'monospace',
        'size': 8}

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
                    'Ant': (MEAN_ANT, STD_ANT)}

prospect_distribution = {'N_leaf': GAMMA_DIST,
                         'Cab': GAMMA_DIST,
                         'Car': GAMMA_DIST,
                         'Cbrown': GAMMA_DIST,
                         'Cw': GAMMA_DIST,
                         'Cm': GAMMA_DIST,
                         'Ant': GAMMA_DIST}

prospect_covariates = {'N_leaf': ((MIN_N_LEAF, MAX_N_LEAF),
                                  (1.3, 1.8)),
                       'Car': ((MIN_CAR, MAX_CAR),
                               (20, 40)),
                       'Cbrown': ((MIN_CBROWN, MAX_CBROWN),
                                  (0, 0.2)),
                       'Cw': ((MIN_CW, MAX_CW),
                              (0.005, MAX_CW)),
                       'Cm': ((MIN_CM, MAX_CM),
                              (0.005, 0.011)),
                       'Ant': ((MIN_ANT, MAX_ANT),
                               (0, 10))}

prosail_bounds = {'N_leaf': (MIN_N_LEAF, MAX_N_LEAF),
                  'Cab': (MIN_CAB, MAX_CAB),
                  'Car': (MIN_CAR, MAX_CAR),
                  'Cbrown': (MIN_CBROWN, MAX_CBROWN),
                  'Cw': (MIN_CW, MAX_CW),
                  'Cm': (MIN_CM, MAX_CM),
                  'Ant': (MIN_ANT, MAX_ANT),
                  'LAI': (MIN_LAI, MAX_LAI),
                  'leaf_angle': (MIN_LEAF_ANGLE, MAX_LEAF_ANGLE),
                  'hotspot': (MIN_HOTSPOT, MAX_HOTSPOT),
                  'bs': (MIN_BS, MAX_BS)}

prosail_moments = {'N_leaf': (MEAN_N_LEAF, STD_N_LEAF),
                   'Cab': (MEAN_CAB, STD_CAB),
                   'Car': (MEAN_CAR, STD_CAR),
                   'Cbrown': (MEAN_CBROWN, STD_CBROWN),
                   'Cw': (MEAN_CW, STD_CW),
                   'Cm': (MEAN_CM, STD_CM),
                   'Ant': (MEAN_ANT, STD_ANT),
                   'LAI': (MEAN_LAI, STD_LAI),
                   'leaf_angle': (MEAN_LEAF_ANGLE, STD_LEAF_ANGLE),
                   'hotspot': (MEAN_HOTSPOT, STD_HOTSPOT),
                   'bs': (MEAN_BS, STD_BS)}

prosail_distribution = {'N_leaf': UNIFORM_DIST,
                        'Cab': GAUSSIAN_DIST,
                        'Car': GAUSSIAN_DIST,
                        'Cbrown': UNIFORM_DIST,
                        'Cw': GAUSSIAN_DIST,
                        'Cm': GAUSSIAN_DIST,
                        'Ant': GAUSSIAN_DIST,
                        'LAI': GAUSSIAN_DIST,
                        'leaf_angle': GAUSSIAN_DIST,
                        'hotspot': GAUSSIAN_DIST,
                        'bs': GAUSSIAN_DIST}

prosail_covariates = {'leaf_angle': ((MIN_LEAF_ANGLE, MAX_LEAF_ANGLE),
                                     (55, 65)),
                      'hotspot': ((MIN_HOTSPOT, MAX_HOTSPOT),
                                  (0.1, 0.5)),
                      'bs': ((MIN_BS, MAX_BS),
                             (0.5, 1.2)),
                      **prospect_covariates}


def train_reg(X_array,
              Y_array,
              scaling_input=None,
              scaling_output=None,
              reduce_pca=False,
              outfile=None,
              reg_method="neural_network",
              regressor_opts={'activation': 'logistic'}):
    print('Fitting %s' % reg_method)

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
        # Reduce input variables using Principal Component Analisys
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

    if reg_method == "neural_network":
        reg = ann_sklearn(**regressor_opts)
    elif reg_method == "random_forest":
        reg = rf_sklearn(**regressor_opts)
    elif reg_method == "svm":
        reg = svm.SVR(**regressor_opts)
    reg_object = reg.fit(X_array, np.ravel(Y_array))
    if outfile:
        fid = open(outfile, 'wb')
        pickle.dump(reg_object, fid, -1)
        fid.close()

    return reg_object, input_scaler, output_scaler, pca


def test_reg(X_array,
             Y_array,
             reg_object,
             scaling_input=None,
             scaling_output=None,
             reduce_pca=None,
             outfile=None,
             param_name=None):
    print('Testing Regression fit')

    X_array = np.asarray(X_array)
    Y_array = np.asarray(Y_array)

    if scaling_input:
        X_array = scaling_input.transform(X_array)
    if reduce_pca:
        X_array = reduce_pca.transform(X_array)

    Y_test = reg_object.predict(X_array)

    if scaling_output:
        Y_test = scaling_output.inverse_transform(Y_test.reshape(-1, 1)).reshape(-1)

    f = plt.figure()
    rmse = np.sqrt(np.nanmean((Y_array - Y_test) ** 2))
    bias = np.nanmean(Y_array - Y_test)
    cor = pearsonr(Y_array, Y_test)[0]
    plt.scatter(Y_test,
                Y_array,
                color='black',
                s=5,
                alpha=0.1,
                marker='.')

    absline = np.asarray([[np.amin(Y_array), np.amax(Y_array)],
                          [np.amin(Y_array), np.amax(Y_array)]])

    plt.plot(absline[0], absline[1], color='red')
    plt.title(param_name)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.figtext(0.1, 0.8,
                "bias: {:>7.2f}\nrmse: {:>7.2f}\n   r: {:>7.2f}".format(bias,
                                                                        rmse,
                                                                        cor))
    plt.tight_layout()
    if outfile:
        f.savefig(outfile)
    plt.close()

    return rmse, bias, cor


def build_prospect_database(n_simulations,
                            param_bounds=None,
                            moments=None,
                            distribution=None,
                            apply_covariate=False,
                            covariate=None,
                            random_seed=None):
    if covariate is None:
        covariate = prospect_covariates
    if distribution is None:
        distribution = prospect_distribution
    if moments is None:
        moments = prospect_moments
    if param_bounds is None:
        param_bounds = prospect_bounds

    if isinstance(distribution, dict):
        input_param = probabilistic_distribution(n_simulations,
                                                 moments,
                                                 param_bounds,
                                                 distribution,
                                                 random_seed)
    elif distribution == SALTELLI_DIST:
        input_param = montecarlo_distribution(n_simulations, param_bounds, random_seed)

    # Apply covariates where needed
    if apply_covariate:
        input_param = covariate_constraint(input_param,
                                           param_bounds,
                                           covariate,
                                           "Cab")

    return input_param


def build_prosail_database(n_simulations,
                           param_bounds=None,
                           moments=None,
                           distribution=None,
                           apply_covariate=False,
                           covariate=None,
                           random_seed=None):
    if covariate is None:
        covariate = prosail_covariates
    if distribution is None:
        distribution = prosail_distribution
    if moments is None:
        moments = prosail_moments
    if param_bounds is None:
        param_bounds = prosail_bounds

    if isinstance(distribution, dict):
        input_param = probabilistic_distribution(n_simulations,
                                                 moments,
                                                 param_bounds,
                                                 distribution,
                                                 random_seed)
    elif distribution == SALTELLI_DIST:
        input_param = montecarlo_distribution(n_simulations, param_bounds, random_seed)

    # Apply covariates where needed
    if apply_covariate:
        input_param = covariate_constraint(input_param,
                                           param_bounds,
                                           covariate,
                                           "LAI")

    return input_param


def covariate_constraint(input_dict,
                         bounds,
                         covariates,
                         ref_param):
    range = bounds[ref_param][1] - bounds[ref_param][0]
    for param, limits in covariates.items():
        v_min = limits[0][0] + input_dict[ref_param] * (limits[1][0] -
                                                        limits[0][0]) / range
        v_max = limits[0][1] + input_dict[ref_param] * (limits[1][1] -
                                                        limits[0][1]) / range

        input_dict[param] = np.clip(input_dict[param], v_min, v_max)

    return input_dict


def probabilistic_distribution(simulations, moments_dict, bounds,
                            distribution_type, random_seed=None):
    rng = rnd.default_rng(random_seed)
    output = dict()
    for param in bounds:
        if distribution_type[param] == UNIFORM_DIST:
            output[param] = rng.uniform(low=bounds[param][0],
                                        high=bounds[param][1],
                                        size=simulations)

        elif distribution_type[param] == GAUSSIAN_DIST:
            output[param] = rng.normal(loc=moments_dict[param][0],
                                       scale=moments_dict[param][1],
                                       size=simulations)


        elif distribution_type[param] == GAMMA_DIST:
            scale = moments_dict[param][1] ** 2 / (
                    moments_dict[param][0] - bounds[param][0])
            shape = (moments_dict[param][0] - bounds[param][0]) / scale
            output[param] = gamma.rvs(shape,
                                      scale=scale,
                                      loc=bounds[param][0],
                                      size=simulations,
                                      random_state=rng)

        output[param] = np.clip(output[param],
                                bounds[param][0],
                                bounds[param][1])

    return output


def montecarlo_distribution(simulations, bounds, random_seed=None):
    problem = {'num_vars': len(bounds),
               'names': [name for name in bounds.keys()],
               'bounds': [bounds for key, bounds in
                          bounds.items()]
               }
    # The total number of simulations by saltelli.sample (calc_second_order=True)
    # is N * (2D + 2) where D is the number of parameters
    n_simulations = int(np.round(simulations / (2 * len(bounds) + 2)))
    param_values = sobol.sample(problem, n_simulations, seed=random_seed).T
    output = dict()

    for i, param in enumerate(bounds.keys()):
        output[param] = param_values[i]

    return output


def simulate_prospectd_lut(input_dict,
                           wls_sim,
                           srf=None,
                           outfile=None):
    [wls, r, t] = prospect.prospectd_vec(input_dict['N_leaf'],
                                         input_dict['Cab'], input_dict['Car'],
                                         input_dict['Cbrown'], input_dict['Cw'],
                                         input_dict['Cm'], input_dict['Ant'])
    # Convolve the simulated spectra to a gaussian filter per band
    rho_leaf = []
    tau_leaf = []

    if srf:
        if type(srf) == float or type(srf) == int:
            wls_sim = np.asarray(wls_sim)
            # Convolve spectra by full width half maximum
            for wl in wls_sim:
                weight = srf_from_fwhm(wl, srf)
                weight = np.tile(weight, (1, r.shape[2])).T
                rho_leaf.append(
                    float(np.sum(weight * r, axis=0) / np.sum(weight, axis=0)))
                tau_leaf.append(
                    float(np.sum(weight * t, axis=0) / np.sum(weight, axis=0)))

        elif type(srf) == list or type(srf) == tuple:
            for weight in srf:
                weight = np.tile(weight, (1, r.shape[2])).T
                rho_leaf.append(
                    float(np.sum(weight * r, axis=0) / np.sum(weight, axis=0)))
                tau_leaf.append(
                    float(np.sum(weight * t, axis=0) / np.sum(weight, axis=0)))

    else:
        rho_leaf = np.copy(r)
        tau_leaf = np.copy(t)

    rho_leaf = np.asarray(rho_leaf)
    tau_leaf = np.asarray(tau_leaf)

    if outfile:
        fid = open(outfile + '_rho', 'wb')
        pickle.dump(rho_leaf, fid, -1)
        fid.close()
        fid = open(outfile + '_param', 'wb')
        pickle.dump(input_dict, fid, -1)
        fid.close()

    return rho_leaf, input_dict


def simulate_prosail_lut_parallel(n_jobs,
                                  input_dict,
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

    input_dict = pd.DataFrame.from_dict(input_dict)
    simulations = input_dict.shape[0]
    print("Running %i simulations" % simulations)
    if np.isscalar(vza):
        vza = np.full_like(input_dict["LAI"], vza)
    if np.isscalar(sza):
        sza = np.full_like(input_dict["LAI"], sza)
    if np.isscalar(psi):
        psi = np.full_like(input_dict["LAI"], psi)
    # Calculate the total number of multiprocess loops
    tp = mp.Pool(n_jobs)
    subsample_size = np.ceil(simulations / n_jobs)
    jobs = []
    for i in range(n_jobs):
        start = int(i * subsample_size)
        end = int(np.minimum((i + 1) * subsample_size,
                             simulations))

        temp = input_dict.loc[start:end - 1].to_records()
        subsample_dict = {name: temp[name] for name in temp.dtype.names}
        del temp
        jobs.append((i,
                     subsample_dict,
                     wls_sim,
                     rsoil_vec[:, start:end],
                     skyl,
                     sza[start:end],
                     vza[start:end],
                     psi[start:end],
                     srf,
                     calc_FAPAR,
                     reduce_4sail))

    results = tp.starmap(simulate_prosail_lut_worker, jobs)

    tp.close()
    tp.join()

    output_dict = {name: np.empty(simulations) for name in results[0][1][1].keys()}
    if not srf:
        n_bands = len(wls_sim)
    else:
        n_bands = len(srf)
    rho_canopy = np.empty((simulations, n_bands))
    print("Filling output matrix")
    for k, result in results:
        start = int(k * subsample_size)
        end = int(np.minimum((k + 1) * subsample_size,
                             simulations))

        rho_canopy[start:end, :] = result[0]
        for var, array in result[1].items():
            output_dict[var][start:end] = array

    if outfile:
        fid = open(outfile + '_rho', 'wb')
        pickle.dump(rho_canopy, fid, -1)
        fid.close()
        fid = open(outfile + '_param', 'wb')
        pickle.dump(output_dict, fid, -1)
        fid.close()

    return np.array(rho_canopy), output_dict


def simulate_prosail_lut_worker(job,
                                input_dict,
                                wls_sim,
                                rsoil_vec,
                                skyl=0.1,
                                sza=37,
                                vza=0,
                                psi=0,
                                srf=None,
                                calc_FAPAR=False,
                                reduce_4sail=False):
    print("Running job %i" % job)
    rho_canopy, input_dict = simulate_prosail_lut(input_dict,
                                                  wls_sim,
                                                  rsoil_vec,
                                                  skyl=skyl,
                                                  sza=sza,
                                                  vza=vza,
                                                  psi=psi,
                                                  srf=srf,
                                                  outfile=None,
                                                  calc_FAPAR=calc_FAPAR,
                                                  reduce_4sail=reduce_4sail)

    print("Finished job %i" % job)
    return (job, [rho_canopy, input_dict])


def simulate_prosail_lut(input_dict,
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
    print('Starting %i Simulations' % np.size(input_dict['leaf_angle']))

    # Calculate the lidf
    lidf = sail.calc_lidf_campbell_vec(input_dict['leaf_angle'])
    # for i,wl in enumerate(wls_wim):
    [wls, r, t] = prospect.prospectd_vec(input_dict['N_leaf'],
                                         input_dict['Cab'],
                                         input_dict['Car'],
                                         input_dict['Cbrown'],
                                         input_dict['Cw'],
                                         input_dict['Cm'],
                                         input_dict['Ant'])

    r = r.T
    t = t.T

    if type(skyl) == float:
        skyl = np.full(r.shape, skyl)

    # Convolve the simulated spectra to a gaussian filter per band
    rho_leaf = []
    tau_leaf = []
    skyl_rho = []
    rsoil = []
    if srf and reduce_4sail:
        if type(srf) == float or type(srf) == int:

            wls_sim = np.asarray(wls_sim)
            # Convolve spectra by full width half maximum
            for wl in wls_sim:
                weight = srf_from_fwhm(wl, srf)
                weight = np.tile(weight, (r.shape[1], 1)).T
                rho_leaf.append(
                    np.sum(weight * r, axis=0) / np.sum(weight, axis=0))
                tau_leaf.append(
                    np.sum(weight * t, axis=0) / np.sum(weight, axis=0))
                skyl_rho.append(
                    np.sum(weight * skyl, axis=0) / np.sum(weight, axis=0))
                rsoil.append(
                    np.sum(weight * rsoil_vec, axis=0) / np.sum(weight, axis=0))

        elif type(srf) == list or type(srf) == tuple:
            if skyl.shape != r.shape:
                skyl = np.tile(skyl, (r.shape[1], 1)).T
            for weight in srf:
                weight = np.tile(weight, (r.shape[1], 1)).T
                rho_leaf.append(
                    np.sum(weight * r, axis=0) / np.sum(weight, axis=0))
                tau_leaf.append(
                    np.sum(weight * t, axis=0) / np.sum(weight, axis=0))
                skyl_rho.append(
                    np.sum(weight * skyl, axis=0) / np.sum(weight, axis=0))
                rsoil.append(
                    np.sum(weight * rsoil_vec, axis=0) / np.sum(weight, axis=0))
            skyl_rho = np.asarray(skyl_rho)

        if calc_FAPAR:
            par_index = wls <= 700
            rho_leaf_fapar = np.mean(r[par_index, :], axis=0)
            tau_leaf_fapar = np.mean(t[par_index, :], axis=0)
            skyl_rho_fapar = np.mean(skyl[par_index, :], axis=0)
            rsoil_vec_fapar = np.mean(rsoil_vec[par_index, :], axis=0)

    elif reduce_4sail:
        wls_sim = np.asarray(wls_sim)
        for wl in wls_sim:
            rho_leaf.append(r[wls == wl].reshape(-1))
            tau_leaf.append(t[wls == wl].reshape(-1))
            skyl_rho.append(skyl[wls == wl].reshape(-1))
            rsoil.append(rsoil_vec[wls == wl].reshape(-1))
        rho_leaf = np.asarray(rho_leaf)
        tau_leaf = np.asarray(tau_leaf)
        skyl_rho = np.asarray(skyl_rho)
        rsoil = np.asarray(rsoil)
        if calc_FAPAR:
            par_index = wls <= 700
            rho_leaf_fapar = np.mean(r[par_index, :], axis=0)
            tau_leaf_fapar = np.mean(t[par_index, :], axis=0)
            skyl_rho_fapar = np.mean(skyl[par_index, :], axis=0)
            rsoil_vec_fapar = np.mean(rsoil_vec[par_index, :], axis=0)

    else:
        rho_leaf = r
        tau_leaf = t
        if skyl.ndim == 2:
            skyl_rho = np.asarray(skyl.T)
        else:
            skyl_rho = np.repeat(skyl[:, np.newaxis], rho_leaf.shape[1], 1)
        rsoil = np.asarray(rsoil_vec)

    rho_leaf = np.asarray(rho_leaf)
    tau_leaf = np.asarray(tau_leaf)
    skyl_rho = np.asarray(skyl_rho)
    rsoil = np.asarray(rsoil)

    if np.isscalar(vza):
        vza = np.full_like(input_dict['LAI'], vza)
    if np.isscalar(sza):
        sza = np.full_like(input_dict['LAI'], sza)
    if np.isscalar(psi):
        psi = np.full_like(input_dict['LAI'], psi)

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
     _] = sail.foursail_vec(input_dict['LAI'],
                            input_dict['hotspot'],
                            lidf,
                            sza,
                            vza,
                            psi,
                            rho_leaf,
                            tau_leaf,
                            rsoil)

    r2 = rdot * skyl_rho + rsot * (1 - skyl_rho)
    del rdot, rsot

    if calc_FAPAR:
        fAPAR_array, fIPAR_array = calc_fapar_campbell(skyl_rho_fapar,
                                                    input_dict['LAI'],
                                                    input_dict['leaf_angle'],
                                                    np.ones(input_dict[
                                                                'LAI'].shape) * sza,
                                                    rho_leaf_fapar,
                                                    tau_leaf_fapar,
                                                    rsoil_vec_fapar)

        fAPAR_array[np.isfinite(fAPAR_array)] = np.clip(
            fAPAR_array[np.isfinite(fAPAR_array)],
            0,
            1)

        fIPAR_array[np.isfinite(fIPAR_array)] = np.clip(
            fIPAR_array[np.isfinite(fIPAR_array)],
            0,
            1)

        input_dict['fAPAR'] = fAPAR_array
        input_dict['fIPAR'] = fIPAR_array
        del fAPAR_array, fIPAR_array

    rho_canopy = []
    if srf and not reduce_4sail:
        if type(srf) == float or type(srf) == int:
            # Convolve spectra by full width half maximum
            for wl in wls_sim:
                weight = srf_from_fwhm(wl, srf)
                rho_canopy.append(
                    np.sum(weight * r2, axis=0) / np.sum(weight, axis=0))

        elif type(srf) == list or type(srf) == tuple:
            for weight in srf:
                weight = np.tile(weight, (1, r2.shape[2])).T
                rho_canopy.append(
                    np.sum(weight * r2, axis=0) / np.sum(weight, axis=0))
    elif reduce_4sail:
        rho_canopy = np.asarray(r2)

    else:
        for wl in wls_sim:
            rho_canopy.append(r2[wls == wl].reshape(-1))

    rho_canopy = np.asarray(rho_canopy)

    if outfile:
        fid = open(outfile + '_rho', 'wb')
        pickle.dump(rho_canopy, fid, -1)
        fid.close()
        fid = open(outfile + '_param', 'wb')
        pickle.dump(input_dict, fid, -1)
        fid.close()

    return rho_canopy.T, input_dict


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

    # Initialize values
    S_0 = np.zeros(rho_leaf.shape)
    S_1 = np.zeros(rho_leaf.shape)

    # vzas = np.linspace(0, 90, num=36)
    # vaas = np.linspace(0, 360, num=12)
    # step_vza_radians = np.radians(vzas[1] - vzas[0])
    # step_psi_radians = np.radians(vaas[1] - vaas[0])
    # vzas, psis = np.meshgrid(vzas, vaas, indexing="ij")
    # Start the hemispherical integration
    vzas = np.arange(0, 90 - STEP_VZA / 2., STEP_VZA)
    psis = np.arange(0, 360, STEP_PSI)
    vzas, psis = np.meshgrid(vzas, psis, indexing="ij")
    step_vza_radians, step_psi_radians = np.radians(STEP_VZA), np.radians(
        STEP_PSI)

    for vza, psi in zip(vzas.reshape(-1), psis.reshape(-1)):
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
         _] = sail.foursail_vec(LAI,
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
        Ed_up_1 = (rsoil * (Es_1 + tsd * Es + tdd * Ed)) / (1. - rsoil * rdd)
        # Downwelling diffuse shortwave radiation at ground level            
        Ed_down_1 = tsd * Es + tdd * Ed + rdd * Ed_up_1
        # Upwelling shortwave (beam and diffuse) radiation towards the observer (at angle psi/vza)
        Eo_0 = rdot * Ed + rsot * Es
        # Spectral flux at the top of the canopy        
        # & add the top of the canopy flux to the integral and continue through the hemisphere
        S_0 += Eo_0 * cosvza * sinvza * step_vza_radians * step_psi_radians / np.pi
        # Spectral flus at the bottom of the canopy
        # & add the bottom of the canopy flux to the integral and continue through the hemisphere
        S_1 += (Es_1 + tdd * Ed) * cosvza * sinvza * step_vza_radians * step_psi_radians / np.pi
        # Absorbed flux at ground lnevel
        Sn_soil = (1. - rsoil) * (
                    Es_1 + Ed_down_1)  # narrowband net soil shortwave radiation

    # Calculate the vegetation (sw/lw) net radiation (divergence) as residual of top of the canopy and net soil radiation
    Rn_sw_veg_nb = 1.0 - S_0 - Sn_soil  # narrowband net canopy sw radiation
    fAPAR = np.sum(Rn_sw_veg_nb, axis=0) / Rn_sw_veg_nb.shape[
        0]  # broadband net canopy sw radiation
    fIPAR = np.sum(1.0 - S_1, axis=0) / S_1.shape[0]
    return fAPAR, fIPAR


def calc_fapar_campbell(skyl, lai, leaf_angle, sza, rho_leaf, tau_leaf, rsoil):
    x_lad = rad.leafangle_2_chi(leaf_angle)
    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                         sza,
                                                         rho_leaf,
                                                         tau_leaf,
                                                         rsoil,
                                                         x_lad=x_lad)
    fapar = (1.0 - taubt) * (1.0 - albb) * (1. - skyl) + \
            (1.0 - taudt) * (1.0 - albd) * skyl

    akb = rad.calc_K_be_Campbell(sza, x_lad=x_lad)
    taub = np.exp(-akb * lai)
    taud = rad._calc_taud(x_lad, lai)
    fipar = (1.0 - taub) * (1.0 - skyl) + (1.0 - taud) * skyl

    return fapar, fipar

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
    x_LAD = ((alpha / 9.65) ** (1. / -1.65)) - 3.

    return x_LAD


def x_LAD2alpha(x_LAD):
    alpha = 9.65 * (3 + x_LAD) ** -1.65
    alpha = np.degrees(alpha)
    return alpha

def srf_from_fwhm(wl, fwhm):
    wls_full = np.arange(400, 2501)
    sigma = fwhm2sigma(fwhm)
    srf = norm.pdf(wls_full, loc=wl, scale=sigma)
    # Normalize to get 1 at the maximum response
    srf = srf / np.max(srf)
    return srf


def build_soil_database(soil_albedo_factor,
                        soil_library=sail.SOIL_LIBRARY):

    soil_library = Path(soil_library)
    n_simulations = np.size(soil_albedo_factor)
    soil_files = list(soil_library.glob('jhu.*spectrum.txt'))
    n_soils = len(soil_files)
    soil_spectrum = []
    for soil_file in soil_files:
        r = np.genfromtxt(soil_file)
        soil_spectrum.append(r[:, 1])

    multiplier = int(np.ceil(float(n_simulations / n_soils)))
    soil_spectrum = np.asarray(soil_spectrum * multiplier)
    soil_spectrum = soil_spectrum[:n_simulations]
    soil_spectrum = soil_spectrum * soil_albedo_factor.reshape(-1, 1)
    soil_spectrum = np.clip(soil_spectrum, 0, 1)
    soil_spectrum = soil_spectrum.T
    return soil_spectrum

