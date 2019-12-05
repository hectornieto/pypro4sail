import pypro4sail.ann_inversion as inv
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from agrotig_input_creator import sentinel_biophysical as senet
import os.path as pth
import glob

n_simulations = 20000
white_noise = 0
param_bounds = senet.senet.prosail_bounds
distribution = senet.senet.prosail_distribution
moments = senet.senet.prosail_moments
param_bounds["LAI"] = (0, 7)
param_bounds["Cbrown"] = (0, 3)
param_bounds["Cm"] = (0.01, 0.03)
moments["LAI"] = (1.5, 1)
out_test_dir = pth.expanduser("~")

ann_opts = {'hidden_layer_sizes': (100),
            'activation': 'logistic',
            'learning_rate_init': 0.01,
            'momentum': 0.90,
            'verbose': True,
            'max_iter': 10000}

rf_opts = {"n_estimators": 100,
           "min_samples_leaf": 1,
           "n_jobs": -1}
reg_method = "random_forest"


reduce_4sail = True,
calc_FAPAR = True,

ObjParamNames = ('Cab',
                 'Cbrown',
                 'Cw',
                 "Cm",
                 'Ant',
                 'LAI',
                 'leaf_angle',
                 'fAPAR',
                 'fIPAR')

for param in ObjParamNames:
    distribution[param] = 1
distribution["LAI"] = 1

def plot_scatter(lai_obs, Cab_obs, Cm_obs, fapar_obs, fipar_obs,
                 lai_pre, Cab_pre, Cm_pre, fapar_pre, fipar_pre,
                 out_template, calc_fg, uncertainty):
    def calc_uncertainty(pred, obs, n_bins=100):
        valid = np.logical_and.reduce(
            (pred >= 0, obs >= 0, np.isfinite(pred), np.isfinite(obs)))
        pred = pred[valid]
        obs = obs[valid]
        values = []
        rmses = []
        bins, bin_edges = np.histogram(pred, bins=n_bins)
        for i in range(len(bins)):
            cases = np.logical_and(pred >= bin_edges[i],
                                   pred <= bin_edges[i + 1])
            value = np.mean(pred[cases])
            rmse = np.sqrt(
                np.sum((pred[cases] - obs[cases]) ** 2) / np.size(obs[cases]))

            values.append(value)
            rmses.append(rmse)

        values = np.asarray(values)
        rmses = np.asarray(rmses)

        valid = np.logical_and.reduce(
            (values >= 0, np.isfinite(values), np.isfinite(rmses)))
        values_1 = np.array([values[valid], values[valid] ** 2]).T
        reg = LinearRegression().fit(values_1, rmses[valid])

        return values, rmses, reg

    obs = 1e5 * lai_obs * Cm_obs
    pre = 1e5 * lai_pre * Cm_pre
    plt.scatter(obs,
                pre,
                color='black',
                s=5,
                alpha=0.1,
                marker='.')
    plt.title('Leaf Biomass (kg/ha)')
    absline = np.asarray([[0, np.amax(np.array([obs, pre]))],
                          [0, np.amax(np.array([obs, pre]))]])

    plt.xlim(0, np.amax(np.array([obs, pre])))
    plt.ylim(0, np.amax(np.array([obs, pre])))

    plt.plot(absline[0], absline[1], 'k-')
    values, rmses, reg = calc_uncertainty(pre, obs)
    uncertainty["biomass"] = reg

    values_fit = np.linspace(0, np.amax(pre), 1000)
    plt.plot(values, rmses, "r:")
    plt.plot(values_fit, reg.predict(np.array([values_fit, values_fit**2]).T), "r-")
    plt.savefig(out_template%'biomass')
    plt.close()

    obs = 0.1 * lai_obs * Cab_obs
    pre = 0.1 * lai_pre * Cab_pre
    plt.scatter(obs,
                pre,
                color='black',
                s=5,
                alpha=0.1,
                marker='.')
    plt.title('Canopy Chlorophyll Content (kg/ha)')
    absline = np.asarray([[0, np.amax(np.array([obs, pre]))],
                          [0, np.amax(np.array([obs, pre]))]])

    plt.xlim(0, np.amax(np.array([obs, pre])))
    plt.ylim(0, np.amax(np.array([obs, pre])))

    plt.plot(absline[0], absline[1], 'k-')
    values, rmses, reg = calc_uncertainty(pre, obs)
    uncertainty["CCC"] = reg
    values_fit = np.linspace(0, np.amax(pre), 1000)
    plt.plot(values, rmses, "r:")
    plt.plot(values_fit, reg.predict(np.array([values_fit, values_fit**2]).T), "r-")
    plt.savefig(out_template%'CCC')

    plt.close()


    if calc_fg:
        obs = fapar_obs / fipar_obs
        pre = fapar_pre / fipar_pre
        valid = np.logical_and(np.isfinite(obs), np.isfinite(pre))
        obs = np.clip(obs[valid], 0, 1)
        pre = np.clip(pre[valid], 0, 1)
        plt.scatter(obs,
                    pre,
                    color='black',
                    s=5,
                    alpha=0.1,
                    marker='.')
        plt.title('$f_g$')
        absline = np.asarray([[0, np.amax(np.array([obs, pre]))],
                              [0, np.amax(np.array([obs, pre]))]])

        plt.xlim(0, np.amax(np.array([obs, pre])))
        plt.ylim(0, np.amax(np.array([obs, pre])))

        plt.plot(absline[0], absline[1], 'k-')
        values, rmses, reg = calc_uncertainty(pre, obs)
        uncertainty["F_G"] = reg
        values_fit = np.linspace(0, 1, 1000)
        plt.plot(values, rmses, "r:")
        plt.plot(values_fit, reg.predict(np.array([values_fit, values_fit ** 2]).T), "r-")
        plt.savefig(out_template % 'fg')

        plt.close()

    return uncertainty

params = inv.build_prosail_database(n_simulations,
                        param_bounds=param_bounds,
                        distribution=distribution,
                        moments=moments)

wls_sim = np.arange(400, 2501)
print('Builing standard soil database')
soil_files = glob.glob(pth.join(senet.senet.SOIL_LIBRARY, 'jhu.*spectrum.txt'))
n_soils = len(soil_files)
soil_spectrum = []
for soil_file in soil_files:
    r = np.genfromtxt(soil_file)
    soil_spectrum.append(r[:, 1])

multiplier = int(np.ceil(float(n_simulations/n_soils)))
soil_spectrum = np.asarray(soil_spectrum * multiplier)
soil_spectrum = soil_spectrum[:n_simulations]
soil_spectrum = soil_spectrum * params['bs'].reshape(-1, 1)
soil_spectrum = np.clip(soil_spectrum, 0, 1)
soil_spectrum = soil_spectrum.T


srf = []
srf_file = pth.join(senet.senet.SRF_LIBRARY, 'Sentinel2A.txt')
srfs = np.genfromtxt(srf_file, dtype=None, names=True)
for band in senet.senet.S2_BANDS:
    srf.append(srfs[band])

rho_canopy_vec, params = inv.simulate_prosail_lut(params,
                                                  wls_sim,
                                                  soil_spectrum,
                                                  skyl=np.full(wls_sim.shape, 0.2),
                                                  sza=37.5,
                                                  vza=0,
                                                  psi=180,
                                                  srf=srf,
                                                  outfile=None,
                                                  calc_FAPAR=calc_FAPAR,
                                                  reduce_4sail=reduce_4sail)

# Test ANN with testing simulated database
calc_fg = False

split = np.random.uniform(0, 1, n_simulations)

split = split > 0.40
input_scalers = {}
output_scalers = {}
ANNs = {}
uncertainty = {}
for reg_method in ["neural_network", "random_forest"]:
    if reg_method == "neural_network":
        reg_opts = ann_opts
    elif reg_method == "random_forest":
        reg_opts = rf_opts
    for i, param in enumerate(ObjParamNames):

        ObjParamName = [param]

        if param != 'LAI' and param != 'fAPAR' and param != 'fIPAR' and param != 'Cab' and param != 'Cm':
            training = np.logical_and(split, params['LAI'] > 0)
            testing = np.logical_and(~split, params['LAI'] > 0)

        else:
            training = np.asarray(split)
            testing = ~training


        ANN, input_gauss_scaler, output_gauss_scaler, _ =\
            inv.train_ann(rho_canopy_vec[training],
                          params[param][training].reshape(-1, 1),
                          scaling_input='normalize',
                          scaling_output='normalize',
                          regressor_opts=reg_opts,
                          reg_method=reg_method)

        input_scalers[param] = input_gauss_scaler
        output_scalers[param] = output_gauss_scaler
        ANNs[param] = ANN
        if param == 'fAPAR':
            fapar_obs = params[param][testing]
            fapar_pre = output_gauss_scaler.inverse_transform(
                                    ANN.predict(
                                        input_gauss_scaler.transform(
                                            rho_canopy_vec[testing])))

            calc_fg = True
        elif param == 'fIPAR':
            fipar_obs = params[param][testing]
            fipar_pre = output_gauss_scaler.inverse_transform(
                                    ANN.predict(
                                        input_gauss_scaler.transform(
                                            rho_canopy_vec[testing])))

        elif param == 'LAI':
            lai_obs = params[param][testing]
            lai_pre = output_gauss_scaler.inverse_transform(
                                    ANN.predict(
                                        input_gauss_scaler.transform(
                                            rho_canopy_vec[testing])))

        elif param == 'Cab':
            Cab_obs = params[param][testing]
            Cab_pre = output_gauss_scaler.inverse_transform(
                                    ANN.predict(
                                        input_gauss_scaler.transform(
                                            rho_canopy_vec[testing])))

        elif param == 'Cm':
            Cm_obs = params[param][testing]
            Cm_pre = output_gauss_scaler.inverse_transform(
                                    ANN.predict(
                                        input_gauss_scaler.transform(
                                            rho_canopy_vec[testing])))

        uncertainty[param] = senet.test_ann(rho_canopy_vec[testing],
                                       params[param][testing],
                                       ANN,
                                       scaling_input=input_gauss_scaler,
                                       scaling_output=output_gauss_scaler,
                                       outfile=pth.join(out_test_dir,
                                                        '%s_%s.png' % (param,
                                                                      reg_method)
                                                        ),
                                       param_names=ObjParamName)


    out_template = pth.join(out_test_dir, '%s_%s.png' %("%s", reg_method))
    # Plot scatter plots of predicted versus observed values if wanted
    uncertainty = plot_scatter(lai_obs, Cab_obs, Cm_obs, fapar_obs, fipar_obs,
                        lai_pre, Cab_pre, Cm_pre, fapar_pre, fipar_pre,
                        out_template, calc_fg, uncertainty)

