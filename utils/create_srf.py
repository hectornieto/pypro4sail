import numpy as np
import scipy.interpolate as interp
import os
import pandas as pd

SRF_FOLDER = os.path.join(os.getcwd(), "pypro4sail", "spectra", "sensor_response_functions")

def interpolate_srf(out_wl, srf, normalize='max'):
    ''' Intepolates an Spectral Response Function to a given set of wavelenghts
    Parameters
    ----------
    out_wl : numpy array
        Wavelenths at which the spectral response function will be interpolated
    srf : 2D numpy array
        1rst elements is wavelength and second element is the corresponding contribution to the signal
    normalize : str
        Method used to normalize the output response signal: 'max', 'sum', 'none'

    Returns
    -------
    fltr : float or numpy array
        Integrated filter
    '''

    out_wl = np.asarray(out_wl)
    srf = np.maximum(0, np.asarray(srf))
    # Create the linear interpolation object
    f = interp.interp1d(srf[0], srf[1], bounds_error=False,
                        fill_value='extrapolate')
    # Interpolate and normalize to max=1
    fltr = f(out_wl)
    if normalize == 'max':
        fltr = fltr / np.max(fltr)
    elif normalize == 'sum':
        fltr = fltr / np.sum(fltr)
    elif normalize == 'none':
        pass
    else:
        print('Wrong normalize keyword, use "max" or "sum"')
    return fltr

if __name__ == "__main__":

    orig_bands = ["landsat7_etm_b1", "landsat7_etm_b2", "landsat7_etm_b3",
                  "landsat7_etm_b4", "landsat7_etm_b5", "landsat7_etm_b7"]
    dest_bands = ["band1", "band2", "band3", "band4", "band5", "band7"]

    out_srf_file = os.path.join(SRF_FOLDER, "Landsat7.txt")
    orig_srf_folder = os.path.join(os.path.expanduser("~"),
                                   "libradtran",
                                   "data",
                                   "filter",
                                   "landsat"
                                   )

    headers = ["WL_SR"] + dest_bands
    df = pd.DataFrame(columns=headers)
    out_wl = np.arange(400, 2501)
    df["WL_SR"] = out_wl

    for band, dest in zip(orig_bands, dest_bands):
        srf_file = os.path.join(orig_srf_folder, band)
        srf = np.genfromtxt(srf_file, delimiter=" ", comments="#")
        out_srf = interpolate_srf(out_wl, srf.T, normalize='max')
        df[dest] = out_srf

    df.to_csv(out_srf_file, sep="\t", index=False)


