import numpy as np
import scipy.interpolate as interp
from pathlib import Path
import pandas as pd
from scipy.stats import norm

SRF_FOLDER = Path.home() / "codes" / "pypro4sail" / "pypro4sail" / "spectra" / "sensor_response_functions"


def interpolate_srf(out_wl, in_wl, srf, normalize='max'):
    ''' Intepolates an Spectral Response Function to a given set of wavelenghts
    Parameters
    ----------
    out_wl : numpy array
        Wavelenths at which the spectral response function will be interpolated
    in_wl : numpy array
        Wavelenths for the input spectral response function
    srf : 2D numpy array
        Input Spectral response function
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
    f = interp.interp1d(in_wl, srf, bounds_error=False,
                        fill_value='extrapolate')
    # Interpolate and normalize to max=1
    fltr = f(out_wl)
    fltr[out_wl < in_wl.min()] = 0
    fltr[out_wl > in_wl.max()] = 0
    if normalize == 'max':
        fltr = fltr / np.max(fltr)
    elif normalize == 'sum':
        fltr = fltr / np.sum(fltr)
    elif normalize == 'none':
        pass
    else:
        print('Wrong normalize keyword, use "max" or "sum"')
    return fltr



def fwhm_srf(center_wl, fwhm, out_wl,
             min_signal=1e-6,
             normalize='max'):
    ''' Intepolates an Spectral Response Function to a given set of wavelenghts
    Parameters
    ----------
    center_wl : float
        Central wavelength (nm) or the input spectral response function
    fwhm : float
        Full-witdh-half-maximum (nm) for the input spectral response function
    out_wl : numpy array
        Wavelenths at which the spectral response function will be interpolated
    normalize : str
        Method used to normalize the output response signal: 'max', 'sum', 'none'

    Returns
    -------
    fltr : float or numpy array
        Integrated filter
    '''
    def fwhm2sigma(fwhm):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return sigma

    sigma = fwhm2sigma(fwhm)
    dist = norm(loc=center_wl, scale=sigma)
    fltr = dist.pdf(out_wl)
    low_signal = fltr < min_signal
    fltr[low_signal]= 0
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
    """
    # Landsat 8 & 9
    orig_bands = ["CoastalAerosol", "Blue", "Green", "Red",
                  "NIR", "SWIR1", "SWIR2"]
    dest_bands = ["band1", "band2", "band3", "band4", "band5", "band6", "band7"]

    orig_bands = ["Blue-L5 TM", "Green-L5 TM", "Red-L5 TM",
                  "NIR-L5 TM", "SWIR(5)-L5 TM", "SWIR(7)-L5 TM"]
    dest_bands = ["band1", "band2", "band3", "band4", "band5", "band7"]

    out_srf_file = SRF_FOLDER / "Landsat5.txt"
    srf_file = Path.home() / "Downloads" / "L5_TM_RSR.xlsx"

    headers = ["WL_SR"] + dest_bands
    df = pd.DataFrame(columns=headers)
    out_wl = np.arange(400, 2501)
    df["WL_SR"] = out_wl

    for band, dest in zip(orig_bands, dest_bands):
        srf = pd.read_excel(srf_file, band, header=None, skiprows=1)
        out_srf = interpolate_srf(out_wl, srf[0].values, srf[1].values, normalize='max')
        df[dest] = out_srf

    df.to_csv(out_srf_file, sep="\t", index=False)
    """

    out_srf_file = SRF_FOLDER / "Micasense_Dual.txt"
    bands = {"B01" : (475, 32),
             "B02" : (560, 27),
             "B03" : (668, 14),
             "B04" : (842, 57),
             "B05" : (717, 12),
             "B06" : (444, 28),
             "B07" : (531, 14),
             "B08" : (650, 16),
             "B09" : (705, 10),
             "B10" : (740, 18),
             }
    headers = ["WL_SR"] + list(bands.keys())
    df = pd.DataFrame(columns=headers)
    out_wl = np.arange(400, 2501)
    df["WL_SR"] = out_wl
    for band, (wl, fwhm) in bands.items():
        out_srf = fwhm_srf(wl, fwhm, out_wl, normalize='max')
        valid = out_srf >= 0.5
        est_fwhm = max(out_wl[valid]) - min(out_wl[valid])
        print(f"{wl}: {fwhm} vs. {est_fwhm}")
        df[band] = out_srf

    df.to_csv(out_srf_file, sep="\t", index=False)




