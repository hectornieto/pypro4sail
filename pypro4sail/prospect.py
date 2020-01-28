# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:32:19 2015

@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Apr 14 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main functions to run the leaf radiative transfer model 
PROSPECT5.

PACKAGE CONTENTS
================
* :func:`prospectd` Runs PROSPECT5 leaf radiative transfer model.
* :func:`prospectd_wl` Runs PROSPECT5 leaf radiative transfer model for a specific wavelenght, aimed for computing speed.

Ancillary functions
-------------------
* :func:`tav` Average transmittivity at the leaf surface. 
* :func:`tav_wl` Average transmittivity at the leaf surface for :func:`Prospect5_wl`.

EXAMPLE
=======
.. code-block:: python

    # Running Prospect5
    import Prospect5
    # Simulate leaf full optical spectrum (400-2500nm) 
    wl, rho_leaf, tau_leaf = Prospect5.Prospect5(1.2, 30., 10., 0.0, 0.015, 0.009)
    
"""

# Extinction coefficients and refractive index
from pypro4sail import spectral_lib
from scipy.special import expi
import numpy as np

wls, refr_index, Cab_k, Car_k, Cbrown_k, Cw_k, Cm_k, Ant_k = spectral_lib

params_prospect = ('N_leaf', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'Ant')


def prospectd(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant):
    '''PROSPECT D Plant leaf reflectance and transmittance modeled
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------    
    Nleaf : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw : float
        equivalent water thickness (g cm-2 or cm).
    Cm : float
        dry matter content (g cm-2).
    Ant : float
        Anthocianins concentration (mug cm-2).

    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    
    Notes
    -----
    Some examples observed during the LOPEX'93 experiment on
    Fresh (F) and dry (D) leaves :
    ---------------------------------------------
                   N     Cab     Cw        Cm
    ---------------------------------------------
    min          1.000    0.0  0.004000  0.001900
    max          3.000  100.0  0.040000  0.016500
    corn (F)     1.518   58.0  0.013100  0.003662
    rice (F)     2.275   23.7  0.007500  0.005811
    clover (F)   1.875   46.7  0.010000  0.003014
    laurel (F)   2.660   74.1  0.019900  0.013520
    ---------------------------------------------
    min          1.500    0.0  0.000063  0.0019
    max          3.600  100.0  0.000900  0.0165
    bamboo (D)   2.698   70.8  0.000117  0.009327
    lettuce (D)  2.107   35.2  0.000244  0.002250
    walnut (D)   2.656   62.8  0.000263  0.006573
    chestnut (D) 1.826   47.7  0.000307  0.004305
    ==============================================================================

    References
    ----------
    .. [Jacquemoud96] Jacquemoud S., Ustin S.L., Verdebout J., Schmuck G., Andreoli G.,
        Hosgood B. (1996), Estimating leaf biochemistry using the PROSPECT
        leaf optical properties model, Remote Sensing of Environment, 56:194-202
        http://dx.doi.org/10.1016/0034-4257(95)00238-3.
    .. [Jacquemoud90] Jacquemoud S., Baret F. (1990), PROSPECT: a model of leaf optical
        properties spectra, Remote Sensing of Environment, 34:75-91
        http://dx.doi.org/10.1016/0034-4257(90)90100-Z.
    .. [Feret08] Feret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
        Properties Model Separating Photosynthetic Pigments, Remote Sensing of
        Environment. http://dx.doi.org/10.1016/j.rse.2008.02.012.
    '''

    l = np.array(wls)
    k = (Cab * np.array(Cab_k) + Car * np.array(Car_k)
         + Cbrown * np.array(Cbrown_k) + Cw * np.array(Cw_k)
         + Cm * np.array(Cm_k) + Ant * np.array(Ant_k)) / Nleaf
    k[k <= 0] = 0

    trans = (1. - k) * np.exp(-k) - k ** 2. * expi(-k)
    # trans=(1.-k)*np.exp(-k)+k**2.*expn(1.,k)
    trans[k <= 0.0] = 1.0

    alpha = 40.
    # reflectance and transmittance of one layer
    rho, tau, Ra, Ta = refl_trans_one_layer(alpha, refr_index, trans)
    # reflectance and transmittance of multiple layers
    rho, tau = reflectance_n_layers_stokes(rho, tau, Ra, Ta, Nleaf)

    return l, rho, tau


def reflectance_n_layers_stokes(r, t, Ra, Ta, Nleaf):
    """reflectance and transmittance of N layers

    References
    ----------
    [STOKES62] Stokes G.G. (1862), On the intensity of the light reflected from
                or transmitted through a pile of plates,
                Proc. Roy. Soc. Lond., 11:545-556.

    """

    D = np.sqrt((1. + r + t) * (1. + r - t) * (1. - r + t) * (1. - r - t))
    a = (1. + r ** 2 - t ** 2 + D) / (2. * r)
    b = (1. - r ** 2 + t ** 2 + D) / (2. * t)

    bNm1 = np.power(b, Nleaf - 1.)
    bN2 = bNm1 ** 2
    a2 = a ** 2
    denom = a2 * bN2 - 1.
    Rsub = a * (bN2 - 1.) / denom
    Tsub = bNm1 * (a2 - 1.) / denom

    # Case of zero absorption
    j = r + t >= 1.
    Tsub[j] = t[j] / (t[j] + (1. - t[j]) * (Nleaf - 1.))
    Rsub[j] = 1. - Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom = 1. - Rsub * r
    tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom

    return refl, tran


def reflectance_n_layers_stokes_vec(r, t, Ra, Ta, Nleaf):
    """reflectance and transmittance of N layers

    References
    ----------
    .. [Stokes1862] Stokes G.G. (1862), On the intensity of the light reflected from
                or transmitted through a pile of plates,
                Proc. Roy. Soc. Lond., 11:545-556.

    """

    D = np.sqrt((1. + r + t) * (1. + r - t) * (1. - r + t) * (1. - r - t))
    a = (1. + r ** 2 - t ** 2 + D) / (2. * r)
    b = (1. - r ** 2 + t ** 2 + D) / (2. * t)

    bNm1 = np.power(b, Nleaf - 1.)
    bN2 = bNm1 ** 2
    a2 = a ** 2
    denom = a2 * bN2 - 1.
    Rsub = a * (bN2 - 1.) / denom
    Tsub = bNm1 * (a2 - 1.) / denom

    # Case of zero absorption
    j = r + t >= 1.
    Tsub[j] = t[j] / (t[j] + (1. - t[j]) * (np.repeat(Nleaf, r.shape[1], axis=1)[j] - 1.))
    Rsub[j] = 1. - Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom = 1. - Rsub * r
    tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom

    return refl, tran


def prospectd_vec(Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant):
    """PROSPECT 5 Plant leaf reflectance and transmittance modeled
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------
    Nleaf : 1D array
        leaf structure parameter.
    Cab : 1D array
        chlorophyll a+b content (mug cm-2).
    Car : 1D array
        carotenoids content (mug cm-2).
    Cbrown : 1D array
        brown pigments concentration (unitless).
    Cw : 1D array
        equivalent water thickness (g cm-2 or cm).
    Cm : 1D array
        dry matter content (g cm-2).

    Returns
    -------
    l : 1D array
        wavelenght (nm).
    rho : 2D array
        leaf reflectance .
    tau : 2D array
        leaf transmittance .

    Notes
    -----
    Some examples observed during the LOPEX'93 experiment on
    Fresh (F) and dry (D) leaves :
    ---------------------------------------------
                   N     Cab     Cw        Cm
    ---------------------------------------------
    min          1.000    0.0  0.004000  0.001900
    max          3.000  100.0  0.040000  0.016500
    corn (F)     1.518   58.0  0.013100  0.003662
    rice (F)     2.275   23.7  0.007500  0.005811
    clover (F)   1.875   46.7  0.010000  0.003014
    laurel (F)   2.660   74.1  0.019900  0.013520
    ---------------------------------------------
    min          1.500    0.0  0.000063  0.0019
    max          3.600  100.0  0.000900  0.0165
    bamboo (D)   2.698   70.8  0.000117  0.009327
    lettuce (D)  2.107   35.2  0.000244  0.002250
    walnut (D)   2.656   62.8  0.000263  0.006573
    chestnut (D) 1.826   47.7  0.000307  0.004305
    ---------------------------------------------

    References
    ----------
    .. [Jacquemoud96] Jacquemoud S., Ustin S.L., Verdebout J., Schmuck G., Andreoli G.,
        Hosgood B. (1996), Estimating leaf biochemistry using the PROSPECT
        leaf optical properties model, Remote Sensing of Environment, 56:194-202
        http://dx.doi.org/10.1016/0034-4257(95)00238-3.
    .. [Jacquemoud90] Jacquemoud S., Baret F. (1990), PROSPECT: a model of leaf optical
        properties spectra, Remote Sensing of Environment, 34:75-91
        http://dx.doi.org/10.1016/0034-4257(90)90100-Z.
    .. [Feret08] Feret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
        Properties Model Separating Photosynthetic Pigments, Remote Sensing of
        Environment. http://dx.doi.org/10.1016/j.rse.2008.02.012.
    """
    # Vectorize the inputs
    Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant = (Nleaf[:, np.newaxis],
                                            Cab[:, np.newaxis],
                                            Car[:, np.newaxis],
                                            Cbrown[:, np.newaxis],
                                            Cw[:, np.newaxis],
                                            Cm[:, np.newaxis],
                                            Ant[:, np.newaxis])

    l = np.array(wls)
    k = (Cab * np.array(Cab_k) + Car * np.array(Car_k)
         + Cbrown * np.array(Cbrown_k) + Cw * np.array(Cw_k)
         + Cm * np.array(Cm_k) + Ant * np.array(Ant_k)) / Nleaf

    k[k <= 0] = 0

    trans = (1. - k) * np.exp(-k) + k ** 2. * (-expi(-k))
    trans[k <= 0.0] = 1.0
    trans[k > 85] = 0
    # trans = trans_approx(k)
    del k

    alpha = 40.
    # reflectance and transmittance of one layer
    rho, tau, Ra, Ta = refl_trans_one_layer(alpha, refr_index, trans)
    # reflectance and transmittance of multiple layers
    rho, tau = reflectance_n_layers_stokes_vec(rho, tau, Ra, Ta, Nleaf)
    return l, rho, tau


def prospectd_wl(wl, Nleaf, Cab, Car, Cbrown, Cw, Cm, Ant):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from a single given wavelenght. Aimed for computation speed.

    Parameters
    ----------    
    wl : float
        wavelenght (nm) to simulate.
    Nleaf : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw : float
        equivalent water thickness (g cm-2 or cm).
    Cm : float
        dry matter content (g cm-2).

    Returns
    -------
    l : float
        wavelenght (nm).
    rho : float
        leaf reflectance .
    tau : float
        leaf transmittance. 
    
    Notes
    -----
    Some examples observed during the LOPEX'93 experiment on
    Fresh (F) and dry (D) leaves :
    ---------------------------------------------
                   N     Cab     Cw        Cm
    ---------------------------------------------
    min          1.000    0.0  0.004000  0.001900
    max          3.000  100.0  0.040000  0.016500
    corn (F)     1.518   58.0  0.013100  0.003662
    rice (F)     2.275   23.7  0.007500  0.005811
    clover (F)   1.875   46.7  0.010000  0.003014
    laurel (F)   2.660   74.1  0.019900  0.013520
    ---------------------------------------------
    min          1.500    0.0  0.000063  0.0019
    max          3.600  100.0  0.000900  0.0165
    bamboo (D)   2.698   70.8  0.000117  0.009327
    lettuce (D)  2.107   35.2  0.000244  0.002250
    walnut (D)   2.656   62.8  0.000263  0.006573
    chestnut (D) 1.826   47.7  0.000307  0.004305
    ---------------------------------------------

    References
    ----------
    .. [Jacquemoud96] Jacquemoud S., Ustin S.L., Verdebout J., Schmuck G., Andreoli G.,
        Hosgood B. (1996), Estimating leaf biochemistry using the PROSPECT
        leaf optical properties model, Remote Sensing of Environment, 56:194-202
        http://dx.doi.org/10.1016/0034-4257(95)00238-3.
    .. [Jacquemoud90] Jacquemoud S., Baret F. (1990), PROSPECT: a model of leaf optical
        properties spectra, Remote Sensing of Environment, 34:75-91
        http://dx.doi.org/10.1016/0034-4257(90)90100-Z.
    .. [Feret08] Feret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
        Properties Model Separating Photosynthetic Pigments, Remote Sensing of
        Environment. http://dx.doi.org/10.1016/j.rse.2008.02.012.
    '''

    wl_index = wls == wl
    Cab_abs = float(Cab_k[wl_index])
    Car_abs = float(Car_k[wl_index])
    Cbrown_abs = float(Cbrown_k[wl_index])
    Cw_abs = float(Cw_k[wl_index])
    Cm_abs = float(Cm_k[wl_index])
    Ant_abs = float(Ant_k[wl_index])
    n = float(refr_index[wl_index])

    k = (Cab * Cab_abs + Car * Car_abs + Cbrown * Cbrown_abs
         + Cw * Cw_abs + Cm * Cm_abs + Ant * Ant_abs) / Nleaf
    if k <= 0.:
        trans = 1.0
    else:
        trans = (1. - k) * np.exp(-k) + (k ** 2) * (-expi(-k))
        if trans < 0.0:
            trans = 0.0

    alpha = 40.
    # reflectance and transmittance of one layer
    rho, tau, Ra, Ta = refl_trans_one_layer(alpha, n, trans)
    # reflectance and transmittance of multiple layers
    rho, tau = reflectance_n_layers_stokes(rho, tau, Ra, Ta, Nleaf)

    return wl, rho, tau


def refl_trans_one_layer(alpha, nr, tau):
    """ reflectance and transmittance of one layer
    References
    ----------
    ..[Allen69] Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
            Interaction of isotropic ligth with a compact plant leaf,
            J. Opt. Soc. Am., 59(10):1376-1379.
    """
    # reflectivity and transmissivity at the interface
    talf = tav(alpha, nr)
    ralf = 1.0 - talf
    t12 = tav(90., nr)
    r12 = 1. - t12
    t21 = t12 / nr ** 2
    r21 = 1 - t21

    # top surface side
    denom = 1. - r21 ** 2 * tau ** 2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    # bottom surface side
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    return r, t, Ra, Ta


def tav(theta, ref):
    """
    Average transmittivity at the leaf surface within a given solid angle.

    Parameters
    ----------
    theta : float
        incidence solid angle (radian). The average angle that works in most cases is 40degrees.
    ref : array_like
        Refaction index.

    Returns
    -------
    f : array_like
        Average transmittivity at the leaf surface.

    References
    ----------
    .. [Stern64] Stern F. (1964), Transmission of isotropic radiation across an
        interface between two dielectrics,
        Applied Optics, 3(1):111-113,
        http://dx.doi.org/10.1364/AO.3.000111.
    .. [Allen1973] Allen W.A. (1973), Transmission of isotropic light across a
        dielectric surface in two and three dimensions,
        Journal of the Optical Society of America, 63(6):664-666.
        http://dx.doi.org/10.1364/JOSA.63.000664.
    """

    theta = np.radians(theta)
    r2 = ref ** 2.0
    rp = r2 + 1.0
    rm = r2 - 1.0
    a = ((ref + 1.0) ** 2.0) / 2.0
    k = -(r2 - 1.0) ** 2.0 / 4.0
    ds = np.sin(theta)
    if theta == 0.0:
        f = 4.0 * ref / (ref + 1.0) ** 2.0
        return f
    else:
        if theta == np.pi / 2.0:
            b1 = np.zeros(ref.shape)
        else:
            b1 = np.sqrt((ds ** 2.0 - rp / 2.0) ** 2.0 + k)
    k2 = k ** 2.0
    rm2 = rm ** 2.0
    b2 = ds ** 2.0 - rp / 2.0
    b = b1 - b2
    ts = (k2 / (6.0 * b ** 3.0) + k / b - b / 2.0) - (k2 / (6.0 * a ** 3.0) + k / a - a / 2.0)
    tp1 = -2.0 * r2 * (b - a) / (rp ** 2.0)
    tp2 = -2.0 * r2 * rp * np.log(b / a) / rm2
    tp3 = r2 * (b ** -1.0 - a ** -1.0) / 2.0
    tp4 = 16.0 * r2 ** 2.0 * (r2 ** 2.0 + 1.0) * np.log((2.0 * rp * b - rm2) / (2.0 * rp * a - rm2)) / (rp ** 3.0 * rm2)
    tp5 = 16.0 * r2 ** 3.0 * ((2.0 * rp * b - rm2) ** -1.0 - (2.0 * rp * a - rm2) ** -1.0) / rp ** 3.0
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    f = (ts + tp) / (2.0 * ds ** 2.0)

    return f


def tav_wl(theta, ref):
    """ Average transmittivity at the leaf surface within a given solid angle.

    Parameters
    ----------
    theta : float
        incidence solid angle (radian). The average angle that works in most cases is 40degrees.
    ref : float
        Refaction index.

    Returns
    -------
    f : float
        Average transmittivity at the leaf surface.

    References
    ----------
    .. [Stern64] Stern F. (1964), Transmission of isotropic radiation across an
        interface between two dielectrics,
        Applied Optics, 3(1):111-113,
        http://dx.doi.org/10.1364/AO.3.000111.
    .. [Allen1973] Allen W.A. (1973), Transmission of isotropic light across a
        dielectric surface in two and three dimensions,
        Journal of the Optical Society of America, 63(6):664-666.
        http://dx.doi.org/10.1364/JOSA.63.000664.
    """

    theta = np.radians(theta)
    r2 = ref ** 2.0
    rp = r2 + 1.0
    rm = r2 - 1.0
    a = ((ref + 1.0) ** 2.0) / 2.0
    k = -(r2 - 1.0) ** 2.0 / 4.0
    ds = np.sin(theta)
    if theta == 0.0:
        f = 4.0 * ref / (ref + 1.0) ** 2.0
        return f
    elif theta == np.pi / 2.0:
        b1 = 0.0
    else:
        b1 = np.sqrt((ds ** 2.0 - rp / 2.0) ** 2.0 + k)
    k2 = k ** 2.0
    rm2 = rm ** 2.0
    b2 = ds ** 2.0 - rp / 2.0
    b = b1 - b2
    ts = (k2 / (6.0 * b ** 3.0) + k / b - b / 2.0) - (k2 / (6.0 * a ** 3.0) + k / a - a / 2.0)
    tp1 = -2.0 * r2 * (b - a) / (rp ** 2.0)
    tp2 = -2.0 * r2 * rp * np.log(b / a) / rm2
    tp3 = r2 * (b ** -1.0 - a ** -1.0) / 2.0
    tp4 = 16.0 * r2 ** 2.0 * (r2 ** 2.0 + 1.0) * np.log((2.0 * rp * b - rm2) / (2.0 * rp * a - rm2)) / (rp ** 3.0 * rm2)
    tp5 = 16.0 * r2 ** 3.0 * ((2.0 * rp * b - rm2) ** -1.0 - (2.0 * rp * a - rm2) ** -1.0) / rp ** 3.0
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    f = (ts + tp) / (2.0 * ds ** 2.0)

    return f

def trans_approx(k):
    trans = np.zeros(k.shape)
    i = k <= 0
    trans[i] = 1
    i = np.logical_and(k > 0.0, k <= 4.0)
    xx = 0.5 * k[i] - 1.0
    yy = (((((((((((((((-3.60311230482612224e-13
        * xx + 3.46348526554087424e-12) * xx-2.99627399604128973e-11)
        * xx + 2.57747807106988589e-10) * xx-2.09330568435488303e-9)
        * xx + 1.59501329936987818e-8) * xx-1.13717900285428895e-7)
        * xx + 7.55292885309152956e-7) * xx-4.64980751480619431e-6)
        * xx + 2.63830365675408129e-5) * xx-1.37089870978830576e-4)
        * xx + 6.47686503728103400e-4) * xx-2.76060141343627983e-3)
        * xx + 1.05306034687449505e-2) * xx-3.57191348753631956e-2)
        * xx + 1.07774527938978692e-1) * xx-2.96997075145080963e-1
    yy = (yy * xx + 8.64664716763387311e-1) * xx + 7.42047691268006429e-1
    yy = yy - np.log(k[i])
    trans[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i]**2 * yy

    i = np.logical_and(k > 4.0, k <= 85.0)
    xx = 14.5 / (k[i] + 3.25) - 1.0
    yy = (((((((((((((((-1.62806570868460749e-12
        * xx - 8.95400579318284288e-13) * xx - 4.08352702838151578e-12)
        * xx - 1.45132988248537498e-11) * xx - 8.35086918940757852e-11)
        * xx - 2.13638678953766289e-10) * xx - 1.10302431467069770e-9)
        * xx - 3.67128915633455484e-9) * xx - 1.66980544304104726e-8)
        * xx - 6.11774386401295125e-8) * xx - 2.70306163610271497e-7)
        * xx - 1.05565006992891261e-6) * xx - 4.72090467203711484e-6)
        * xx - 1.95076375089955937e-5) * xx - 9.16450482931221453e-5)
        * xx - 4.05892130452128677e-4) * xx - 2.14213055000334718e-3
    yy = ((yy * xx - 1.06374875116569657e-2)
             * xx - 8.50699154984571871e-2) * xx + 9.23755307807784058e-1
    yy = np.exp(-k[i]) * yy / k[i]

    trans[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i]**2 * yy

    i = k > 85.0
    trans[i] = 0
    return trans
