# -*- coding: utf-8 -*-
"""
Created on Mon Mar 2 11:32:19 2015

@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Apr 14 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main functions to run the canopy radiative transfer model 
4SAIL.

PACKAGE CONTENTS
================
* :func:`foursail` Runs 4SAIL canopy radiative transfer model.
* :func:`foursail_wl` Runs 4SAIL canopy radiative transfer model for a specific wavelenght, aimed for computing speed.

Ancillary functions
-------------------
* :func:`calc_lidf_verhoef` Calculate the Leaf Inclination Distribution Function based on the [Verhoef1998] bimodal LIDF distribution.
* :func:`calc_lidf_campbell` Calculate the Leaf Inclination Distribution Function based on the [Campbell1990] ellipsoidal LIDF distribution.
* :func:`volscatt` Colume scattering functions and interception coefficients.
* :func:`jfunc1` J1 function with avoidance of singularity problem.
* :func:`jfunc1_wl` J1 function with avoidance of singularity problem for :func:`FourSAIL_wl`.
* :func:`jfunc2` J2 function with avoidance of singularity problem.
* :func:`jfunc2_wl` J2 function with avoidance of singularity problem for :func:`FourSAIL_wl`.
* :func;`get_sun_angles` Calculate the Sun Zenith and Azimuth Angles.

EXAMPLE
=======
.. code-block:: python

    # Running the coupled Prospect and 4SAIL
    import Prospect5, FourSAIL
    # Simulate leaf full optical spectrum (400-2500nm) 
    wl, rho_leaf, tau_leaf = Prospect5.Prospect5(Nleaf, Cab, Car, Cbrown, Cw, Cm)
    # Estimate the Leaf Inclination Distribution Function of a canopy
    lidf = CalcLIDF_Campbell(alpha)
    # Simulate leaf reflectance and transmittance factors of a canopy 
    tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammasowl = FourSAIL.FourSAIL(lai,hotspot,lidf,SZA,VZA,PSI,rho_leaf,tau_leaf,rho_soil)
    # Simulate the canopy reflectance factor for a given difuse/total radiation condition (skyl)
    rho_canopy = rdot*skyl+rsot*(1-skyl)
    
"""
import numpy as np
from pathlib import Path
import Py6S as sixs

params_sail = ('LAI', 'hotspot', 'leaf_angle')
params_prosail = ('N_leaf',
                  'Cab',
                  'Car',
                  'Cbrown',
                  'Cw',
                  'Cm',
                  'Ant',
                  'LAI',
                  'hotspot',
                  'leaf_angle')

SOIL_LIBRARY = Path(__file__).parent / "spectra" / "soil_spectral_library"
SRF_LIBRARY = Path(__file__).parent / "spectra" / "sensor_response_functions"


def calc_lidf_verhoef(a, b, n_elements=18):
    """Calculate the Leaf Inclination Distribution Function based on the
    Verhoef's bimodal LIDF distribution.

    Parameters
    ----------
    a : float
        controls the average leaf slope.
    b : float
        controls the distribution's bimodality.

            * LIDF type     [a,b].
            * Planophile    [1,0].
            * Erectophile   [-1,0].
            * Plagiophile   [0,-1].
            * Extremophile  [0,1].
            * Spherical     [-0.35,-0.15].
            * Uniform       [0,0].
            * requirement: |LIDFa| + |LIDFb| < 1.
    n_elements : int
        Total number of equally spaced inclination angles.

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function at equally spaced angles.

    References
    ----------
    .. [Verhoef1998] Verhoef, Wout. Theory of radiative transfer models applied
        in optical remote sensing of vegetation canopies.
        Nationaal Lucht en Ruimtevaartlaboratorium, 1998.
        http://library.wur.nl/WebQuery/clc/945481.
        """

    freq = 1.0
    step = 90.0 / n_elements
    lidf = []
    angles = [i * step for i in reversed(range(n_elements))]
    for angle in angles:
        tl1 = np.radians(angle)
        if a > 1.0:
            f = 1.0 - np.cos(tl1)
        else:
            eps = 1e-8
            delx = 1.0
            x = 2.0 * tl1
            p = float(x)
            while delx >= eps:
                y = a * np.sin(x) + .5 * b * np.sin(2. * x)
                dx = .5 * (y - x + p)
                x = x + dx
                delx = abs(dx)
            f = (2. * y + p) / np.pi
        freq = freq - f
        lidf.append(freq)
        freq = float(f)
    lidf = list(reversed(lidf))
    return lidf


def calc_lidf_campbell(alpha, n_elements=18):
    """Calculate the Leaf Inclination Distribution Function based on the
    mean angle of [Campbell1990] ellipsoidal LIDF distribution.

    Parameters
    ----------
    alpha : float
        Mean leaf angle (degrees) use 57 for a spherical LIDF.
    n_elements : int
        Total number of equally spaced inclination angles .

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function for 18 equally spaced angles.

    References
    ----------
    .. [Campbell1986] G.S. Campbell, Extinction coefficients for radiation in
        plant canopies calculated using an ellipsoidal inclination angle distribution,
        Agricultural and Forest Meteorology, Volume 36, Issue 4, 1986, Pages 317-321,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(86)90010-9.
    .. [Campbell1990] G.S Campbell, Derivation of an angle density function for
        canopies with ellipsoidal leaf angle distributions,
        Agricultural and Forest Meteorology, Volume 49, Issue 3, 1990, Pages 173-176,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(90)90030-A.
    """

    alpha = float(alpha)
    excent = np.exp(-1.6184e-5 * alpha ** 3. + 2.1145e-3 * alpha ** 2. - 1.2390e-1 * alpha + 3.2491)
    sum0 = 0.
    freq = []
    step = 90.0 / n_elements
    for i in range(n_elements):
        tl1 = np.radians(i * step)
        tl2 = np.radians((i + 1.) * step)
        x1 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl1) ** 2.))
        x2 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl2) ** 2.))
        if excent == 1.:
            freq.append(abs(np.cos(tl1) - np.cos(tl2)))
        else:
            alph = excent / np.sqrt(abs(1. - excent ** 2.))
            alph2 = alph ** 2.
            x12 = x1 ** 2.
            x22 = x2 ** 2.
            if excent > 1.:
                alpx1 = np.sqrt(alph2 + x12)
                alpx2 = np.sqrt(alph2 + x22)
                dum = x1 * alpx1 + alph2 * np.log(x1 + alpx1)
                freq.append(abs(dum - (x2 * alpx2 + alph2 * np.log(x2 + alpx2))))
            else:
                almx1 = np.sqrt(alph2 - x12)
                almx2 = np.sqrt(alph2 - x22)
                dum = x1 * almx1 + alph2 * np.arcsin(x1 / alph)
                freq.append(abs(dum - (x2 * almx2 + alph2 * np.arcsin(x2 / alph))))
    sum0 = sum(freq)
    lidf = []
    for i in range(n_elements):
        lidf.append(float(freq[i]) / sum0)

    return lidf


def calc_lidf_campbell_vec(alpha, n_elements=18):
    """Calculate the Leaf Inclination Distribution Function based on the
    mean angle of [Campbell1990] ellipsoidal LIDF distribution.

    Parameters
    ----------
    alpha : float
        Mean leaf angle (degrees) use 57 for a spherical LIDF.
    n_elements : int
        Total number of equally spaced inclination angles .

    Returns
    -------
    lidf : list
        Leaf Inclination Distribution Function for 18 equally spaced angles.

    References
    ----------
    .. [Campbell1986] G.S. Campbell, Extinction coefficients for radiation in
        plant canopies calculated using an ellipsoidal inclination angle distribution,
        Agricultural and Forest Meteorology, Volume 36, Issue 4, 1986, Pages 317-321,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(86)90010-9.
    .. [Campbell1990] G.S Campbell, Derivation of an angle density function for
        canopies with ellipsoidal leaf angle distributions,
        Agricultural and Forest Meteorology, Volume 49, Issue 3, 1990, Pages 173-176,
        ISSN 0168-1923, http://dx.doi.org/10.1016/0168-1923(90)90030-A.
    """

    alpha = np.asarray(alpha).reshape(-1)
    excent = np.exp(-1.6184e-5 * alpha ** 3. + 2.1145e-3 * alpha ** 2. - 1.2390e-1 * alpha + 3.2491)
    freq = np.zeros((n_elements, alpha.shape[0]))
    step = 90.0 / n_elements
    for i in range(n_elements):
        tl1 = np.radians(i * step)
        tl2 = np.radians((i + 1.) * step)
        index = excent == 1.
        freq[i, index] = np.abs(np.cos(tl1) - np.cos(tl2))
        x1 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl1) ** 2.))
        x2 = excent / (np.sqrt(1. + excent ** 2. * np.tan(tl2) ** 2.))
        alph = excent / np.sqrt(np.abs(1. - excent ** 2.))
        alph2 = alph ** 2.
        x12 = x1 ** 2.
        x22 = x2 ** 2.
        alpx1 = np.sqrt(alph2 + x12)
        alpx2 = np.sqrt(alph2 + x22)
        index = excent > 1.
        dum = x1[index] * alpx1[index] + alph2[index] * np.log(x1[index] + alpx1[index])
        freq[i, index] = np.abs(dum - (x2[index] * alpx2[index] + alph2[index] * np.log(x2[index] + alpx2[index])))
        index = excent < 1.
        almx1 = np.sqrt(alph2[index] - x12[index])
        almx2 = np.sqrt(alph2[index] - x22[index])
        dum = x1[index] * almx1 + alph2[index] * np.arcsin(x1[index] / alph[index])
        freq[i, index] = np.abs(dum - (x2[index] * almx2 + alph2[index] * np.arcsin(x2[index] / alph[index])))
    sum0 = np.sum(freq, axis=0)
    lidf = freq / sum0

    return lidf


def foursail(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    """ Runs 4SAIL canopy radiative transfer model.

    Parameters
    ----------
    lai : float
        Leaf Area Index.
    hotspot : float
        Hotspot parameter.
    lidf : list
        Leaf Inclination Distribution at regular angle steps.
    tts : float
        Sun Zenith Angle (degrees).
    tto : float
        View(sensor) Zenith Angle (degrees).
    psi : float
        Relative Sensor-Sun Azimuth Angle (degrees).
    rho : array_like
        leaf lambertian reflectance.
    tau : array_like
        leaf transmittance.
    rsoil : array_like
        soil lambertian reflectance.

    Returns
    -------
    tss : array_like
        beam transmittance in the sun-target path.
    too : array_like
        beam transmittance in the target-view path.
    tsstoo : array_like
        beam tranmittance in the sur-target-view path.
    rdd : array_like
        canopy bihemisperical reflectance factor.
    tdd : array_like
        canopy bihemishperical transmittance factor.
    rsd : array_like
        canopy directional-hemispherical reflectance factor.
    tsd : array_like
        canopy directional-hemispherical transmittance factor.
    rdo : array_like
        canopy hemispherical-directional reflectance factor.
    tdo : array_like
        canopy hemispherical-directional transmittance factor.
    rso : array_like
        canopy bidirectional reflectance factor.
    rsos : array_like
        single scattering contribution to rso.
    rsod : array_like
        multiple scattering contribution to rso.
    rddt : array_like
        surface bihemispherical reflectance factor.
    rsdt : array_like
        surface directional-hemispherical reflectance factor.
    rdot : array_like
        surface hemispherical-directional reflectance factor.
    rsodt : array_like
        reflectance factor.
    rsost : array_like
        reflectance factor.
    rsot : array_like
        surface bidirectional reflectance factor.
    gammasdf : array_like
        'Thermal gamma factor'.
    gammasdb : array_like
        'Thermal gamma factor'.
    gammaso : array_like
        'Thermal gamma factor'.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """

    # weighted_sum_over_lidf
    ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)

    # Geometric factors to be used later with rho and tau
    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1. + bf)
    ddf = 0.5 * (1. - bf)
    # Here rho and tau come in
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    if np.size(sigf) > 1:
        sigf[sigf == 0.0] = 1e-36
        sigb[sigb == 0.0] = 1e-36
    else:
        sigf = max(1e-36, sigf)
        sigb = max(1e-36, sigb)
    att = 1. - sigf
    m = np.sqrt(att ** 2. - sigb ** 2.)
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    # Here the LAI comes in
    if lai <= 0:
        tss = 1
        too = 1
        tsstoo = 1
        rdd = 0
        tdd = 1
        rsd = 0
        tsd = 0
        rdo = 0
        tdo = 0
        rso = 0
        rsos = 0
        rsod = 0
        rddt = rsoil
        rsdt = rsoil
        rdot = rsoil
        rsodt = 0
        rsost = rsoil
        rsot = rsoil
        gammasdf = 0
        gammaso = 0
        gammasdb = 0

        return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
                rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]

    e1 = np.exp(-m * lai)
    e2 = e1 ** 2.
    rinf = (att - m) / sigb
    rinf2 = rinf ** 2.
    re = rinf * e1
    denom = 1. - rinf2 * e2
    J1ks = jfunc1(ks, m, lai)
    J2ks = jfunc2(ks, m, lai)
    J1ko = jfunc1(ko, m, lai)
    J2ko = jfunc2(ko, m, lai)
    Pss = (sf + sb * rinf) * J1ks
    Qss = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko
    tdd = (1. - rinf2) * e1 / denom
    rdd = rinf * (1. - e2) / denom
    tsd = (Pss - re * Qss) / denom
    rsd = (Qss - re * Pss) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    # Thermal "sd" quantities
    gammasdf = (1. + rinf) * (J1ks - re * J2ks) / denom
    gammasdb = (1. + rinf) * (-re * J1ks + J2ks) / denom
    tss = np.exp(-ks * lai)
    too = np.exp(-ko * lai)
    z = jfunc2(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qss + tdo * Pss) * rinf
    # Multiple scattering contribution to bidirectional canopy reflectance
    rsod = (T1 + T2 - T3) / (1. - rinf2)
    # Thermal "sod" quantity
    T4 = Tv1 * (1. + rinf)
    T5 = Tv2 * (1. + rinf)
    T6 = (rdo * J2ks + tdo * J1ks) * (1. + rinf) * rinf
    gammasod = (T4 + T5 - T6) / (1. - rinf2)

    dso = define_geometric_constant(tts, tto, psi)
    tsstoo, sumint = hotspot_calculations(hotspot, lai, ko, ks, dso, tss)

    # Bidirectional reflectance
    # Single scattering contribution
    rsos = w * lai * sumint
    gammasos = ko * lai * sumint
    # Total canopy contribution
    rso = rsos + rsod
    gammaso = gammasos + gammasod
    # Interaction with the soil
    dn = 1. - rsoil * rdd
    if np.size(dn) > 1:
        dn[dn < 1e-36] = 1e-36
    else:
        dn = max(1e-36, dn)
    rddt = rdd + tdd * rsoil * tdd / dn
    rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
    rdot = rdo + tdd * rsoil * (tdo + too) / dn
    rsodt = ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
    rsost = rso + tsstoo * rsoil
    rsot = rsost + rsodt

    return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
            rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]


def foursail_vec(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    """ Runs 4SAIL canopy radiative transfer model.

    Parameters
    ----------
    lai : ndarray
        Leaf Area Index.
    hotspot : ndarray
        Hotspot parameter.
    lidf : ndarray
        Leaf Inclination Distribution at regular angle steps.
    tts : ndarray
        Sun Zenith Angle (degrees).
    tto : ndarray
        View(sensor) Zenith Angle (degrees).
    psi : ndarray
        Relative Sensor-Sun Azimuth Angle (degrees).
    rho : ndarray
        leaf lambertian reflectance.
    tau : ndarray
        leaf transmittance.
    rsoil : ndarray
        soil lambertian reflectance.

    Returns
    -------
    tss : ndarray
        beam transmittance in the sun-target path.
    too : ndarray
        beam transmittance in the target-view path.
    tsstoo : ndarray
        beam tranmittance in the sur-target-view path.
    rdd : ndarray
        canopy bihemisperical reflectance factor.
    tdd : ndarray
        canopy bihemishperical transmittance factor.
    rsd : ndarray
        canopy directional-hemispherical reflectance factor.
    tsd : ndarray
        canopy directional-hemispherical transmittance factor.
    rdo : ndarray
        canopy hemispherical-directional reflectance factor.
    tdo : ndarray
        canopy hemispherical-directional transmittance factor.
    rso : ndarray
        canopy bidirectional reflectance factor.
    rsos : ndarray
        single scattering contribution to rso.
    rsod : ndarray
        multiple scattering contribution to rso.
    rddt : ndarray
        surface bihemispherical reflectance factor.
    rsdt : ndarray
        surface directional-hemispherical reflectance factor.
    rdot : ndarray
        surface hemispherical-directional reflectance factor.
    rsodt : ndarray
        reflectance factor.
    rsost : ndarray
        reflectance factor.
    rsot : ndarray
        surface bidirectional reflectance factor.
    gammasdf : ndarray
        'Thermal gamma factor'.
    gammasdb : ndarray
        'Thermal gamma factor'.
    gammaso : ndarray
        'Thermal gamma factor'.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """

    # weighted_sum_over_lidf
    ks, ko, bf, sob, sof = weighted_sum_over_lidf_vec(lidf, tts, tto, psi)

    # Geometric factors to be used later with rho and tau
    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1. + bf)
    ddf = 0.5 * (1. - bf)

    # Here rho and tau come in
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sigf = np.maximum(1e-36, sigf)
    sigb = np.maximum(1e-36, sigb)
    att = 1. - sigf
    m = np.sqrt(att ** 2. - sigb ** 2.)
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    del sdb, sdf, dob, dof, ddb, ddf

    e1 = np.exp(-lai * m)
    e2 = e1 ** 2.
    rinf = (att - m) / sigb
    rinf2 = rinf ** 2.
    re = rinf * e1
    denom = 1. - rinf2 * e2
    J1ks = jfunc1_vec(ks, m, lai)
    J2ks = jfunc2(ks, m, lai)
    J1ko = jfunc1_vec(ko, m, lai)
    J2ko = jfunc2(ko, m, lai)
    Pss = (sf + sb * rinf) * J1ks
    Qss = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko

    tdd = (1. - rinf2) * e1 / denom
    rdd = rinf * (1. - e2) / denom
    tsd = (Pss - re * Qss) / denom
    rsd = (Qss - re * Pss) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    del e1, e2, Qv, Pv, att, sigb, sigf, J2ko

    # Thermal "sd" quantities
    gammasdf = (1. + rinf) * (J1ks - re * J2ks) / denom
    gammasdb = (1. + rinf) * (-re * J1ks + J2ks) / denom

    del denom, re

    tss = np.exp(-ks * lai)
    too = np.exp(-ko * lai)
    z = jfunc2(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qss + tdo * Pss) * rinf

    del vb, vf, Pss, Qss, J1ko, g1, g2, m, sb, sf, z

    # Multiple scattering contribution to bidirectional canopy reflectance
    rsod = (T1 + T2 - T3) / (1. - rinf2)
    # Thermal "sod" quantity
    T4 = Tv1 * (1. + rinf)
    T5 = Tv2 * (1. + rinf)
    T6 = (rdo * J2ks + tdo * J1ks) * (1. + rinf) * rinf
    gammasod = (T4 + T5 - T6) / (1. - rinf2)

    del Tv1, Tv2, T1, T2, T3, T4, T5, T6, rinf2, rinf, J2ks, J1ks

    # Hotspot effect
    dso = define_geometric_constant(tts, tto, psi)
    tsstoo, sumint = hotspot_calculations_vec(hotspot, lai, ko, ks, dso, tss)  # Bidirectional reflectance

    # Single scattering contribution
    rsos = w * lai * sumint
    gammasos = ko * lai * sumint

    del ko, ks, sumint, w, dso, tts, tto, psi, lai

    # Total canopy contribution
    rso = rsos + rsod
    gammaso = gammasos + gammasod

    del gammasod, gammasos

    # Interaction with the soil
    dn = 1. - rsoil * rdd
    dn = np.maximum(1e-36, dn)
    rddt = rdd + tdd * rsoil * tdd / dn
    rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
    rdot = rdo + tdd * rsoil * (tdo + too) / dn
    rsodt = ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
    rsost = rso + tsstoo * rsoil
    rsot = rsost + rsodt

    return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
            rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]


def fousail_wl(lai, hotspot, lidf, tts, tto, psi, rho, tau, rsoil):
    """Runs 4SAIL canopy radiative transfer model for a single wavelenght.

    Parameters
    ----------
    lai : float
        Leaf Area Index.
    hotspot : float
        Hotspot parameter.
    lidf : list
        Leaf Inclination Distribution at regular angle steps.
    tts : float
        Sun Zenith Angle (degrees).
    tto : float
        View(sensor) Zenith Angle (degrees).
    psi : float
        Relative Sensor-Sun Azimuth Angle (degrees).
    rho : float
        leaf lambertian reflectance.
    tau : float
        leaf transmittance.
    rsoil : float
        soil lambertian reflectance.

    Returns
    -------
    tss : float
        beam transmittance in the sun-target path.
    too : float
        beam transmittance in the target-view path.
    tsstoo : float
        beam tranmittance in the sur-target-view path.
    rdd : float
        canopy bihemisperical reflectance factor.
    tdd : float
        canopy bihemishperical transmittance factor.
    rsd : float
        canopy directional-hemispherical reflectance factor.
    tsd : float
        canopy directional-hemispherical transmittance factor.
    rdo : float
        canopy hemispherical-directional reflectance factor.
    tdo : float
        canopy hemispherical-directional transmittance factor.
    rso : float
        canopy bidirectional reflectance factor.
    rsos : float
        single scattering contribution to rso.
    rsod : float
        multiple scattering contribution to rso.
    rddt : float
        surface bihemispherical reflectance factor.
    rsdt : float
        surface directional-hemispherical reflectance factor.
    rdot : float
        surface hemispherical-directional reflectance factor.
    rsodt : float
        reflectance factor.
    rsost : float
        reflectance factor.
    rsot : float
        surface bidirectional reflectance factor.
    gammasdf : float
        'Thermal gamma factor'.
    gammasdb : float
        'Thermal gamma factor'.
    gammaso : float
        'Thermal gamma factor'.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """

    # Weighted sums over LIDF
    ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, tts, tto, psi)

    # Geometric factors to be used later with rho and tau
    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1. + bf)
    ddf = 0.5 * (1. - bf)
    # Here rho and tau come in
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    att = 1. - sigf
    if att ** 2 - sigb ** 2 > 0:
        m = np.sqrt(att ** 2 - sigb ** 2)
    else:
        m = 0.0
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau
    # Here the LAI comes in
    if lai <= 0:
        tss = 1
        too = 1
        tsstoo = 1
        rdd = 0
        tdd = 1
        rsd = 0
        tsd = 0
        rdo = 0
        tdo = 0
        rso = 0
        rsos = 0
        rsod = 0
        rddt = rsoil
        rsdt = rsoil
        rdot = rsoil
        rsodt = 0
        rsost = rsoil
        rsot = rsoil
        gammasdf = 0
        gammaso = 0
        gammasdb = 0

        return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
                rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]

    e1 = np.exp(-m * lai)
    e2 = e1 ** 2.
    if abs(sigb) >= 1e-36:
        rinf = (att - m) / sigb
    else:
        rinf = 1e36

    rinf2 = rinf ** 2.
    re = rinf * e1
    denom = 1. - rinf2 * e2
    if denom < 1e-36:
        denom = 1e-36

    J1ks = jfunc1_wl(ks, m, lai)
    J2ks = jfunc2_wl(ks, m, lai)
    J1ko = jfunc1_wl(ko, m, lai)
    J2ko = jfunc2_wl(ko, m, lai)
    Pss = (sf + sb * rinf) * J1ks
    Qss = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko
    tdd = (1. - rinf2) * e1 / denom
    rdd = rinf * (1. - e2) / denom
    tsd = (Pss - re * Qss) / denom
    rsd = (Qss - re * Pss) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    # Thermal "sd" quantities
    gammasdf = (1. + rinf) * (J1ks - re * J2ks) / denom
    gammasdb = (1. + rinf) * (-re * J1ks + J2ks) / denom
    tss = np.exp(-ks * lai)
    too = np.exp(-ko * lai)
    z = jfunc2_wl(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qss + tdo * Pss) * rinf

    # Multiple scattering contribution to bidirectional canopy reflectance
    if rinf2 >= 1:
        rinf2 = 1 - 1e-16
    rsod = (T1 + T2 - T3) / (1. - rinf2)

    # Thermal "sod" quantity
    T4 = Tv1 * (1. + rinf)
    T5 = Tv2 * (1. + rinf)
    T6 = (rdo * J2ks + tdo * J1ks) * (1. + rinf) * rinf
    gammasod = (T4 + T5 - T6) / (1. - rinf2)

    # Hotspot effect
    dso = define_geometric_constant(tts, tto, psi)
    tsstoo, sumint = hotspot_calculations(hotspot, lai, ko, ks, dso, tss)  # Bidirectional reflectance

    # Bidirectional reflectance
    # Single scattering contribution
    rsos = w * lai * sumint
    gammasos = ko * lai * sumint
    # Total canopy contribution
    rso = rsos + rsod
    gammaso = gammasos + gammasod
    # Interaction with the soil
    dn = 1. - rsoil * rdd
    if dn == 0.0:
        dn = 1e-36
    rddt = rdd + tdd * rsoil * tdd / dn
    rsdt = rsd + (tsd + tss) * rsoil * tdd / dn
    rdot = rdo + tdd * rsoil * (tdo + too) / dn
    rsodt = ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
    rsost = rso + tsstoo * rsoil
    rsot = rsost + rsodt

    return [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo,
            rso, rsos, rsod, rddt, rsdt, rdot, rsodt, rsost, rsot, gammasdf, gammasdb, gammaso]


def volscatt(tts, tto, psi, ttl):
    """Compute volume scattering functions and interception coefficients
    for given solar zenith, viewing zenith, azimuth and leaf inclination angle.

    Parameters
    ----------
    tts : float
        Solar Zenith Angle (degrees).
    tto : float
        View Zenight Angle (degrees).
    psi : float
        View-Sun reliative azimuth angle (degrees).
    ttl : float
        leaf inclination angle (degrees).

    Returns
    -------
    chi_s : float
        Interception function  in the solar path.
    chi_o : float
        Interception function  in the view path.
    frho : float
        Function to be multiplied by leaf reflectance to obtain the volume scattering.
    ftau : float
        Function to be multiplied by leaf transmittance to obtain the volume scattering.

    References
    ----------
    Wout Verhoef, april 2001, for CROMA.
    """

    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    sts = np.sin(np.radians(tts))
    sto = np.sin(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    psir = np.radians(psi)
    cttl = np.cos(np.radians(ttl))
    sttl = np.sin(np.radians(ttl))
    cs = cttl * cts
    co = cttl * cto
    ss = sttl * sts
    so = sttl * sto
    cosbts = 5.
    if abs(ss) > 1e-6: cosbts = -cs / ss
    cosbto = 5.
    if abs(so) > 1e-6: cosbto = -co / so
    if abs(cosbts) < 1.0:
        bts = np.arccos(cosbts)
        ds = ss
    else:
        bts = np.pi
        ds = cs
    chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)
    if abs(cosbto) < 1.0:
        bto = np.arccos(cosbto)
        do_ = so
    else:
        if tto < 90.:
            bto = np.pi
            do_ = co
        else:
            bto = 0.0
            do_ = -co
    chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
    btran1 = abs(bts - bto)
    btran2 = np.pi - abs(bts + bto - np.pi)
    if psir <= btran1:
        bt1 = psir
        bt2 = btran1
        bt3 = btran2
    else:
        bt1 = btran1
        if psir <= btran2:
            bt2 = psir
            bt3 = btran2
        else:
            bt2 = btran2
            bt3 = psir
    t1 = 2. * cs * co + ss * so * cospsi
    t2 = 0.
    if bt2 > 0.: t2 = np.sin(bt2) * (2. * ds * do_ + ss * so * np.cos(bt1) * np.cos(bt3))
    denom = 2. * np.pi ** 2
    frho = ((np.pi - bt2) * t1 + t2) / denom
    ftau = (-bt2 * t1 + t2) / denom
    if frho < 0.: frho = 0.
    if ftau < 0.: ftau = 0.

    return [chi_s, chi_o, frho, ftau]


def volscatt_vec(tts, tto, psi, ttl):
    """Compute volume scattering functions and interception coefficients
    for given solar zenith, viewing zenith, azimuth and leaf inclination angle.

    Parameters
    ----------
    tts : float
        Solar Zenith Angle (degrees).
    tto : float
        View Zenight Angle (degrees).
    psi : float
        View-Sun reliative azimuth angle (degrees).
    ttl : float
        leaf inclination angle (degrees).

    Returns
    -------
    chi_s : float
        Interception function  in the solar path.
    chi_o : float
        Interception function  in the view path.
    frho : float
        Function to be multiplied by leaf reflectance to obtain the volume scattering.
    ftau : float
        Function to be multiplied by leaf transmittance to obtain the volume scattering.

    References
    ----------
    Wout Verhoef, april 2001, for CROMA.
    """

    tts, tto, psi = map(np.asarray, (tts, tto, psi))
    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    sts = np.sin(np.radians(tts))
    sto = np.sin(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    psir = np.radians(psi)
    cttl = np.cos(np.radians(ttl))
    sttl = np.sin(np.radians(ttl))
    cs = cttl * cts
    co = cttl * cto
    ss = sttl * sts
    so = sttl * sto
    cosbts = np.ones(cs.shape) * 5.
    cosbto = np.ones(co.shape) * 5.
    cosbts[np.abs(ss) > 1e-6] = -cs[np.abs(ss) > 1e-6] / ss[np.abs(ss) > 1e-6]
    cosbto[np.abs(so) > 1e-6] = -co[np.abs(so) > 1e-6] / so[np.abs(so) > 1e-6]

    bts = np.ones(cosbts.shape) * np.pi
    ds = np.array(cs)
    bts[np.abs(cosbts) < 1.0] = np.arccos(cosbts[np.abs(cosbts) < 1.0])
    ds[np.abs(cosbts) < 1.0] = ss[np.abs(cosbts) < 1.0]
    chi_s = 2. / np.pi * ((bts - np.pi * 0.5) * cs + np.sin(bts) * ss)

    bto = np.zeros(cosbto.shape)
    do_ = np.zeros(cosbto.shape)
    bto[np.abs(cosbto) < 1.0] = np.arccos(cosbto[np.abs(cosbto) < 1.0])
    do_[np.abs(cosbto) < 1.0] = so[np.abs(cosbto) < 1.0]
    bto[np.logical_and(np.abs(cosbto) > 1.0, tto < 90.)] = np.pi
    do_[np.logical_and(np.abs(cosbto) > 1.0, tto < 90.)] = co[np.logical_and(np.abs(cosbto) > 1.0, tto < 90.)]
    bto[np.logical_and(np.abs(cosbto) > 1.0, tto > 90.)] = 0
    do_[np.logical_and(np.abs(cosbto) > 1.0, tto > 90.)] = -co[np.logical_and(np.abs(cosbto) > 1.0, tto > 90.)]

    chi_o = 2.0 / np.pi * ((bto - np.pi * 0.5) * co + np.sin(bto) * so)
    btran1 = np.abs(bts - bto)
    btran2 = np.pi - np.abs(bts + bto - np.pi)
    bt1 = np.array(psir)
    bt2 = np.array(btran1)
    bt3 = np.array(btran2)
    bt1[psir > btran1] = btran1[psir > btran1]
    bt2[np.logical_and(psir > btran1, psir <= btran2)] = psir[np.logical_and(psir > btran1, psir <= btran2)]
    bt3[np.logical_and(psir > btran1, psir <= btran2)] = btran2[np.logical_and(psir > btran1, psir <= btran2)]
    bt2[np.logical_and(psir > btran1, psir > btran2)] = btran2[np.logical_and(psir > btran1, psir > btran2)]
    bt3[np.logical_and(psir > btran1, psir > btran2)] = psir[np.logical_and(psir > btran1, psir > btran2)]

    t1 = 2. * cs * co + ss * so * cospsi
    t2 = np.zeros(t1.shape)
    t2[bt2 > 0.] = np.sin(bt2[bt2 > 0.]) * (
                2. * ds[bt2 > 0.] * do_[bt2 > 0.] + ss[bt2 > 0.] * so[bt2 > 0.] * np.cos(bt1[bt2 > 0.]) * np.cos(
            bt3[bt2 > 0.]))
    denom = 2. * np.pi ** 2
    frho = ((np.pi - bt2) * t1 + t2) / denom
    ftau = (-bt2 * t1 + t2) / denom
    frho[frho < 0.] = 0.
    ftau[ftau < 0.] = 0.

    return [chi_s, chi_o, frho, ftau]


def weighted_sum_over_lidf(lidf, tts, tto, psi):
    ks = 0.
    ko = 0.
    bf = 0.
    sob = 0.
    sof = 0.
    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    ctscto = cts * cto

    n_angles = len(lidf)
    angle_step = float(90.0 / n_angles)
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)

    for i, ili in enumerate(litab):
        ttl = 1. * ili
        cttl = np.cos(np.radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s, chi_o, frho, ftau] = volscatt(tts, tto, psi, ttl)
        # Extinction coefficients
        ksli = chi_s / cts
        koli = chi_o / cto
        # Area scattering coefficient fractions
        sobli = frho * np.pi / ctscto
        sofli = ftau * np.pi / ctscto
        bfli = cttl ** 2.
        ks += ksli * float(lidf[i])
        ko += koli * float(lidf[i])
        bf += bfli * float(lidf[i])
        sob += sobli * float(lidf[i])
        sof += sofli * float(lidf[i])

    return ks, ko, bf, sob, sof


def weighted_sum_over_lidf_vec(lidf, tts, tto, psi):
    ks = 0
    ko = 0.
    bf = 0.
    sob = 0.
    sof = 0.
    cts = np.cos(np.radians(tts))
    cto = np.cos(np.radians(tto))
    ctscto = cts * cto

    n_angles = len(lidf)
    angle_step = 90.0 / n_angles
    litab = np.arange(n_angles) * angle_step + (angle_step * 0.5)

    for i, ili in enumerate(litab):
        ttl = 1. * ili
        cttl = np.cos(np.radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s, chi_o, frho, ftau] = volscatt_vec(tts, tto, psi, ttl)
        # Extinction coefficients
        ksli = chi_s / cts
        koli = chi_o / cto
        # Area scattering coefficient fractions
        sobli = frho * np.pi / ctscto
        sofli = ftau * np.pi / ctscto
        bfli = cttl ** 2.
        ks += ksli * lidf[i]
        ko += koli * lidf[i]
        bf += bfli * lidf[i]
        sob += sobli * lidf[i]
        sof += sofli * lidf[i]

    return ks, ko, bf, sob, sof


def jfunc1(k, l, t):
    """ J1 function with avoidance of singularity problem."""

    nb = np.size(l)
    del_ = (k - l) * t
    if nb > 1:
        result = np.zeros(nb)
        result[abs(del_) > 1e-3] = (np.exp(-l[abs(del_) > 1e-3] * t) - np.exp(-k * t)) / (k - l[abs(del_) > 1e-3])
        result[abs(del_) <= 1e-3] = 0.5 * t * (np.exp(-k * t) + np.exp(-l[abs(del_) <= 1e-3] * t)) * (
                    1. - (del_[abs(del_) <= 1e-3] ** 2.) / 12.)
    else:
        if abs(del_) > 1e-3:
            result = (np.exp(-l * t) - np.exp(-k * t)) / (k - l)
        else:
            result = 0.5 * t * (np.exp(-k * t) + np.exp(-l * t)) * (1. - (del_ ** 2.) / 12.)
    return result


def jfunc1_vec(k, l, t):
    """ J1 function with avoidance of singularity problem."""

    del_ = (k - l) * t
    t = np.repeat(t[np.newaxis, :], del_.shape[0], 0)
    k = np.repeat(k[np.newaxis, :], del_.shape[0], 0)
    result = np.zeros(del_.shape)
    index = np.abs(del_) > 1e-3
    result[index] = (np.exp(-l[index] * t[index]) - np.exp(-k[index] * t[index])) / (k[index] - l[index])
    result[~index] = 0.5 * t[~index] * (np.exp(-k[~index] * t[~index]) + np.exp(-l[~index] * t[~index])) * (
                1. - (del_[~index] ** 2.) / 12.)
    return result


def jfunc2(k, l, t):
    """J2 function."""

    return (1. - np.exp(-(k + l) * t)) / (k + l)


def jfunc1_wl(k, l, t):
    """J1 function with avoidance of singularity problem."""

    del_ = (k - l) * t
    if abs(del_) > 1e-3:
        result = (np.exp(-l * t) - np.exp(-k * t)) / (k - l)
    else:
        result = 0.5 * t * (np.exp(-k * t) + np.exp(-l * t)) * (1. - (del_ ** 2.) / 12.)
    return result


def jfunc2_wl(k, l, t):
    """J2 function."""

    return (1. - np.exp(-(k + l) * t)) / (k + l)


def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    """Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float
        latitude of the site (degrees).
    long : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    sza : float
        Sun Zenith Angle (degrees).
    saa : float
        Sun Azimuth Angle (degrees).

    """

    lat, lon, stdlon, doy, ftime = map(
        np.asarray, (lat, lon, stdlon, doy, ftime))
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
          3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.asarray((solar_time - 12.0) * 15.)
    # Get solar elevation angle
    sin_thetha = np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat)) + \
                 np.sin(declination) * np.sin(np.radians(lat))
    sun_elev = np.arcsin(sin_thetha)
    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))
    # Get solar azimuth angle
    cos_phi = np.asarray(
        (np.sin(declination) * np.cos(np.radians(lat)) -
         np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))) /
        np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return np.asarray(sza), np.asarray(saa)


def hotspot_calculations(hotspot, lai, ko, ks, dso, tss):
    # Treatment of the hotspot-effect
    alf = 1e36

    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0.:
        alf = (dso / hotspot) * 2. / (ks + ko)
    if alf == 0.:
        # The pure hotspot
        tsstoo = tss
        sumint = (1. - tss) / (ks * lai)
    else:
        # Outside the hotspot
        alf = (dso / hotspot) * 2. / (ks + ko)
        fhot = lai * np.sqrt(ko * ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1 = 0.
        y1 = 0.
        f1 = 1.
        fint = (1. - np.exp(-alf)) * .05
        sumint = 0.
        for istep in range(1, 21):
            if istep < 20:
                x2 = -np.log(1. - istep * fint) / alf
            else:
                x2 = 1.
            y2 = -(ko + ks) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
            f2 = np.exp(y2)
            sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1)
            x1 = x2
            y1 = y2
            f1 = f2

        tsstoo = f1
        if np.isnan(sumint):
            sumint = 0.

    return tsstoo, sumint


def hotspot_calculations_vec(hotspot, lai, ko, ks, dso, tss):
    tsstoo = np.zeros(tss.shape)
    sumint = np.zeros(lai.shape)

    # Treatment of the hotspot-effect
    alf = np.ones(lai.shape) * 1e36
    alf[hotspot > 0] = (dso[hotspot > 0] / hotspot[hotspot > 0]) * 2. / (ks[hotspot > 0] + ko[hotspot > 0])

    index = np.logical_and(lai > 0, alf == 0)
    # The pure hotspot
    tsstoo[index] = tss[index]
    sumint[index] = (1. - tss[index]) / (ks[index] * lai[index])

    # Outside the hotspot
    index = np.logical_and(lai > 0, alf != 0)
    fhot = lai[index] * np.sqrt(ko[index] * ks[index])
    # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
    x1 = np.zeros(fhot.shape)
    y1 = np.zeros(fhot.shape)
    f1 = np.ones(fhot.shape)
    fint = (1. - np.exp(-alf[index])) * .05
    for istep in range(1, 21):
        if istep < 20:
            x2 = -np.log(1. - istep * fint) / alf[index]
        else:
            x2 = np.ones(fhot.shape)
        y2 = -(ko[index] + ks[index]) * lai[index] * x2 + fhot * (1. - np.exp(-alf[index] * x2)) / alf[index]
        f2 = np.exp(y2)
        sumint[index] = sumint[index] + (f2 - f1) * (x2 - x1) / (y2 - y1)
        x1 = np.copy(x2)
        y1 = np.copy(y2)
        f1 = np.copy(f2)

    tsstoo[index] = f1
    sumint[np.isnan(sumint)] = 0.

    return tsstoo, sumint


def define_geometric_constant(tts, tto, psi):
    tants = np.tan(np.radians(tts))
    tanto = np.tan(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    dso = np.sqrt(tants ** 2. + tanto ** 2. - 2. * tants * tanto * cospsi)
    return dso


def rsoil_inv(lai, hotspot, lidf, vza, sza, psi, skyl, rho, tau, rho_canopy):
    """ Analytical inversion of 4SAIL to retrieve soil reflectance based on
    leaf and canopy spectra.

    Parameters
    ----------
    lai : ndarray
        Leaf Area Index.
    hotspot : ndarray
        Hotspot parameter.
    lidf : ndarray
        Leaf Inclination Distribution at regular angle steps.
    vza : ndarray
        View(sensor) Zenith Angle (degrees).
    sza : ndarray
        Sun Zenith Angle (degrees).
    psi : ndarray
        Relative Sensor-Sun Azimuth Angle (degrees).
    skyl : ndarray
        Diffuse to total irradiance ratio
    rho : ndarray
        leaf lambertian reflectance.
    tau : ndarray
        leaf transmittance.
    rho_canopy : ndarray
        canopy directional reflectance.

    Returns
    -------
    rsoil : ndarray
        Soil lambertian reflectance.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """

    # weighted_sum_over_lidf
    if np.isscalar(lai):
        ks, ko, bf, sob, sof = weighted_sum_over_lidf(lidf, sza, vza, psi)
    else:
        ks, ko, bf, sob, sof = weighted_sum_over_lidf_vec(lidf, sza, vza, psi)

    # Geometric factors to be used later with rho and tau
    sdb = 0.5 * (ks + bf)
    sdf = 0.5 * (ks - bf)
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1. + bf)
    ddf = 0.5 * (1. - bf)

    # Here rho and tau come in
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sigf = np.maximum(1e-36, sigf)
    sigb = np.maximum(1e-36, sigb)
    att = 1. - sigf
    m = np.sqrt(att ** 2. - sigb ** 2.)
    sb = sdb * rho + sdf * tau
    sf = sdf * rho + sdb * tau
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau
    w = sob * rho + sof * tau

    del sdb, sdf, dob, dof, ddb, ddf

    e1 = np.exp(-lai * m)
    e2 = e1 ** 2.
    rinf = (att - m) / sigb
    rinf2 = rinf ** 2.
    re = rinf * e1
    denom = 1. - rinf2 * e2
    if np.isscalar(lai):
        J1ks = jfunc1(ks, m, lai)
        J1ko = jfunc1(ko, m, lai)
    else:
        J1ks = jfunc1_vec(ks, m, lai)
        J1ko = jfunc1_vec(ko, m, lai)
    J2ks = jfunc2(ks, m, lai)
    J2ko = jfunc2(ko, m, lai)
    Pss = (sf + sb * rinf) * J1ks
    Qss = (sf * rinf + sb) * J2ks
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko

    tdd = (1. - rinf2) * e1 / denom
    rdd = rinf * (1. - e2) / denom
    tsd = (Pss - re * Qss) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    del e1, e2, Qv, Pv, att, sigb, sigf, J2ko, denom, re

    tss = np.exp(-ks * lai)
    too = np.exp(-ko * lai)
    z = jfunc2(ks, ko, lai)
    g1 = (z - J1ks * too) / (ko + m)
    g2 = (z - J1ko * tss) / (ks + m)
    Tv1 = (vf * rinf + vb) * g1
    Tv2 = (vf + vb * rinf) * g2
    T1 = Tv1 * (sf + sb * rinf)
    T2 = Tv2 * (sf * rinf + sb)
    T3 = (rdo * Qss + tdo * Pss) * rinf

    del vb, vf, Pss, Qss, J1ko, g1, g2, m, sb, sf, z

    # Multiple scattering contribution to bidirectional canopy reflectance
    rsod = (T1 + T2 - T3) / (1. - rinf2)

    del Tv1, Tv2, T1, T2, T3, rinf2, rinf, J2ks, J1ks

    # Hotspot effect
    dso = define_geometric_constant(sza, vza, psi)
    if np.isscalar(lai):
        tsstoo, sumint = hotspot_calculations(hotspot, lai, ko, ks, dso, tss)
    else:
        tsstoo, sumint = hotspot_calculations_vec(hotspot, lai, ko, ks, dso, tss)

    # Single scattering contribution
    rsos = w * lai * sumint

    del ko, ks, sumint, w, dso, psi, lai

    # Total canopy contribution
    rso = rsos + rsod

    # Invert Eq. 11 of Verhoef (2007) to retrieve rsoil
    # a_factor * rsoil**2 + b_factor * rsoil + c_factor = 0
    c_factor = rho_canopy - ((1. - skyl) * rso + skyl * rdo)
    a_factor = rdd * (1. - skyl) * (tsstoo - tss * too)
    b_factor = - (c_factor * rdd +
                  (1. - skyl) * (tsstoo + (tss + tsd) * tdo + tsd * too) +
                  skyl * tdd * (tdo + too))

    root = np.sqrt(b_factor ** 2 - 4 * a_factor * c_factor)
    # Only one root solution is valid
    rsoil = (-b_factor - root) / (2 * a_factor)
    # For non-valid solutions we assume that soil reflectance equals the measured reflectance
    no_valid = ~np.isfinite(rsoil)
    rsoil[no_valid] = rho_canopy[no_valid]
    return rsoil
  
    
def surface_emissivity(lai, lidf, vza, e_leaf=0.99, e_soil=0.97, tau=0):
    """ 4SAIL simulations to retrieve surface directional emissivity.

    Parameters
    ----------
    lai : ndarray
        Leaf Area Index.
    lidf : ndarray
        Leaf Inclination Distribution at regular angle steps.
    vza : ndarray
        View(sensor) Zenith Angle (degrees).
    e_leaf : ndarray
        leaf bihemispherical emissivity.
    e_soil : ndarray
        soil bihemispherical emissivity.

    Returns
    -------
    emiss : ndarray
        Surface directional emissivity.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    """

    # weighted_sum_over_lidf
    ko, bf = weighted_sum_over_lidf_vec(lidf,
                                        np.zeros(vza.shape),
                                        vza,
                                        np.zeros(vza.shape))[1:3]

    # Geometric factors to be used later with rho and tau
    dob = 0.5 * (ko + bf)
    dof = 0.5 * (ko - bf)
    ddb = 0.5 * (1. + bf)
    ddf = 0.5 * (1. - bf)

    # Here rho and tau come in
    rho = 1. - e_leaf
    if np.isscalar(rho):
        rho = np.full(lai.shape, rho)[np.newaxis, :]
    if np.isscalar(tau):
        tau = np.full(lai.shape, tau)[np.newaxis, :]
    sigb = ddb * rho + ddf * tau
    sigf = ddf * rho + ddb * tau
    sigf = np.maximum(1e-36, sigf)
    sigb = np.maximum(1e-36, sigb)
    att = 1. - sigf
    m = np.sqrt(att ** 2. - sigb ** 2.)
    vb = dob * rho + dof * tau
    vf = dof * rho + dob * tau

    del dob, dof, ddb, ddf

    J1ko = jfunc1_vec(ko, m, lai)
    J2ko = jfunc2(ko, m, lai)
    e1 = np.exp(-lai * m)
    e2 = e1 ** 2.
    rinf = (att - m) / sigb
    rinf2 = rinf ** 2.
    re = rinf * e1
    denom = 1. - rinf2 * e2
    Pv = (vf + vb * rinf) * J1ko
    Qv = (vf * rinf + vb) * J2ko
    
    del att, sigb, sigf, J2ko, J1ko, vb, vf, m

    tdd = (1. - rinf2) * e1 / denom
    rdd = rinf * (1. - e2) / denom
    tdo = (Pv - re * Qv) / denom
    rdo = (Qv - re * Pv) / denom

    too = np.exp(-ko * lai)

    del ko, e1, e2, Qv, Pv, denom, rinf2, rinf

    # Interaction with the soil
    rsoil = 1. - e_soil
    dn = 1. - rsoil * rdd
    dn = np.maximum(1e-36, dn)
    rdot = rdo + tdd * rsoil * (tdo + too) / dn

    emiss = 1 - rdot
    
    return emiss


def get_diffuse_radiation_6S(aot, wvp, sza, saa, date,
                             altitude=0.1, wls_step=10, n_jobs=None):
    s = sixs.SixS()
    s.atmos_profile = sixs.AtmosProfile.PredefinedType(
        sixs.AtmosProfile.MidlatitudeSummer)

    s.aeroprofile = sixs.AeroProfile.PredefinedType(
        sixs.AeroProfile.Continental)

    s.ground_reflectance = sixs.GroundReflectance.HomogeneousLambertian(0)

    if np.isfinite(wvp) and wvp > 0:
        s.atmos_profile = sixs.AtmosProfile.UserWaterAndOzone(wvp, 0.9)

    if np.isfinite(aot) and aot > 0:
        s.aot550 = aot

    s.geometry.solar_z = sza
    s.geometry.solar_a = saa
    s.geometry.view_z = 0
    s.geometry.view_a = 0
    s.geometry.day = date.day
    s.geometry.month = date.month

    s.altitudes.set_target_custom_altitude(altitude)
    s.wavelength = sixs.Wavelength(0.4, 2.5)

    wls = np.arange(400, 2501)
    wls_sim = np.arange(400, 2501, wls_step)

    wv, res = sixs.SixSHelpers.Wavelengths.run_wavelengths(s,
                                                           wls_sim / 1000.,
                                                           verbose=False,
                                                           n=n_jobs)

    eg_d = np.array(sixs.SixSHelpers.Wavelengths.extract_output(res,
                                                                'diffuse_solar_irradiance'))

    eg_s = np.array(sixs.SixSHelpers.Wavelengths.extract_output(res,
                                                                'direct_solar_irradiance'))

    eg_d = np.maximum(eg_d, 0)
    eg_s = np.maximum(eg_s, 0)
    skyl = np.full_like(wls, np.nan, dtype=np.float64)
    # Fill the diffuse values into a full wavelenght array
    valid = np.in1d(wls, wls_sim, assume_unique=True)
    skyl[valid] = eg_d / (eg_d + eg_s)
    # Fill nans by linear interpolation
    nans, x = np.isnan(skyl), lambda z: z.nonzero()[0]
    skyl[nans] = np.interp(x(nans), x(~nans), skyl[~nans])

    return skyl



