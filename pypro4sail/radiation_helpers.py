# This file is part of pyPro4SAIL for calculating the radiation component
# based on simplified Campbell RTM
# Copyright 2025 Hector Nieto and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Adapted on August 13 2025 from pyTSEB.net_radiation module
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This package contains functions for estimating the net shortwave and longwave radiation
for soil and canopy layers. Additional packages needed are.

* :doc:`meteo_utils` for the estimation of meteorological variables.

PACKAGE CONTENTS
================
* :func:`calc_spectra_Cambpell` Canopy spectrum using the [Campbell1998]_
    Radiative Transfer Model
* :func:`calc_K_be_Campbell` Beam extinction coefficient.

'''

import numpy as np

#==============================================================================
# List of constants used in the radiation_helpers Module
#==============================================================================
TAUD_STEP_SIZE_DEG = 5


def calc_spectra_Cambpell(lai,
                          sza,
                          rho_leaf,
                          tau_leaf,
                          rho_soil,
                          x_lad=1,
                          lai_eff=None):
    """ Canopy spectra

    Estimate canopy spectral using the [Campbell1998]_
    Radiative Transfer Model

    Parameters
    ----------
    lai : array_like
        Effective Leaf (Plant) Area Index.
    sza : array_like
        Sun Zenith Angle (degrees).
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float, or array_like
        Soil bihemispherical reflectance
    x_lad : array_like,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    lai_eff : array_like or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    albb : array_like
        Beam (black sky) canopy albedo
    albd : array_like
        Diffuse (white sky) canopy albedo
    taubt : array_like
        Beam (black sky) canopy transmittance
    taudt : array_like
        Beam (white sky) canopy transmittance

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    """

    # calculate aborprtivity
    amean = 1.0 - rho_leaf - tau_leaf
    amean_sqrt = np.sqrt(amean)
    del rho_leaf, tau_leaf, amean

    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if lai_eff is None:
        lai_eff = np.asarray(lai)
    else:
        lai_eff = np.asarray(lai_eff)

    # D I F F U S E   C O M P O N E N T S
    # Integrate to get the diffuse transmitance
    taud = _calc_taud(x_lad, lai)

    # Diffuse light canopy reflection coefficients  for a deep canopy
    akd = -np.log(taud) / lai
    rcpy= (1.0 - amean_sqrt) / (1.0 + amean_sqrt)  # Eq 15.7
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)  # Eq 15.8

    # Diffuse canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akd * lai
    del akd
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    xnum = (rdcpy * rdcpy - 1.0) * neg_exp
    xden = (rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp
    taudt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)  # Eq 15.9
    del rdcpy, fact

    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = calc_K_be_Campbell(sza, x_lad)  # Eq. 15.4

    # Direct beam canopy reflection coefficients for a deep canopy
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)  # Eq 15.8
    del rcpy, sza, x_lad
    # Beam canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    del amean_sqrt, akb, lai_eff
    xnum = (rbcpy * rbcpy - 1.0) * neg_exp
    xden = (rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp
    taubt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
    del expfac
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)  # Eq 15.9
    del rbcpy, fact

    taubt, taudt, albb, albd, rho_soil = map(np.array,
                                             [taubt, taudt, albb, albd, rho_soil])

    taubt[np.isnan(taubt)] = 1
    taudt[np.isnan(taudt)] = 1
    albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
    albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]

    return albb, albd, taubt, taudt


def calc_K_be_Campbell(theta, x_lad=1, radians=False):
    ''' Beam extinction coefficient

    Calculates the beam extinction coefficient based on [Campbell1998]_ ellipsoidal
    leaf inclination distribution function.

    Parameters
    ----------
    theta : array_like
        incidence zenith angle.
    x_lad : array_like, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.
    radians : bool, optional
        Should be True if theta is in radians.
        Default is False.

    Returns
    -------
    K_be : array_like
        beam extinction coefficient.

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    if not radians:
        theta = np.radians(theta)

    K_be = (np.sqrt(x_lad**2 + np.tan(theta)**2)
            / (x_lad + 1.774 * (x_lad + 1.182)**-0.733))

    return K_be


def _calc_taud(x_lad, lai):

    taud = 0
    for angle in range(0, 90, TAUD_STEP_SIZE_DEG):
        angle = np.radians(angle)
        akd = calc_K_be_Campbell(angle, x_lad, radians=True)
        taub = np.exp(-akd * lai)
        taud += taub * np.cos(angle) * np.sin(angle) * np.radians(TAUD_STEP_SIZE_DEG)

    return 2.0 * taud


def leafangle_2_chi(alpha):
    """
    Convert [Campbell1990]_ mean leaf inclination angle to the Xi parameter

    Parameters
    ----------
    alpha : array_like
        Mean leaf inclination angle (degrees)


    Returns
    -------
    x_lad: array_like
        Xi parameter for the ellipsoidal Leaf Angle Distribution function.

    References
    ----------
    .. [Campbell1990] Campbell, G.S., 1990.
        Derivation of an angle density function for canopies with
        ellipsoidal leaf angle distributions.
        Agricultural and Forest Meteorology 49, 173–176.
        https://doi.org/10.1016/0168-1923(90)90030-A
    """
    alpha = np.radians(alpha)
    x_lad = ((alpha / 9.65) ** (1. / -1.65)) - 3.

    return x_lad

