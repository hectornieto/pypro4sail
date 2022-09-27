# -*- coding: utf-8 -*-
'''
Created on Apr 6 2015
@author: Hector Nieto (hnieto@ias.csic.es)

Modified on Apr 14 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains the main functions to run the coupled leaf-canopy model 
PROSPECT5+4SAIL. It requires to import both radiative transfer models.

* :doc:`FourSAIL` for simulating the canopy reflectance and transmittance factors.
* :doc:`Prospect5` for simulating the lambertian reflectance and transmittance of a leaf.

PACKAGE CONTENTS
================

* :func:`run` run Pro4SAIL based on originial PyProSAIL interface at http://pyprosail.readthedocs.org/en/latest/.
* :func:`run_TIR` runs the thermal component of 4SAIL to estimate the broadband at-sensor thermal radiance.
* :func:`CalcStephanBoltzmann` Blackbody Broadband radiation emission. 

EXAMPLE
=======

.. code-block:: python

    [N, chloro, caroten, brown, EWT, LMA, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth, LIDF]=[1.5,40,8,0.0,0,01,0,009,3,0.01,30,180,10,180,(-0.35,-0.15)]
    import pyPro4SAIL
    wl,rho=pyPro4SAIL.run(N, chloro, caroten, brown, EWT, LMA, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth, LIDF, skyl=0.2, soilType=pyPro4SAIL.DEFAULT_SOIL)

'''

from pypro4sail import four_sail, prospect
import numpy as np
import os

# Define Constants
SOIL_FOLDER = os.path.join(os.path.dirname(four_sail.__file__), 'spectra', 'soil_spectral_library')
DEFAULT_SOIL = 'ProSAIL_WetSoil.txt'
SB = 5.670373e-8  # Stephan Boltzmann constant (W m-2 K-4)

# Common leaf distributions
PLANOPHILE = (1, 0)
ERECTOPHILE = (-1, 0)
PLAGIOPHILE = (0, -1)
EXTREMOPHILE = (0, 1)
SPHERICAL = (-0.35, -0.15)
UNIFORM = (0, 0)


def run(N, chloro, caroten, brown, EWT, LMA, Ant, LAI, hot_spot, solar_zenith, solar_azimuth,
        view_zenith, view_azimuth, LIDF, skyl=0.2, soilType=DEFAULT_SOIL):
    ''' Runs Prospect5 4SAIL model to estimate canopy directional reflectance factor.
    
    Parameters
    ----------
    N : float
        Leaf structural parameter.
    chloro : float
        chlorophyll a+b content (mug cm-2).
    caroten : float
        carotenoids content (mug cm-2).
    brown : float
        brown pigments concentration (unitless).
    EWT  : float
        equivalent water thickness (g cm-2 or cm).
    LMA  : float
        dry matter content (g cm-2).
    LAI : float
        Leaf Area Index.
    hot_spot : float
        Hotspot parameter.
    solar_zenith : float
        Sun Zenith Angle (degrees).
    solar_azimuth : float
        Sun Azimuth Angle (degrees).
    view_zenith : float
        View(sensor) Zenith Angle (degrees).
    view_azimuth : float
        View(sensor) Zenith Angle (degrees).
    LIDF : float or tuple(float,float)
        Leaf Inclination Distribution Function parameter.
        
            * if float, mean leaf angle for the Cambpell Spherical LIDF.
            * if tuple, (a,b) parameters of the Verhoef's bimodal LIDF |LIDF[0]| + |LIDF[1]|<=1.
    skyl : float, optional
       Fraction of diffuse shortwave radiation, default=0.2.
    soilType : str, optional
        filename of the soil type, defautl use inceptisol soil type,
        see SoilSpectralLibrary folder.
    
    Returns
    -------
    wl : array_like
        wavelenghts.
    rho_canopy : array_like
        canopy reflectance factors.
    
    References
    ----------
    .. [Feret08] Feret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
        Properties Model Separating Photosynthetic Pigments, Remote Sensing of
        Environment.
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844.
    '''

    # Read the soil reflectance        
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, soilType))
    # wl_soil=rsoil[:,0]
    rsoil = np.array(rsoil[:, 1])

    # Calculate the lidf
    if type(LIDF) == tuple or type(LIDF) == list:
        if len(LIDF) != 2:
            print("ERROR, Verhoef's bimodal LIDF distribution must have two elements (LIDFa, LIDFb)")
            return None, None
        elif LIDF[0] + LIDF[1] > 1:
            print("ERROR,  |LIDFa| + |LIDFb| > 1 in Verhoef's bimodal LIDF distribution")
        else:
            lidf = four_sail.calc_lidf_verhoef(LIDF[0], LIDF[1])
    else:
        lidf = four_sail.calc_lidf_campbell(LIDF)

    # PROSPECT5 for leaf bihemispherical reflectance and transmittance
    wl, rho_leaf, tau_leaf = prospect.prospectd(N, chloro, caroten, brown, EWT, LMA, Ant)

    # Get the relative sun-view azimth angle
    psi = abs(solar_azimuth - view_azimuth)
    # 4SAIL for canopy reflectance and transmittance factors       
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot,
     rsodt, rsost, rsot, gammasdf, gammasdb,
     gammaso] = four_sail.foursail(LAI, hot_spot,
                                   lidf, solar_zenith, view_zenith, psi, rho_leaf, tau_leaf, rsoil)
    rho_canopy = rdot * skyl + rsot * (1 - skyl)

    return wl, rho_canopy


def run_TIR(emisVeg, emisSoil, T_Veg, T_Soil, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth,
            LIDF, T_VegSunlit=None, T_SoilSunlit=None, T_atm=0):
    ''' Estimates the broadband at-sensor thermal radiance using 4SAIL model.
    
    Parameters
    ----------
    emisVeg : float
        Leaf hemispherical emissivity.
    emisSoil : float
        Soil hemispherical emissivity.
    T_Veg : float
        Leaf temperature (Kelvin).
    T_Soil : float
        Soil temperature (Kelvin).
    LAI : float
        Leaf Area Index.
    hot_spot : float
        Hotspot parameter.
    solar_zenith : float
        Sun Zenith Angle (degrees).
    solar_azimuth : float
        Sun Azimuth Angle (degrees).
    view_zenith : float
        View(sensor) Zenith Angle (degrees).
    view_azimuth : float
        View(sensor) Zenith Angle (degrees).
    LIDF : float or tuple(float,float)
        Leaf Inclination Distribution Function parameter.
        
            * if float, mean leaf angle for the Cambpell Spherical LIDF.
            * if tuple, (a,b) parameters of the Verhoef's bimodal LIDF |LIDF[0]| + |LIDF[1]|<=1.
    T_VegSunlit : float, optional
        Sunlit leaf temperature accounting for the thermal hotspot effect,
        default T_VegSunlit=T_Veg.
    T_SoilSunlit : float, optional
        Sunlit soil temperature accounting for the thermal hotspot effect
        default T_SoilSunlit=T_Soil.
    T_atm : float, optional
        Apparent sky brightness temperature (Kelvin), 
        default T_atm =0K (no downwellig radiance).
    
    Returns
    -------
    Lw : float
        At sensor broadband radiance (W m-2).
    TB_obs : float
        At sensor brightness temperature (Kelvin).
    emiss : float
        Surface directional emissivity.

    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844.
    '''

    # Apply Kirchoff's law to get the soil and leaf bihemispherical reflectances
    rsoil = 1 - emisSoil
    rho_leaf = 1 - emisVeg
    tau_leaf = 0
    # Calculate the lidf,
    if type(LIDF) == tuple or type(LIDF) == list:
        if len(LIDF) != 2:
            print("ERROR, Verhoef's bimodal LIDF distribution must have two elements (LIDFa, LIDFb)")
            return None, None
        elif LIDF[0] + LIDF[1] > 1:
            print("ERROR,  |LIDFa| + |LIDFb| > 1 in Verhoef's bimodal LIDF distribution")
        else:
            lidf = four_sail.calc_lidf_verhoef(LIDF[0], LIDF[1])
    else:
        lidf = four_sail.calc_lidf_campbell(LIDF)

    # Get the relative sun-view azimth angle
    psi = abs(solar_azimuth - view_azimuth)
    # 4SAIL for canopy reflectance and transmittance factors       
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, rsdt, rdot,
     rsodt, rsost, rsot, gammasdf, gammasdb,
     gammaso] = four_sail.foursail(LAI, hot_spot,
                                   lidf, solar_zenith, view_zenith, psi, rho_leaf, tau_leaf, rsoil)

    tso = tsstoo + tss * (tdo + rsoil * rdd * too) / (1. - rsoil * rdd)
    gammad = 1 - rdd - tdd
    gammao = 1 - rdo - tdo - too
    ttot = (too + tdo) / (1. - rsoil * rdd)
    gammaot = gammao + ttot * rsoil * gammad
    gammasot = gammaso + ttot * rsoil * gammasdf

    aeev = gammaot
    aees = ttot * emisSoil

    # Get the different canopy broadband emssion components
    Hvc = CalcStephanBoltzmann(T_Veg)
    Hgc = CalcStephanBoltzmann(T_Soil)
    Hsky = CalcStephanBoltzmann(T_atm)

    if T_VegSunlit:  # Accout for different suntlit shaded temperatures
        Hvh = CalcStephanBoltzmann(T_VegSunlit)
    else:
        Hvh = Hvc
    if T_SoilSunlit:  # Accout for different suntlit shaded temperatures
        Hgh = CalcStephanBoltzmann(T_SoilSunlit)
    else:
        Hgh = Hgc

    # Calculate the blackbody emission temperature
    Lw = (rdot * Hsky + (
                aeev * Hvc + gammasot * emisVeg * (Hvh - Hvc) + aees * Hgc + tso * emisSoil * (Hgh - Hgc))) / np.pi
    TB_obs = (np.pi * Lw / SB) ** (0.25)

    # Estimate the apparent surface directional emissivity
    emiss = 1 - rdot
    return Lw, TB_obs, emiss


def CalcStephanBoltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody.
    
    Parameters
    ----------
    T_K : float
        body temperature (Kelvin).
    
    Returns
    -------
    M : float
        Emitted radiance (W m-2).'''
    import numpy as np

    M = SB * T_K ** 4
    return np.asarray(M)



