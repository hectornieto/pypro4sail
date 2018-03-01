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
* :func:`Prospect5` Runs PROSPECT5 leaf radiative transfer model.
* :func:`Prospect5_wl` Runs PROSPECT5 leaf radiative transfer model for a specific wavelenght, aimed for computing speed.

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
from pyPro4Sail import spectral_lib
from scipy.special import expi
import numpy as np

wls,refr_index,Cab_k,Car_k,Cbrown_k,Cw_k,Cm_k,Ant_k=spectral_lib

paramsProspect5=('N_leaf','Cab','Car','Cbrown','Cw','Cm', 'Ant')

def Prospect5(Nleaf,Cab,Car,Cbrown,Cw,Cm):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------    
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================
   
    l=np.array(wls)
    k=(Cab*np.array(Cab_k)+Car*np.array(Car_k)
        +Cbrown*np.array(Cbrown_k)+Cw*np.array(Cw_k)
        +Cm*np.array(Cm_k))/Nleaf
    k[k<=0]=1e-6
    
    trans=(1.-k)*np.exp(-k)+k**2.*expi(-k)
    trans[k<=0.0]=1.0
    
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.;
    n=np.array(refr_index)
    t12=tav(alpha,n)
    t21=tav(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav(alpha,n)/tav(90.,n)
    y=x*(tav(90.,n)-1.)+1.-tav(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    denominator=np.zeros(va.shape)+1e-14
    denominator[va*(beta-r90)> 1e-14]=va[va*(beta-r90)> 1e-14]*(beta[va*(beta-r90)> 1e-14]-r90[va*(beta-r90)> 1e-14])
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return l,rho,tau


def Prospect5_vec(Nleaf,Cab,Car,Cbrown,Cw,Cm):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------    
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================
    # Vectorize the inputs
    Nleaf,Cab,Car,Cbrown,Cw,Cm=(Nleaf[:,np.newaxis],
                                Cab[:,np.newaxis],
                                Car[:,np.newaxis],
                                Cbrown[:,np.newaxis],
                                Cw[:,np.newaxis],
                                Cm[:,np.newaxis])
    
    l=np.array(wls)
    k=(Cab*np.array(Cab_k)+Car*np.array(Car_k)
        +Cbrown*np.array(Cbrown_k)+Cw*np.array(Cw_k)
        +Cm*np.array(Cm_k))/Nleaf
    k[k<=0]=1e-6
    
    trans=(1.-k)*np.exp(-k)+k**2.*expi(-k)
    trans[k<=0.0]=1.0
    
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.;
    n=np.array(refr_index)
    t12=tav(alpha,n)
    t21=tav(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav(alpha,n)/tav(90.,n)
    y=x*(tav(90.,n)-1.)+1.-tav(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    denominator=np.zeros(va.shape)+1e-14
    denominator[va*(beta-r90)> 1e-14]=va[va*(beta-r90)> 1e-14]*(beta[va*(beta-r90)> 1e-14]-r90[va*(beta-r90)> 1e-14])
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return l,rho,tau

def Prospect5_wl(wl,Nleaf,Cab,Car,Cbrown,Cw,Cm):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from a single given wavelenght. Aimed for computation speed.

    Parameters
    ----------    
    wl : float
        wavelenght (nm) to simulate.
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : float
        wavelenght (nm).
    rho : float
        leaf reflectance .
    tau : float
        leaf transmittance. 
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================
    #Vectorize the input
    Nleaf,Cab,Car,Cbrown,Cw,Cm=(Nleaf[:,np.newaxis],
                                Cab[:,np.newaxis],
                                Car[:,np.newaxis],
                                Cbrown[:,np.newaxis],
                                Cw[:,np.newaxis],
                                Cm[:,np.newaxis])

    wl_index=wls==wl
    Cab_abs=float(Cab_k[wl_index])
    Car_abs=float(Car_k[wl_index])
    Cbrown_abs=float(Cbrown_k[wl_index])
    Cw_abs=float(Cw_k[wl_index])
    Cm_abs=float(Cm_k[wl_index])
    n=float(refr_index[wl_index])
    k=(Cab*Cab_abs+Car*Car_abs+Cbrown*Cbrown_abs
        +Cw*Cw_abs+Cm*Cm_abs)/Nleaf
    if k<=0.: 
        trans=1.0
    else:
        trans=(1.-k)*np.exp(-k)+(k**2)*expi(-k)
        if trans <0.0:trans=0.0
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.
    t12=tav_wl(alpha,n)
    t21=tav_wl(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav_wl(alpha,n)/tav_wl(90.,n)
    y=x*(tav_wl(90.,n)-1.)+1.-tav_wl(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    if ((t90**2.-r90**2.-1.)**2.-4.*r90**2.) > 0:
        delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    else:
        delta = 0
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    if va*(beta-r90)> 1e-14:
        denominator=va*(beta-r90)
    else:
        denominator=1e-36
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return wl,rho,tau

def ProspectD(Nleaf,Cab,Car,Cbrown,Cw,Cm, Ant):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------    
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================

    l=np.array(wls)
    k=(Cab*np.array(Cab_k)+Car*np.array(Car_k)
        +Cbrown*np.array(Cbrown_k)+Cw*np.array(Cw_k)
        +Cm*np.array(Cm_k)+Ant*np.array(Ant_k))/Nleaf
    k[k<=0]=1e-6
    
    trans=(1.-k)*np.exp(-k)+k**2.*expi(-k)
    trans[k<=0.0]=1.0
    
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.;
    n=np.array(refr_index)
    t12=tav(alpha,n)
    t21=tav(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav(alpha,n)/tav(90.,n)
    y=x*(tav(90.,n)-1.)+1.-tav(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    denominator=np.zeros(va.shape)+1e-14
    denominator[va*(beta-r90)> 1e-14]=va[va*(beta-r90)> 1e-14]*(beta[va*(beta-r90)> 1e-14]-r90[va*(beta-r90)> 1e-14])
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return l,rho,tau

def ProspectD_vec(Nleaf,Cab,Car,Cbrown,Cw,Cm, Ant):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from 400 nm to 2500 nm (1 nm step).

    Parameters
    ----------    
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : array_like
        wavelenght (nm).
    rho : array_like
        leaf reflectance .
    tau : array_like
        leaf transmittance .
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================
    # Vectorize the inputs
    Nleaf,Cab,Car,Cbrown,Cw,Cm, Ant=(Nleaf[:,np.newaxis],
                                Cab[:,np.newaxis],
                                Car[:,np.newaxis],
                                Cbrown[:,np.newaxis],
                                Cw[:,np.newaxis],
                                Cm[:,np.newaxis],
                                Ant[:,np.newaxis])
    
    l=np.array(wls)
    k=(Cab*np.array(Cab_k)+Car*np.array(Car_k)
        +Cbrown*np.array(Cbrown_k)+Cw*np.array(Cw_k)
        +Cm*np.array(Cm_k)+Ant*np.array(Ant_k))/Nleaf
    k[k<=0]=1e-6
    
    trans=(1.-k)*np.exp(-k)+k**2.*expi(-k)
    trans[k<=0.0]=1.0
    
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.;
    n=np.array(refr_index)
    t12=tav(alpha,n)
    t21=tav(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav(alpha,n)/tav(90.,n)
    y=x*(tav(90.,n)-1.)+1.-tav(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    denominator=np.zeros(va.shape)+1e-14
    denominator[va*(beta-r90)> 1e-14]=va[va*(beta-r90)> 1e-14]*(beta[va*(beta-r90)> 1e-14]-r90[va*(beta-r90)> 1e-14])
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return l,rho,tau

def ProspectD_wl(wl,Nleaf,Cab,Car,Cbrown,Cw,Cm, Ant):
    '''PROSPECT 5 Plant leaf reflectance and transmittance modeled 
    from a single given wavelenght. Aimed for computation speed.

    Parameters
    ----------    
    wl : float
        wavelenght (nm) to simulate.
    N   : float
        leaf structure parameter.
    Cab : float
        chlorophyll a+b content (mug cm-2).
    Car : float
        carotenoids content (mug cm-2).
    Cbrown : float
        brown pigments concentration (unitless).
    Cw  : float
        equivalent water thickness (g cm-2 or cm).
    Cm  : float
        dry matter content (g cm-2).

    Returns
    -------
    l : float
        wavelenght (nm).
    rho : float
        leaf reflectance .
    tau : float
        leaf transmittance. 
    
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
    #==============================================================================
    # Here are some examples observed during the LOPEX'93 experiment on
    # Fresh (F) and dry (D) leaves :
    # 
    # ---------------------------------------------
    #                N     Cab     Cw        Cm    
    # ---------------------------------------------
    # min          1.000    0.0  0.004000  0.001900
    # max          3.000  100.0  0.040000  0.016500
    # corn (F)     1.518   58.0  0.013100  0.003662
    # rice (F)     2.275   23.7  0.007500  0.005811
    # clover (F)   1.875   46.7  0.010000  0.003014
    # laurel (F)   2.660   74.1  0.019900  0.013520
    # ---------------------------------------------
    # min          1.500    0.0  0.000063  0.0019
    # max          3.600  100.0  0.000900  0.0165
    # bamboo (D)   2.698   70.8  0.000117  0.009327
    # lettuce (D)  2.107   35.2  0.000244  0.002250
    # walnut (D)   2.656   62.8  0.000263  0.006573
    # chestnut (D) 1.826   47.7  0.000307  0.004305
    #==============================================================================

    wl_index=wls==wl
    Cab_abs=float(Cab_k[wl_index])
    Car_abs=float(Car_k[wl_index])
    Cbrown_abs=float(Cbrown_k[wl_index])
    Cw_abs=float(Cw_k[wl_index])
    Cm_abs=float(Cm_k[wl_index])
    Ant_abs=float(Ant_k[wl_index])
    n=float(refr_index[wl_index])
    
    k=(Cab*Cab_abs+Car*Car_abs+Cbrown*Cbrown_abs
        +Cw*Cw_abs+Cm*Cm_abs+Ant*Ant_abs)/Nleaf
    if k<=0.: 
        trans=1.0
    else:
        trans=(1.-k)*np.exp(-k)+(k**2)*expi(-k)
        if trans <0.0:trans=0.0
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.
    t12=tav_wl(alpha,n)
    t21=tav_wl(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav_wl(alpha,n)/tav_wl(90.,n)
    y=x*(tav_wl(90.,n)-1.)+1.-tav_wl(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    r90=(ra-y)/x
    t90=ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    if ((t90**2.-r90**2.-1.)**2.-4.*r90**2.) > 0:
        delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    else:
        delta = 0
    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    if va*(beta-r90)> 1e-14:
        denominator=va*(beta-r90)
    else:
        denominator=1e-36
    vb=np.sqrt(beta*(va-r90)/(denominator))
    vbNN = vb**(Nleaf-1.)
    vbNNinv = 1./vbNN
    vainv = 1./va
    s1=ta*t90*(vbNN-vbNNinv)
    s2=ta*(va-vainv)
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    rho=ra+s1/s3
    tau=s2/s3
    
    return wl,rho,tau

def tav(theta,ref):
    '''
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
    '''


    s=ref.shape[0]
    theta=np.radians(theta)
    r2=ref**2.0
    rp=r2+1.0
    rm=r2-1.0
    a=((ref+1.0)**2.0)/2.0
    k=-(r2-1.0)**2.0/4.0
    ds=np.sin(theta)
    if theta==0.0:
        f=4.0*ref/(ref+1.0)**2.0
        return f
    else:
        if theta == np.pi/2.0:
            b1=np.zeros(s)
        else:
            b1=np.sqrt((ds**2.0-rp/2.0)**2.0+k)
    k2=k**2.0
    rm2=rm**2.0;
    b2=ds**2.0-rp/2.0;
    b=b1-b2;
    ts=(k2/(6.0*b**3.0)+k/b-b/2.0)-(k2/(6.0*a**3.0)+k/a-a/2.0)
    tp1=-2.0*r2*(b-a)/(rp**2.0)
    tp2=-2.0*r2*rp*np.log(b/a)/rm2
    tp3=r2*(b**-1.0-a**-1.0)/2.0
    tp4=16.0*r2**2.0*(r2**2.0+1.0)*np.log((2.0*rp*b-rm2)/(2.0*rp*a-rm2))/(rp**3.0*rm2)
    tp5=16.0*r2**3.0*((2.0*rp*b-rm2)**-1.0-(2.0*rp*a-rm2)**-1.0)/rp**3.0
    tp=tp1+tp2+tp3+tp4+tp5
    f=(ts+tp)/(2.0*ds**2.0)
    
    return f

def tav_wl(theta,ref):
    '''
    Average transmittivity at the leaf surface within a given solid angle. 

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
    '''
        

    theta=np.radians(theta)
    r2=ref**2.0
    rp=r2+1.0
    rm=r2-1.0
    a=((ref+1.0)**2.0)/2.0
    k=-(r2-1.0)**2.0/4.0
    ds=np.sin(theta)
    if theta==0.0:
        f=4.0*ref/(ref+1.0)**2.0
        return f
    elif theta == np.pi/2.0:
        b1=0.0
    else:
        b1=np.sqrt((ds**2.0-rp/2.0)**2.0+k)
    k2=k**2.0
    rm2=rm**2.0;
    b2=ds**2.0-rp/2.0;
    b=b1-b2;
    ts=(k2/(6.0*b**3.0)+k/b-b/2.0)-(k2/(6.0*a**3.0)+k/a-a/2.0)
    tp1=-2.0*r2*(b-a)/(rp**2.0)
    tp2=-2.0*r2*rp*np.log(b/a)/rm2
    tp3=r2*(b**-1.0-a**-1.0)/2.0
    tp4=16.0*r2**2.0*(r2**2.0+1.0)*np.log((2.0*rp*b-rm2)/(2.0*rp*a-rm2))/(rp**3.0*rm2)
    tp5=16.0*r2**3.0*((2.0*rp*b-rm2)**-1.0-(2.0*rp*a-rm2)**-1.0)/rp**3.0
    tp=tp1+tp2+tp3+tp4+tp5
    f=(ts+tp)/(2.0*ds**2.0)
    
    return f