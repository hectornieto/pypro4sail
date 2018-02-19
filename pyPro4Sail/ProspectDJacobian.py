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
* :func:`JacProspect5` Computes the PROSPECT5 Jacobian.
* :func:`JacProspect5_wl` Computes the PROSPECT5 Jacobian for a specific wavelenght, aimed for computing speed.

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
from scipy.special import expn
import numpy as np

wls,refr_index,Cab_k,Car_k,Cbrown_k,Cw_k,Cm_k,Ant_k=spectral_lib

paramsProspect5=('N_leaf','Cab','Car','Cbrown','Cw','Cm', 'Ant')

def JacProspectD(Nleaf,Cab,Car,Cbrown,Cw,Cm, Ant):
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
    l : 1D_array_like
        wavelenght (nm).
    Delta_rho : 2D_array_like
        Jacobian, leaf reflectance .
    Delta_tau : 2D_array_like
        Jacobian, leaf transmittance .
    rho : 1D_array_like
        leaf reflectance .
    tau : 1D_array_like
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
    
    l=np.array(wls)
    
    k=(float(Cab)*np.array(Cab_k)+float(Car)*np.array(Car_k)
        +float(Cbrown)*np.array(Cbrown_k)+float(Cw)*np.array(Cw_k)
        +float(Cm)*np.array(Cm_k)+float(Ant)*np.array(Ant_k))/float(Nleaf)
    Delta_k=np.array([-(float(Cab)*np.array(Cab_k)+float(Car)*np.array(Car_k)
        +float(Cbrown)*np.array(Cbrown_k)+float(Cw)*np.array(Cw_k)
        +float(Cm)*np.array(Cm_k)+float(Ant)*np.array(Ant_k))/float(Nleaf)**2,np.array(Cab_k)/float(Nleaf),
        np.array(Car_k)/float(Nleaf),np.array(Cbrown_k)/float(Nleaf),
        np.array(Cw_k)/float(Nleaf),np.array(Cm_k)/float(Nleaf),np.array(Ant_k)/float(Nleaf)])
    k[k<=0]=1e-6    
    trans=(1.-k)*np.exp(-k)+k**2.*expn(1.,k)
    Delta_trans=(-np.exp(-k)-(1.-k)*np.exp(-k)+2*k*expn(1.,k)-k**2*expn(0,k))*Delta_k
    
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
    Delta_ra=((2.*t12*t21*r21*trans*(1.-r21**2.*trans**2.)+2.*t12*t21*r21*trans**2.*r21**2.*trans)/(1.-r21**2.*trans**2.)**2)*Delta_trans

    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    Delta_ta=((t12*t21*(1.-r21**2.*trans**2.)+2.*t12*t21*trans*r21**2.*trans)/(1.-r21**2.*trans**2.)**2)*Delta_trans 

    r90=(ra-y)/x
    Delta_r90=Delta_ra/x

    t90=ta/x
    Delta_t90=Delta_ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    Delta_delta=(0.5*((t90**2.-r90**2.-1.)**2.-4.*r90**2.)**-0.5)*(2*(t90**2.-r90**2.-1.)
                *(2*t90*Delta_t90-2*r90*Delta_r90)-8*r90*Delta_r90)

    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    Delta_beta=((2*r90*Delta_r90-2*t90*Delta_t90-Delta_delta)*(2.*r90)-(1.+r90**2.-t90**2.-delta)*2*Delta_r90)/(2.*r90)**2
    
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    Delta_va=((2*r90*Delta_r90-2*t90*Delta_t90+Delta_delta)*(2.*r90)-(1.+r90**2.-t90**2.+delta)*2*Delta_r90)/(2.*r90)**2

    vb=np.sqrt(beta*(va-r90)/(va*(beta-r90)))
    Delta_vb=0.5*(beta*(va-r90)/(va*(beta-r90)))**-.5*(((Delta_beta*(va-r90)+
        beta*(Delta_va-Delta_r90))*va*(beta-r90))-(beta*(va-r90)*(Delta_va*(beta-r90)+va*(Delta_beta-Delta_r90))))/(va*(beta-r90))**2

    vbNN = vb**(float(Nleaf)-1.)
    Delta_vbNN=np.zeros(Delta_vb.shape)
    Delta_vbNN[0,:]=np.exp((float(Nleaf)-1.)*np.log(vb))*(np.log(vb)+(float(Nleaf)-1.)*(1./vb)*Delta_vb[0,:])
    Delta_vbNN[1:,:]=np.exp((float(Nleaf)-1.)*np.log(vb))*((float(Nleaf)-1.)*(1./vb)*Delta_vb[1:,:])

    vbNNinv = 1./vbNN
    Delta_vbNNinv=-Delta_vbNN/vbNN**2

    vainv = 1./va
    Delta_vainv=-Delta_va/va**2

    s1=ta*t90*(vbNN-vbNNinv)
    Delta_s1=(Delta_ta*t90+ta*Delta_t90)*(vbNN-vbNNinv)+ta*t90*(Delta_vbNN-Delta_vbNNinv)
    
    s2=ta*(va-vainv)
    Delta_s2=Delta_ta*(va-vainv)+ta*(Delta_va-Delta_vainv)
    
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    Delta_s3=(Delta_va*vbNN+va*Delta_vbNN)-(Delta_vainv*vbNNinv+vainv*Delta_vbNNinv)-(Delta_r90*(vbNN-vbNNinv)+r90*(Delta_vbNN-Delta_vbNNinv))

    rho=ra+s1/s3
    Delta_rho=Delta_ra+(Delta_s1*s3-s1*Delta_s3)/s3**2    

    tau=s2/s3
    Delta_tau=(Delta_s2*s3-s2*Delta_s3)/s3**2            
    return l,Delta_rho,Delta_tau, rho, tau


def JacProspectD_wl(wl,Nleaf,Cab,Car,Cbrown,Cw,Cm,Ant):
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
    Delta_rho : 1D_array_like
        Jacobian, leaf reflectance .
    Delta_tau : 1D_array_like
        Jacobian, leaf transmittance .
    rho : float
        leaf reflectance .
    tau : float
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

    wl_index=wls.index(wl)
    Cab_abs=float(Cab_k[wl_index])
    Car_abs=float(Car_k[wl_index])
    Cbrown_abs=float(Cbrown_k[wl_index])
    Cw_abs=float(Cw_k[wl_index])
    Cm_abs=float(Cm_k[wl_index])
    Ant_abs=float(Ant_k[wl_index])
    n=float(refr_index[wl_index])

    k=(float(Cab)*Cab_abs+float(Car)*Car_abs
        +float(Cbrown)*Cbrown_abs+float(Cw)*Cw_abs
        +float(Cm)*Cm_abs+float(Ant)*Ant_abs)/float(Nleaf)
    Delta_k=np.array([-(float(Cab)*Cab_abs+float(Car)*Car_abs
        +float(Cbrown)*Cbrown_abs+float(Cw)*Cw_abs
        +float(Cm)*Cm_abs+float(Ant)*Ant_abs)/float(Nleaf)**2,Cab_abs/float(Nleaf),
        Car_abs/float(Nleaf),Cbrown_abs/float(Nleaf),
        Cw_abs/float(Nleaf),Cm_abs/float(Nleaf),Ant_abs/float(Nleaf)])
        
    trans=(1.-k)*np.exp(-k)+k**2.*expn(1.,k)
    Delta_trans=(-np.exp(-k)-(1.-k)*np.exp(-k)+2*k*expn(1.,k)-k**2*expn(0,k))*Delta_k
    
    #==============================================================================
    # reflectance and transmittance of one layer
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969),
    # Interaction of isotropic ligth with a compact plant leaf, J. Opt.
    # Soc. Am., 59(10):1376-1379.
    #==============================================================================
    #reflectivity and transmissivity at the interface
    alpha=40.;
    t12=tav_wl(alpha,n)
    t21=tav_wl(90.,n)/n**2.
    r12=1.-t12
    r21=1.-t21
    x=tav_wl(alpha,n)/tav_wl(90.,n)
    y=x*(tav_wl(90.,n)-1.)+1.-tav_wl(alpha,n)
    #reflectance and transmittance of the elementary layer N = 1
    ra=r12+(t12*t21*r21*trans**2.)/(1.-r21**2.*trans**2.)
    Delta_ra=((2.*t12*t21*r21*trans*(1.-r21**2.*trans**2.)+2.*t12*t21*r21*trans**2.*r21**2.*trans)/(1.-r21**2.*trans**2.)**2)*Delta_trans

    ta=(t12*t21*trans)/(1.-r21**2.*trans**2.)
    Delta_ta=((t12*t21*(1.-r21**2.*trans**2.)+2.*t12*t21*trans*r21**2.*trans)/(1.-r21**2.*trans**2.)**2)*Delta_trans 

    r90=(ra-y)/x
    Delta_r90=Delta_ra/x

    t90=ta/x
    Delta_t90=Delta_ta/x
    #==============================================================================
    # reflectance and transmittance of N layers
    # Stokes G.G. (1862), On the intensity of the light reflected from
    # or transmitted through a pile of plates, Proc. Roy. Soc. Lond.,
    # 11:545-556.
    #==============================================================================
    delta=np.sqrt((t90**2.-r90**2.-1.)**2.-4.*r90**2.)
    Delta_delta=(0.5*((t90**2.-r90**2.-1.)**2.-4.*r90**2.)**-0.5)*(2*(t90**2.-r90**2.-1.)
                *(2*t90*Delta_t90-2*r90*Delta_r90)-8*r90*Delta_r90)

    beta=(1.+r90**2.-t90**2.-delta)/(2.*r90)
    Delta_beta=((2*r90*Delta_r90-2*t90*Delta_t90-Delta_delta)*(2.*r90)-(1.+r90**2.-t90**2.-delta)*2*Delta_r90)/(2.*r90)**2
    
    va=(1.+r90**2.-t90**2.+delta)/(2.*r90)
    Delta_va=((2*r90*Delta_r90-2*t90*Delta_t90+Delta_delta)*(2.*r90)-(1.+r90**2.-t90**2.+delta)*2*Delta_r90)/(2.*r90)**2

    vb=np.sqrt(beta*(va-r90)/(va*(beta-r90)))
    Delta_vb=0.5*(beta*(va-r90)/(va*(beta-r90)))**-.5*(((Delta_beta*(va-r90)+
        beta*(Delta_va-Delta_r90))*va*(beta-r90))-(beta*(va-r90)*(Delta_va*(beta-r90)+va*(Delta_beta-Delta_r90))))/(va*(beta-r90))**2

    vbNN = vb**(float(Nleaf)-1.)
    Delta_vbNN=np.zeros(Delta_vb.shape)
    Delta_vbNN[0]=np.exp((float(Nleaf)-1.)*np.log(vb))*(np.log(vb)+(float(Nleaf)-1.)*(1./vb)*Delta_vb[0])
    Delta_vbNN[1:]=np.exp((float(Nleaf)-1.)*np.log(vb))*((float(Nleaf)-1.)*(1./vb)*Delta_vb[1:])

    vbNNinv = 1./vbNN
    Delta_vbNNinv=-Delta_vbNN/vbNN**2

    vainv = 1./va
    Delta_vainv=-Delta_va/va**2

    s1=ta*t90*(vbNN-vbNNinv)
    Delta_s1=(Delta_ta*t90+ta*Delta_t90)*(vbNN-vbNNinv)+ta*t90*(Delta_vbNN-Delta_vbNNinv)
    
    s2=ta*(va-vainv)
    Delta_s2=Delta_ta*(va-vainv)+ta*(Delta_va-Delta_vainv)
    
    s3=va*vbNN-vainv*vbNNinv-r90*(vbNN-vbNNinv)
    Delta_s3=(Delta_va*vbNN+va*Delta_vbNN)-(Delta_vainv*vbNNinv+vainv*Delta_vbNNinv)-(Delta_r90*(vbNN-vbNNinv)+r90*(Delta_vbNN-Delta_vbNNinv))

    rho=ra+s1/s3
    Delta_rho=Delta_ra+(Delta_s1*s3-s1*Delta_s3)/s3**2    

    tau=s2/s3
    Delta_tau=(Delta_s2*s3-s2*Delta_s3)/s3**2            
    return wl, Delta_rho,Delta_tau, rho, tau


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

    from numpy import log,zeros,sqrt, sin, radians, pi
    s=ref.shape[0]
    theta=radians(theta)
    r2=ref**2.0
    rp=r2+1.0
    rm=r2-1.0
    a=((ref+1.0)**2.0)/2.0
    k=-(r2-1.0)**2.0/4.0
    ds=sin(theta)
    if theta==0.0:
        f=4.0*ref/(ref+1.0)**2.0
        return f
    else:
        if theta == pi/2.0:
            b1=zeros(s)
        else:
            b1=sqrt((ds**2.0-rp/2.0)**2.0+k)
    k2=k**2.0
    rm2=rm**2.0;
    b2=ds**2.0-rp/2.0;
    b=b1-b2;
    ts=(k2/(6.0*b**3.0)+k/b-b/2.0)-(k2/(6.0*a**3.0)+k/a-a/2.0)
    tp1=-2.0*r2*(b-a)/(rp**2.0)
    tp2=-2.0*r2*rp*log(b/a)/rm2
    tp3=r2*(b**-1.0-a**-1.0)/2.0
    tp4=16.0*r2**2.0*(r2**2.0+1.0)*log((2.0*rp*b-rm2)/(2.0*rp*a-rm2))/(rp**3.0*rm2)
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
        
    from math import sin, radians, pi, log, sqrt
    theta=radians(theta)
    r2=ref**2.0
    rp=r2+1.0
    rm=r2-1.0
    a=((ref+1.0)**2.0)/2.0
    k=-(r2-1.0)**2.0/4.0
    ds=sin(theta)
    if theta==0.0:
        f=4.0*ref/(ref+1.0)**2.0
        return f
    elif theta == pi/2.0:
        b1=0.0
    else:
        b1=sqrt((ds**2.0-rp/2.0)**2.0+k)
    k2=k**2.0
    rm2=rm**2.0;
    b2=ds**2.0-rp/2.0;
    b=b1-b2;
    ts=(k2/(6.0*b**3.0)+k/b-b/2.0)-(k2/(6.0*a**3.0)+k/a-a/2.0)
    tp1=-2.0*r2*(b-a)/(rp**2.0)
    tp2=-2.0*r2*rp*log(b/a)/rm2
    tp3=r2*(b**-1.0-a**-1.0)/2.0
    tp4=16.0*r2**2.0*(r2**2.0+1.0)*log((2.0*rp*b-rm2)/(2.0*rp*a-rm2))/(rp**3.0*rm2)
    tp5=16.0*r2**3.0*((2.0*rp*b-rm2)**-1.0-(2.0*rp*a-rm2)**-1.0)/rp**3.0
    tp=tp1+tp2+tp3+tp4+tp5
    f=(ts+tp)/(2.0*ds**2.0)
    
    return f
