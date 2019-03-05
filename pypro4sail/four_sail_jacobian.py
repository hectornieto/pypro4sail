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
* :func:`JacFourSAIL` Computes the 4SAIL Jacobian.

Ancillary functions
-------------------
* :func:`JacCalcLIDF_Campbell` Calculates the Jacobian of the Leaf Inclination Distribution Function based on the [Campbell1990] ellipsoidal LIDF distribution.
* :func:`volscatt` Colume scattering functions and interception coefficients.
* :func:`JacJfunc1` Jacobian  of the J1 function.
* :func:`JacJfunc2` Jacobian of the J2 function.

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

params_SAIL=('LAI','hotspot','leaf_angle')
params_prosail=('N_leaf',
                'Cab',
                'Car',
                'Cbrown',
                'Cw',
                'Cm',
                'Ant',
                'LAI',
                'hotspot',
                'leaf_angle')

def JacCalcLIDF_Campbell(alpha,n_elements=18):
    '''Calculate the Leaf Inclination Distribution Function based on the 
    mean angle of [Campbell1990] ellipsoidal LIDF distribution.

    Parameters
    ----------
    alpha : float
        Mean leaf angle (degrees) use 57 for a spherical LIDF.
    n_elements : int
        Total number of equally spaced inclination angles .
    
    Returns
    -------
    Delta_lidf : list
        Jacobian of the Leaf Inclination Distribution Function for n_elements equally spaced angles.
    lidf : list
        Leaf Inclination Distribution Function for n_elements equally spaced angles.
        
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
    '''
    
    alpha=float(alpha)
    excent=np.exp(-1.6184e-5*alpha**3.+2.1145e-3*alpha**2.-1.2390e-1*alpha+3.2491)
    Delta_excent=np.exp(-1.6184e-5*alpha**3.+2.1145e-3*alpha**2.-1.2390e-1*alpha+3.2491)*(3*-1.6184e-5*alpha**2+
                    2.*2.1145e-3*alpha-1.2390e-1)
    sum0 = 0.
    freq=[]
    Delta_freq=[]
    step=90.0/n_elements
    for  i in range (n_elements):
        tl1=np.radians(i*step)
        tl2=np.radians((i+1.)*step)
        x1  = excent/(np.sqrt(1.+excent**2.*np.tan(tl1)**2.))
        Delta_x1=(Delta_excent*(np.sqrt(1.+excent**2.*np.tan(tl1)**2.))-
            excent*0.5*(1.+excent**2.*np.tan(tl1)**2.)**-0.5 
            *2*excent*np.tan(tl1)**2*Delta_excent)/(1.+excent**2.*np.tan(tl1)**2.)
        x2  = excent/(np.sqrt(1.+excent**2.*np.tan(tl2)**2.))
        Delta_x2=(Delta_excent*(np.sqrt(1.+excent**2.*np.tan(tl2)**2.))-
            excent*0.5*(1.+excent**2.*np.tan(tl2)**2.)**-0.5 
            *2*excent*np.tan(tl2)**2*Delta_excent)/(1.+excent**2.*np.tan(tl2)**2.)
        if excent == 1. :
            freq.append(abs(np.cos(tl1)-np.cos(tl2)))
            Delta_freq.append(0)
        else :
            alph  = excent/np.sqrt(abs(1.-excent**2.))
            Delta_alph=(Delta_excent*np.sqrt(abs(1.-excent**2.))-(excent*0.5*abs(1.-excent**2.)**-0.5 
                *(abs(1.-excent**2)/(1.-excent**2))*2*-excent*Delta_excent))/abs(1.-excent**2.)
            alph2 = alph**2.
            Delta_alph2=2*alph*Delta_alph
            x12 = x1**2.
            Delta_x12=2*x1*Delta_x1
            x22 = x2**2.
            Delta_x22=2*x2*Delta_x2
            if excent > 1. :
                alpx1 = np.sqrt(alph2+x12)
                Delta_alpx1=0.5*(alph2+x12)**-0.5*(Delta_alph2+Delta_x12)
                alpx2 = np.sqrt(alph2+x22)
                Delta_alpx2=0.5*(alph2+x22)**-0.5*(Delta_alph2+Delta_x22)
                dum   = x1*alpx1+alph2*np.log(x1+alpx1)
                Delta_dum=Delta_x1*alpx1+x1*Delta_alpx1+Delta_alph2*np.log(x1+alpx1)+alph2*(1./(x1+alpx1))*(Delta_x1+Delta_alpx1)               

                freq.append(abs(dum-(x2*alpx2+alph2*np.log(x2+alpx2))))
                Delta_freq_i=(abs(dum-(x2*alpx2+alph2*np.log(x2+alpx2)))/(dum-(x2*alpx2+alph2*np.log(x2+alpx2))))*(Delta_dum
                    -(Delta_x2*alpx2+x2*Delta_alpx2+Delta_alph2*np.log(x2+alpx2)+alph2*(1./(x2+alpx2))*(Delta_x2+Delta_alpx2)))
                Delta_freq.append(Delta_freq_i)

            else :
                almx1 = np.sqrt(alph2-x12)
                Delta_almx1=0.5*(alph2-x12)**-0.5*(Delta_alph2-Delta_x12)
                almx2 = np.sqrt(alph2-x22)
                Delta_almx2=0.5*(alph2-x22)**-0.5*(Delta_alph2-Delta_x22)
                dum   = x1*almx1+alph2*np.arcsin(x1/alph)
                Delta_dum=Delta_x1*almx1+x1*Delta_almx1+Delta_alph2*np.arcsin(x1/alph)+alph2*((
                            1./np.sqrt(1.-(x1/alph)**2))*(Delta_x1*alph-x1*Delta_alph)/alph**2)

                freq.append(abs(dum-(x2*almx2+alph2*np.arcsin(x2/alph))))                
                Delta_freq_i=(abs(dum-(x2*almx2+alph2*np.arcsin(x2/alph)))/(dum-
                    (x2*almx2+alph2*np.arcsin(x2/alph))))*(Delta_dum-(Delta_x2*almx2+x2*Delta_almx2
                    +Delta_alph2*np.arcsin(x2/alph)+alph2*(1./np.sqrt(1-(x2/alph)**2))*((Delta_x2*alph-x2*Delta_alph)/alph**2)))
                Delta_freq.append(Delta_freq_i)
    sum0 = sum(freq)
    Delta_sum0=sum(Delta_freq)
    lidf=[]
    Delta_lidf=[]
    for i in range(n_elements):
        lidf.append(float(freq[i])/sum0)
        Delta_lidf.append(float(Delta_freq[i]*sum0-freq[i]*Delta_sum0)/sum0**2)
    
    return lidf, Delta_lidf

def JacFourSAIL(lai,hotspot,lidf,tts,tto,psi,rho,tau,rsoil,Delta_rho=None,Delta_tau=None):
    ''' Runs 4SAIL canopy radiative transfer model.
    
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
    tss : 1D_array_like
        beam transmittance in the sun-target path.
    too : 1D_array_like
        beam transmittance in the target-view path.
    tsstoo : 1D_array_like
        beam tranmittance in the sur-target-view path.
    rdd : 1D_array_like
        canopy bihemisperical reflectance factor.
    tdd : 1D_array_like
        canopy bihemishperical transmittance factor.
    rsd : 1D_array_like 
        canopy directional-hemispherical reflectance factor.
    tsd : 1D_array_like
        canopy directional-hemispherical transmittance factor.
    rdo : 1D_array_like
        canopy hemispherical-directional reflectance factor.
    tdo : 1D_array_like
        canopy hemispherical-directional transmittance factor.
    rso : 1D_array_like
        canopy bidirectional reflectance factor.
    rsos : 1D_array_like
        single scattering contribution to rso.
    rsod : 1D_array_like
        multiple scattering contribution to rso.
    rddt : 1D_array_like
        surface bihemispherical reflectance factor.
    rsdt : 1D_array_like
        surface directional-hemispherical reflectance factor.
    rdot : 1D_array_like
        surface hemispherical-directional reflectance factor.
    rsodt : 1D_array_like
        reflectance factor.
    rsost : 1D_array_like
        reflectance factor.
    rsot : 1D_array_like
        surface bidirectional reflectance factor.
    gammasdf : 1D_array_like
        'Thermal gamma factor'.
    gammasdb : 1D_array_like
        'Thermal gamma factor'.
    gammaso : 1D_array_like
        'Thermal gamma factor'.
    Delta_tss : 2D_array_like
        Jacobian, beam transmittance in the sun-target path.
    Delta_too : 2D_array_like
        Jacobian, beam transmittance in the target-view path.
    Delta_tsstoo : 2D_array_like
        Jacobian, beam tranmittance in the sur-target-view path.
    Delta_rdd : 2D_array_like
        Jacobian, canopy bihemisperical reflectance factor.
    Delta_tdd : 2D_array_like
        Jacobian, canopy bihemishperical transmittance factor.
    Delta_rsd : 2D_array_like 
        Jacobian, canopy directional-hemispherical reflectance factor.
    Delta_tsd : 2D_array_like
        Jacobian, canopy directional-hemispherical transmittance factor.
    Delta_rdo : 2D_array_like
        Jacobian, canopy hemispherical-directional reflectance factor.
    Delta_tdo : 2D_array_like
        Jacobian, canopy hemispherical-directional transmittance factor.
    Delta_rso : 2D_array_like
        Jacobian, canopy bidirectional reflectance factor.
    Delta_rsos : 2D_array_like
        Jacobian, single scattering contribution to rso.
    Delta_rsod : 2D_array_like
        Jacobian, multiple scattering contribution to rso.
    Delta_rddt : 2D_array_like
        Jacobian, surface bihemispherical reflectance factor.
    Delta_rsdt : 2D_array_like
        Jacobian, surface directional-hemispherical reflectance factor.
    Delta_rdot : 2D_array_like
        Jacobian, surface hemispherical-directional reflectance factor.
    Delta_rsodt : 2D_array_like
        Jacobian, reflectance factor.
    Delta_rsost : 2D_array_like
        Jacobian, reflectance factor.
    Delta_rsot : 2D_array_like
        Jacobian, surface bidirectional reflectance factor.
    Delta_gammasdf : 2D_array_like
        Jacobian, 'Thermal gamma factor'.
    Delta_gammasdb : 2D_array_like
        Jacobian, 'Thermal gamma factor'.
    Delta_gammaso : 2D_array_like
        Jacobian, 'Thermal gamma factor'.
    
    References
    ----------
    .. [Verhoef2007] Verhoef, W.; Jia, Li; Qing Xiao; Su, Z., (2007) Unified Optical-Thermal
        Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies,
        IEEE Transactions on Geoscience and Remote Sensing, vol.45, no.6, pp.1808-1822,
        http://dx.doi.org/10.1109/TGRS.2007.895844 based on  in Verhoef et al. (2007).
    '''

    SAIL_params=3
    n_wl=rho.shape[0]

    # Get the leaf spectra parameters
    if not type(Delta_rho)==type(None):
        Delta_rho=np.asarray(Delta_rho)
        leaf_params=Delta_rho.shape[0]
    else:
        leaf_params=0
    n_params=leaf_params+SAIL_params
    rho_array=np.zeros((n_params,rho.shape[0]))
    tau_array=np.zeros((n_params,tau.shape[0]))
    for i in range(n_params):
        rho_array[i]=rho
        tau_array[i]=tau
    Delta_tau_array=np.zeros((n_params,tau.shape[0]))
    Delta_rho_array=np.zeros((n_params,rho.shape[0]))
    if not type(Delta_rho)==type(None):
        Delta_rho_array[:leaf_params]=Delta_rho
        Delta_tau_array[:leaf_params]=Delta_tau
        
        
    # Here the LIDF comes in
    lidf, Delta_lidf=lidf[0],lidf[1]
    # weighted_sum_over_lidf
    [ks, 
     ko, 
     bf, 
     sob, 
     sof, 
     Delta_ks, 
     Delta_ko, 
     Delta_bf, 
     Delta_sob, 
     Delta_sof] = Jac_weighted_sum_over_lidf(lidf, Delta_lidf, tts, tto, psi, leaf_params, n_params)

    # Geometric factors to be used later with rho and tau
    sdb=0.5*(ks+bf)
    Delta_sdb=0.5*(Delta_ks+Delta_bf)
    sdf=0.5*(ks-bf)
    Delta_sdf=0.5*(Delta_ks-Delta_bf)
    dob=0.5*(ko+bf)
    Delta_dob=0.5*(Delta_ko+Delta_bf)
    dof=0.5*(ko-bf)
    Delta_dof=0.5*(Delta_ko-Delta_bf)
    ddb=0.5*(1.+bf)
    Delta_ddb=0.5*Delta_bf
    ddf=0.5*(1.-bf)
    Delta_ddf=-0.5*Delta_bf
    # Here rho and tau come in
    Delta_sigb=np.zeros((n_params,n_wl))
    Delta_sigf=np.zeros((n_params,n_wl))
    sigb=ddb*rho+ddf*tau
    Delta_sigb=np.asarray(Delta_ddb.reshape(-1,1)*rho_array+ddb*Delta_rho_array+Delta_ddf.reshape(-1,1)*tau_array+ddf*Delta_tau_array)
    sigf=ddf*rho+ddb*tau
    Delta_sigf=np.asarray(Delta_ddf.reshape(-1,1)*rho_array+ddf*Delta_rho_array+Delta_ddb.reshape(-1,1)*tau_array+ddb*Delta_tau_array)
    if np.size(sigf)>1:
        sigf[sigf == 0.0]=1e-36
        sigb[sigb == 0.0]=1e-36
    else:
        sigf=max(1e-36,sigf)
        sigb=max(1e-36,sigb)
    att=1.-sigf
    Delta_att=-Delta_sigf
    m=np.sqrt(att**2.-sigb**2.)
    Delta_m=(0.5*(att**2.-sigb**2.)**-0.5)*(2*att*Delta_att-2*sigb*Delta_sigb)
    sb=sdb*rho+sdf*tau
    Delta_sb=np.asarray(Delta_sdb.reshape(-1,1)*rho_array+sdb*Delta_rho_array+Delta_sdf.reshape(-1,1)*tau_array+sdf*Delta_tau_array)
    sf=sdf*rho+sdb*tau
    Delta_sf=np.asarray(Delta_sdf.reshape(-1,1)*rho_array+sdf*Delta_rho_array+Delta_sdb.reshape(-1,1)*tau_array+sdb*Delta_tau_array)
    vb=dob*rho+dof*tau
    Delta_vb=np.asarray(Delta_dob.reshape(-1,1)*rho_array+dob*Delta_rho_array+Delta_dof.reshape(-1,1)*tau_array+dof*Delta_tau_array)
    vf=dof*rho+dob*tau
    Delta_vf=np.asarray(Delta_dof.reshape(-1,1)*rho_array+dof*Delta_rho_array+Delta_dob.reshape(-1,1)*tau_array+dob*Delta_tau_array)
    w =sob*rho+sof*tau
    Delta_w=np.asarray(Delta_sob.reshape(-1,1)*rho_array+sob*Delta_rho_array+Delta_sof.reshape(-1,1)*tau_array+sof*Delta_tau_array)
    # Here the LAI comes inleaf_params
    if lai<=0:
        tss,Delta_tss = 1,np.zeros(n_params)
        too,Delta_too = 1,np.zeros(n_params)
        tsstoo,Delta_tsstoo= 1,np.zeros(n_params)
        rdd,Delta_rdd= 0,np.zeros(n_params)
        tdd,Delta_tdd=1,np.zeros(n_params)
        rsd,Delta_rsd=0,np.zeros(n_params)
        tsd,Delta_tsd=0,np.zeros(n_params)
        rdo,Delta_rdo=0,np.zeros(n_params)
        tdo,Delta_tdo=0,np.zeros(n_params)
        rso,Delta_rso=0,np.zeros(n_params)
        rsos,Delta_rsos=0,np.zeros(n_params)
        rsod,Delta_rsod=0,np.zeros(n_params)
        rddt,Delta_rddt= rsoil,np.zeros((n_params,n_wl))
        rsdt,Delta_rsdt= rsoil,np.zeros((n_params,n_wl))
        rdot,Delta_rdot= rsoil,np.zeros((n_params,n_wl))
        rsodt,Delta_rsodt= 0,np.zeros(n_params)
        rsost,Delta_rsost= rsoil,np.zeros((n_params,n_wl))
        rsot,Delta_rsot= rsoil,np.zeros((n_params,n_wl))
        gammasdf,Delta_gammasdf=0,np.zeros(n_params)
        gammaso,Delta_gammaso=0,np.zeros(n_params)
        gammasdb,Delta_gammasdb=0,np.zeros(n_params)
        
        return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
            rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso,
            Delta_tss,Delta_too,Delta_tsstoo,Delta_rdd,Delta_tdd,Delta_rsd,Delta_tsd,Delta_rdo,Delta_tdo,
            Delta_rso,Delta_rsos,Delta_rsod,Delta_rddt,Delta_rsdt,Delta_rdot,
            Delta_rsodt,Delta_rsost,Delta_rsot,Delta_gammasdf,Delta_gammasdb,Delta_gammaso]
           
    e1=np.exp(-m*lai)
    Delta_e1= np.zeros((n_params,tau.shape[0]))
    Delta_e1[leaf_params]=np.exp(-m*lai)*(-m)
    Delta_e1[0:leaf_params]=np.exp(-m*lai)*(-lai*Delta_m[0:leaf_params])
    Delta_e1[leaf_params+1:]=np.exp(-m*lai)*(-lai*Delta_m[leaf_params+1:])
    e2=e1**2.
    Delta_e2=2*e1*Delta_e1
    rinf=(att-m)/sigb
    Delta_rinf=((Delta_att-Delta_m)*sigb-(att-m)*Delta_sigb)/sigb**2
    rinf2=rinf**2.
    Delta_rinf2=2*rinf*Delta_rinf
    re=rinf*e1
    Delta_re=Delta_rinf*e1+rinf*Delta_e1
    denom=1.-rinf2*e2
    Delta_denom=-(Delta_rinf2*e2+rinf2*Delta_e2)
    
    J1ks, Delta_J1ks = JacJfunc1(ks, m, lai, Delta_ks, Delta_m)
    J2ks, Delta_J2ks = JacJfunc2(ks, m, lai, Delta_ks, Delta_m)
    J1ko, Delta_J1ko = JacJfunc1(ko, m, lai, Delta_ko, Delta_m)
    J2ko, Delta_J2ko = JacJfunc2(ko, m, lai, Delta_ko, Delta_m)
    
    Pss=(sf+sb*rinf)*J1ks
    Delta_Pss=(Delta_sf+Delta_sb*rinf+sb*Delta_rinf)*J1ks+(sf+sb*rinf)*Delta_J1ks
    Qss=(sf*rinf+sb)*J2ks
    Delta_Qss=((Delta_sf*rinf+sf*Delta_rinf)+Delta_sb)*J2ks+(sf*rinf+sb)*Delta_J2ks
    Pv=(vf+vb*rinf)*J1ko
    Delta_Pv=(Delta_vf+Delta_vb*rinf+vb*Delta_rinf)*J1ko+(vf+vb*rinf)*Delta_J1ko
    Qv=(vf*rinf+vb)*J2ko
    Delta_Qv=(Delta_vf*rinf+vf*Delta_rinf+Delta_vb)*J2ko+(vf*rinf+vb)*Delta_J2ko
    tdd=(1.-rinf2)*e1/denom
    Delta_tdd=((-Delta_rinf2*e1+(1.-rinf2)*Delta_e1)*denom-(1.-rinf2)*e1*Delta_denom)/denom**2
    rdd=rinf*(1.-e2)/denom
    Delta_rdd=((Delta_rinf*(1.-e2)-rinf*Delta_e2)*denom-rinf*(1.-e2)*Delta_denom)/denom**2
    tsd=(Pss-re*Qss)/denom
    Delta_tsd=((Delta_Pss-(Delta_re*Qss+re*Delta_Qss))*denom-(Pss-re*Qss)*Delta_denom)/denom**2
    rsd=(Qss-re*Pss)/denom
    Delta_rsd=((Delta_Qss-(Delta_re*Pss+re*Delta_Pss))*denom-(Qss-re*Pss)*Delta_denom)/denom**2
    tdo=(Pv-re*Qv)/denom
    Delta_tdo=((Delta_Pv-(Delta_re*Qv+re*Delta_Qv))*denom-(Pv-re*Qv)*Delta_denom)/denom**2
    rdo=(Qv-re*Pv)/denom
    Delta_rdo=((Delta_Qv-(Delta_re*Pv+re*Delta_Pv))*denom-(Qv-re*Pv)*Delta_denom)/denom**2
    
    # Thermal "sd" quantities
    gammasdf=(1.+rinf)*(J1ks-re*J2ks)/denom
    Delta_gammasdf=((Delta_rinf*(J1ks-re*J2ks)+(1.+rinf)*(Delta_J1ks-(Delta_re*J2ks+re*Delta_J2ks)))*denom-(1.+rinf)*(J1ks-re*J2ks)*Delta_denom)/denom**2
    gammasdb=(1.+rinf)*(-re*J1ks+J2ks)/denom
    Delta_gammasdb=((Delta_rinf*(-re*J1ks+J2ks)+(1.+rinf)*(-Delta_re*J1ks-re*Delta_J1ks+Delta_J2ks))*denom-(1.+rinf)*(-re*J1ks+J2ks)*Delta_denom)/denom**2
    tss=np.exp(-ks*lai)
    Delta_tss=np.zeros(Delta_ks.shape)
    Delta_tss[0:leaf_params]=np.exp(-ks*lai)*(-Delta_ks[0:leaf_params]*lai)
    Delta_tss[leaf_params]=np.exp(-ks*lai)*(-ks)    
    Delta_tss[leaf_params+1:]=np.exp(-ks*lai)*(-Delta_ks[leaf_params+1:]*lai)
    too=np.exp(-ko*lai)
    Delta_too=np.zeros(Delta_ko.shape)
    Delta_too[0:leaf_params]=np.exp(-ko*lai)*(-Delta_ko[0:leaf_params]*lai)
    Delta_too[leaf_params]=np.exp(-ko*lai)*(-ko)    
    Delta_too[leaf_params+1:]=np.exp(-ko*lai)*(-Delta_ko[leaf_params+1:]*lai)
    z, Delta_z = JacJfunc2(ks, ko, lai, Delta_ks, Delta_ko)
    g1=(z-J1ks*too)/(ko+m)
    Delta_g1=((Delta_z.reshape(-1,1)-(Delta_J1ks*too+J1ks*Delta_too.reshape(-1,1)))*(ko+m)-(z-J1ks*too)*(Delta_ko.reshape(-1,1)+Delta_m))/(ko+m)**2
    g2=(z-J1ko*tss)/(ks+m)
    Delta_g2=((Delta_z.reshape(-1,1)-(Delta_J1ko*tss+J1ko*Delta_tss.reshape(-1,1)))*(ks+m)-(z-J1ko*tss)*(Delta_ks.reshape(-1,1)+Delta_m))/(ks+m)**2
    Tv1=(vf*rinf+vb)*g1
    Delta_Tv1=(Delta_vf*rinf+vf*Delta_rinf+Delta_vb)*g1+(vf*rinf+vb)*Delta_g1
    Tv2=(vf+vb*rinf)*g2
    Delta_Tv2=(Delta_vf+Delta_vb*rinf+vb*Delta_rinf)*g2+(vf+vb*rinf)*Delta_g2
    T1=Tv1*(sf+sb*rinf)
    Delta_T1=Delta_Tv1*(sf+sb*rinf)+Tv1*(Delta_sf+Delta_sb*rinf+sb*Delta_rinf)
    T2=Tv2*(sf*rinf+sb)
    Delta_T2=Delta_Tv2*(sf*rinf+sb)+Tv2*(Delta_sf*rinf+sf*Delta_rinf+Delta_sb)
    T3=(rdo*Qss+tdo*Pss)*rinf
    Delta_T3=(Delta_rdo*Qss+rdo*Delta_Qss+Delta_tdo*Pss+tdo*Delta_Pss)*rinf+(rdo*Qss+tdo*Pss)*Delta_rinf
    
    # Multiple scattering contribution to bidirectional canopy reflectance
    rsod=(T1+T2-T3)/(1.-rinf2)
    Delta_rsod=((Delta_T1+Delta_T2-Delta_T3)*(1.-rinf2)+(T1+T2-T3)*Delta_rinf2)/(1.-rinf2)**2
    
    # Thermal "sod" quantity
    T4=Tv1*(1.+rinf)
    Delta_T4=Delta_Tv1*(1.+rinf)+Tv1*Delta_rinf
    T5=Tv2*(1.+rinf)
    Delta_T5=Delta_Tv2*(1.+rinf)+Tv2*Delta_rinf
    T6=(rdo*J2ks+tdo*J1ks)*(1.+rinf)*rinf
    Delta_T6=(Delta_rdo*J2ks+rdo*Delta_J2ks+Delta_tdo*J1ks+tdo*Delta_J1ks)*(1.+rinf)*rinf+(rdo*J2ks+tdo*J1ks)*(Delta_rinf*rinf+(1.+rinf)*Delta_rinf)
    gammasod=(T4+T5-T6)/(1.-rinf2)
    Delta_gammasod=((Delta_T4+Delta_T5-Delta_T6)*(1.-rinf2)+(T4+T5-T6)*Delta_rinf2)/(1.-rinf2)**2
    
    # Hotspot effect
    dso = define_geometric_constant (tts, tto, psi)
    [tsstoo, 
     sumint, 
     Delta_tsstoo, 
     Delta_sumint] = Jac_hotspot_calculations(hotspot, 
                                                lai, 
                                                ko, 
                                                ks, 
                                                dso, 
                                                tss, 
                                                Delta_ko, 
                                                Delta_ks, 
                                                Delta_tss, 
                                                leaf_params, 
                                                n_params)

    # Bidirectional reflectance
    # Single scattering contribution
    rsos=w*lai*sumint
    Delta_rsos=np.zeros(Delta_w.shape)
    Delta_rsos[:leaf_params]=lai*(Delta_w[:leaf_params]*sumint+w*Delta_sumint[:leaf_params].reshape(-1,1))
    Delta_rsos[leaf_params]=(Delta_w[leaf_params]*sumint+w*Delta_sumint[leaf_params].reshape(-1,1))*lai+(w*sumint)
    Delta_rsos[leaf_params+1:]=lai*(Delta_w[leaf_params+1:]*sumint+w*Delta_sumint[leaf_params+1:].reshape(-1,1))
    gammasos=ko*lai*sumint
    Delta_gammasos=np.zeros(Delta_ko.shape)
    Delta_gammasos[leaf_params]=(Delta_ko[leaf_params]*sumint+ko*Delta_sumint[leaf_params])*lai+(ko*sumint)
    Delta_gammasos[leaf_params+1:]=lai*(Delta_ko[leaf_params+1:]*sumint+ko*Delta_sumint[leaf_params+1:])
    # Total canopy contribution
    rso=rsos+rsod
    Delta_rso=Delta_rsos+Delta_rsod
    gammaso=gammasos+gammasod
    Delta_gammaso=Delta_gammasos.reshape(-1,1)+Delta_gammasod
    #Interaction with the soil
    dn=1.-rsoil*rdd
    Delta_dn=-rsoil*Delta_rdd
    if np.size(dn)>1:
        dn[dn < 1e-36]=1e-36
    else:
        dn=max(1e-36,dn)
    rddt=rdd+tdd*rsoil*tdd/dn
    Delta_rddt=Delta_rdd+(rsoil*2*tdd*Delta_tdd*dn-tdd*rsoil*tdd*Delta_dn)/dn**2
    rsdt=rsd+(tsd+tss)*rsoil*tdd/dn
    Delta_rsdt=Delta_rsd+rsoil*((Delta_tdd*(tsd+tss)+tdd*(Delta_tsd+Delta_tss.reshape(-1,1)))*dn-tdd*(tsd+tss)*Delta_dn)/dn**2
    rdot=rdo+tdd*rsoil*(tdo+too)/dn
    Delta_rdot=Delta_rdo+rsoil*((Delta_tdd*(tdo+too)+tdd*(Delta_tdo+Delta_too.reshape(-1,1)))*dn-tdd*(tdo+too)*Delta_dn)/dn**2
    rsodt=((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*rsoil/dn
    Delta_rsodt=rsoil*(((Delta_tss.reshape(-1,1)+Delta_tsd)*tdo+(tss+tsd)
        *Delta_tdo+(Delta_tsd+rsoil*(Delta_tss.reshape(-1,1)*rdd+tss*Delta_rdd))
        *too+(tsd+tss*rsoil*rdd)*Delta_too.reshape(-1,1))*dn
        -((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*Delta_dn)/dn**2
    rsost=rso+tsstoo*rsoil
    Delta_rsost=Delta_rso+rsoil*Delta_tsstoo.reshape(-1,1)
    rsot=rsost+rsodt
    Delta_rsot=Delta_rsost+Delta_rsodt

    return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
            rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso,
            Delta_tss,Delta_too,Delta_tsstoo,Delta_rdd,Delta_tdd,Delta_rsd,Delta_tsd,Delta_rdo,Delta_tdo,
            Delta_rso,Delta_rsos,Delta_rsod,Delta_rddt,Delta_rsdt,Delta_rdot,
            Delta_rsodt,Delta_rsost,Delta_rsot,Delta_gammasdf,Delta_gammasdb,Delta_gammaso]

def Jac_weighted_sum_over_lidf(lidf, Delta_lidf, tts, tto, psi, leaf_params, n_params):
    
    cts   = np.cos(np.radians(tts))
    cto   = np.cos(np.radians(tto))
    ctscto  = cts*cto

    #Initialise sums
    ks=0.
    Delta_ks=np.zeros(n_params)
    ko=0.
    Delta_ko=np.zeros(n_params)
    bf=0.
    Delta_bf=np.zeros(n_params)    
    sob=0.
    Delta_sob=np.zeros(n_params)
    sof=0.
    Delta_sof=np.zeros(n_params)
    # Weighted sums over LIDF
    n_angles=len(lidf)
    angle_step=float(90.0/n_angles)
    litab=[float(angle)*angle_step+(angle_step/2.0) for angle in range(n_angles)]
    for i,ili in enumerate(litab):
        ttl=float(ili)
        cttl=np.cos(np.radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        chi_s, chi_o, frho, ftau = volscatt(tts, tto, psi, ttl)
        # Extinction coefficients
        ksli=chi_s/cts
        koli=chi_o/cto
        # Area scattering coefficient fractions
        sobli=frho*np.pi/ctscto
        sofli=ftau*np.pi/ctscto
        bfli=cttl**2.
        ks+=ksli*float(lidf[i])
        Delta_ks[leaf_params+2]+=ksli*float(Delta_lidf[i])
        ko+=koli*float(lidf[i])
        Delta_ko[leaf_params+2]+=koli*float(Delta_lidf[i])
        bf+=bfli*float(lidf[i])
        Delta_bf[leaf_params+2]+=bfli*float(Delta_lidf[i])
        sob+=sobli*float(lidf[i])
        Delta_sob[leaf_params+2]+=sobli*float(Delta_lidf[i])
        sof+=sofli*float(lidf[i])
        Delta_sof[leaf_params+2]+=sofli*float(Delta_lidf[i])
    
    return ks, ko, bf, sob, sof, Delta_ks, Delta_ko, Delta_bf, Delta_sob, Delta_sof

def Jac_hotspot_calculations(hotspot, 
                             lai, 
                             ko, 
                             ks, 
                             dso, 
                             tss, 
                             Delta_ko, 
                             Delta_ks, 
                             Delta_tss,
                             leaf_params,
                             n_params):

    #Treatment of the hotspot-effect
    alf=1e36
    Delta_alf=np.zeros(n_params)
    Delta_sumint=np.zeros(n_params)
    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0. : 
        alf=(dso/hotspot)*2./(ks+ko)
        Delta_alf[:leaf_params+1]=-2*(dso/hotspot)*(Delta_ks[:leaf_params+1]+Delta_ko[:leaf_params+1])/(ks+ko)**2
        Delta_alf[leaf_params+1]=-(2*dso/(ks+ko))/hotspot**2
        Delta_alf[leaf_params+2:]=-2*(dso/hotspot)*(Delta_ks[leaf_params+2:]+Delta_ko[leaf_params+2:])/(ks+ko)**2
        
    if alf == 0. : 
        # The pure hotspot
        tsstoo=tss
        Delta_tsstoo=np.array(Delta_tss)
        sumint=(1.-tss)/(ks*lai)
        Delta_sumint[leaf_params]=(-Delta_tss[leaf_params]*(ks*lai)-(1.-tss)*ks)/(ks*lai)**2
        Delta_sumint[leaf_params+2]=(-Delta_tss[leaf_params+2]*(ks*lai)-(1.-tss)*(lai*Delta_ks[leaf_params+2]))/(ks*lai)**2
        
    else :
        # Outside the hotspot
        fhot=lai*np.sqrt(ko*ks)
        Delta_fhot=np.zeros(n_params)
        Delta_fhot[leaf_params]=np.sqrt(ko*ks)
        Delta_fhot[leaf_params+2]=lai*0.5*(ko*ks)**-0.5*(Delta_ko[leaf_params+2]*ks+ko*Delta_ks[leaf_params+2])
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1=0.
        Delta_x1=np.zeros(n_params)
        y1=0.
        Delta_y1=np.zeros(n_params)
        f1=1.
        Delta_f1=np.zeros(n_params)
        fint=(1.-np.exp(-alf))*.05
        Delta_fint=0.05*np.exp(-alf)*Delta_alf
        sumint=0.
        Delta_sumint=0.
        for istep in range(1,21):
            if istep < 20 :
                x2=-np.log(1.-istep*fint)/alf
                Delta_x2=-(((1./(1.-istep*fint))*(-istep*Delta_fint))*alf-np.log(1.-istep*fint)*Delta_alf)/alf**2
            else :
                x2=1.
                Delta_x2=np.zeros(n_params)
            y2=-(ko+ks)*lai*x2+fhot*(1.-np.exp(-alf*x2))/alf
            Delta_y2=np.zeros(n_params)
            Delta_y2[leaf_params]=-(ko+ks)*x2+Delta_fhot[leaf_params]*(1.-np.exp(-alf*x2))/alf
            Delta_y2[leaf_params+1:]=-lai*((Delta_ko[leaf_params+1:]+Delta_ks[leaf_params+1:])
                *x2+(ko+ks)*Delta_x2[leaf_params+1:])+((Delta_fhot[leaf_params+1:]*(1.-np.exp(-alf*x2))
                -fhot*np.exp(-alf*x2)*(-Delta_alf[leaf_params+1:]*x2-alf*Delta_x2[leaf_params+1:]))
                *alf-fhot*(1.-np.exp(-alf*x2))*Delta_alf[leaf_params+1:])/alf**2
            f2=np.exp(y2)
            Delta_f2=np.exp(y2)*Delta_y2
            sumint+=(f2-f1)*(x2-x1)/(y2-y1)
            Delta_sumint+=(((Delta_f2-Delta_f1)*(x2-x1)+(f2-f1)*(Delta_x2-Delta_x1))*(y2-y1)-(f2-f1)*(x2-x1)*(Delta_y2-Delta_y1))/(y2-y1)**2
            x1=float(x2)
            Delta_x1=np.array(Delta_x2)
            y1=float(y2)
            Delta_y1=np.array(Delta_y2)
            f1=float(f2)
            Delta_f1=np.array(Delta_f2)

        tsstoo=float(f1)
        Delta_tsstoo=np.array(Delta_f1)
    if np.isnan(sumint) : 
        sumint=0.
        Delta_sumint = np.zeros
        
    return tsstoo, sumint, Delta_tsstoo, Delta_sumint
             
def volscatt(tts,tto,psi,ttl) :
    '''Compute volume scattering functions and interception coefficients
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
    '''

    cts=np.cos(np.radians(tts))
    cto=np.cos(np.radians(tto))
    sts=np.sin(np.radians(tts))
    sto=np.sin(np.radians(tto))
    cospsi=np.cos(np.radians(psi))
    psir=np.radians(psi)
    cttl=np.cos(np.radians(ttl))
    sttl=np.sin(np.radians(ttl))
    cs=cttl*cts
    co=cttl*cto
    ss=sttl*sts
    so=sttl*sto  
    cosbts=5.
    if abs(ss) > 1e-6 : cosbts=-cs/ss
    cosbto=5.
    if abs(so) > 1e-6 : cosbto=-co/so
    if abs(cosbts) < 1.0:
        bts=np.arccos(cosbts)
        ds=ss
    else:
        bts=np.pi
        ds=cs
    chi_s=2./np.pi*((bts-np.pi*0.5)*cs+np.sin(bts)*ss)
    if abs(cosbto) < 1.0:
        bto=np.arccos(cosbto)
        do_=so
    else:
        if tto < 90.:
            bto=np.pi
            do_=co
        else:
            bto=0.0
            do_=-co
    chi_o=2.0/np.pi*((bto-np.pi*0.5)*co+np.sin(bto)*so)
    btran1=abs(bts-bto)
    btran2=np.pi-abs(bts+bto-np.pi)
    if psir <= btran1:
        bt1=psir
        bt2=btran1
        bt3=btran2
    else:
        bt1=btran1
        if psir <= btran2:
            bt2=psir
            bt3=btran2
        else:
            bt2=btran2
            bt3=psir
    t1=2.*cs*co+ss*so*cospsi
    t2=0.
    if bt2 > 0.: t2=np.sin(bt2)*(2.*ds*do_+ss*so*np.cos(bt1)*np.cos(bt3))
    denom=2.*np.pi**2
    frho=((np.pi-bt2)*t1+t2)/denom
    ftau=(-bt2*t1+t2)/denom
    if frho < 0. : frho=0.
    if ftau < 0. : ftau=0.
   
    return chi_s, chi_o, frho, ftau    

def JacJfunc1(k,l,t,Delta_k,Delta_l):
    ''' J1 function with avoidance of singularity problem.'''

    nb=np.size(l)
    del_=(k-l)*t
    Delta_del_=np.zeros(Delta_l.shape)
    if len(Delta_del_.shape)==2:
        Delta_k=Delta_k.reshape(-1,1)

    Delta_del_[0:-3]=(Delta_k[0:-3]-Delta_l[0:-3])*t
    Delta_del_[-3]=(k-l)+(Delta_k[-3]-Delta_l[-3])*t
    Delta_del_[-2:]=(Delta_k[-2:]-Delta_l[-2:])*t
    
    result=np.zeros(nb)
    index=abs(del_)>= 1e-1
    Delta_result=np.zeros(Delta_l.shape)
    result[index]=(np.exp(-l[index]*t)-np.exp(-k*t))/(k-l[index])

    Delta_result[0:-3]=((np.exp(-l*t)*(-t
        *Delta_l[0:-3])-np.exp(-k*t)*(-t*Delta_k[0:-3]))
        *(k-l)-(np.exp(-l*t)-np.exp(-k*t))*(Delta_k[0:-3]
        -Delta_l[0:-3]))/(k-l)**2
    Delta_result[-3]=(np.exp(-l*t)*(-l)-np.exp(-k*t)*(-k))/(k-l)
    Delta_result[-3+1:]=((np.exp(-l*t)*(-t
        *Delta_l[-3+1:])-np.exp(-k*t)*(-t*Delta_k[-3+1:]))
        *(k-l)-(np.exp(-l*t)-np.exp(-k*t))*(Delta_k[-3+1:]
        -Delta_l[-3+1:]))/(k-l)**2
    result[~index]=0.5*t*(np.exp(-k*t)+np.exp(-l[~index]*t))*(1.-(del_[~index]**2.)/12.)

    return result, Delta_result

def JacJfunc2(k,l,t,Delta_k,Delta_l) :
    '''J2 function.'''

    result=(1.-np.exp(-(k+l)*t))/(k+l)
    Delta_result=np.zeros(Delta_l.shape)
    if len(Delta_result.shape)==2:
        Delta_k=Delta_k.reshape(-1,1)
    
    Delta_result[0:-3]=(-np.exp(-(k+l)*t)*-t*(Delta_k[0:-3]+Delta_l[0:-3])*(k+l)-(1.-np.exp(-(k+l)*t))*(Delta_k[0:-3]+Delta_l[0:-3]))/(k+l)**2
    Delta_result[-3]=np.exp(-(k+l)*t)*(k+l)/(k+l)
    Delta_result[-2:]=(-np.exp(-(k+l)*t)*-t*(Delta_k[-2:]+Delta_l[-2:])*(k+l)-(1.-np.exp(-(k+l)*t))*(Delta_k[-2:]+Delta_l[-2:]))/(k+l)**2
    
    return result, Delta_result

def define_geometric_constant (tts, tto, psi):
    tants = np.tan(np.radians(tts))
    tanto = np.tan(np.radians(tto))
    cospsi  = np.cos(np.radians(psi))
    dso = np.sqrt(tants**2.+tanto**2.-2.*tants*tanto*cospsi)
    return dso
