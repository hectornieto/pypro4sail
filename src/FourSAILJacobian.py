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
* :func:`FourSAIL` Runs 4SAIL canopy radiative transfer model.
* :func:`FourSAIL_wl` Runs 4SAIL canopy radiative transfer model for a specific wavelenght, aimed for computing speed.

Ancillary functions
-------------------
* :func:`CalcLIDF_Verhoef` Calculate the Leaf Inclination Distribution Function based on the [Verhoef1998] bimodal LIDF distribution.
* :func:`CalcLIDF_Campbell` Calculate the Leaf Inclination Distribution Function based on the [Campbell1990] ellipsoidal LIDF distribution.
* :func:`volscatt` Colume scattering functions and interception coefficients.
* :func:`Jfunc1` J1 function with avoidance of singularity problem.
* :func:`Jfunc1_wl` J1 function with avoidance of singularity problem for :func:`FourSAIL_wl`.
* :func:`Jfunc2` J2 function with avoidance of singularity problem.
* :func:`Jfunc2_wl` J2 function with avoidance of singularity problem for :func:`FourSAIL_wl`.
* :func;`Get_SunAngles` Calculate the Sun Zenith and Azimuth Angles.

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

def CalcLIDF_Verhoef(a,b,n_elements=18):
    '''Calculate the Leaf Inclination Distribution Function based on the 
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
        '''

    import math as m
    freq=1.0
    step=90.0/n_elements
    lidf=[]
    angles=[i*step for i in reversed(range(n_elements))]
    for angle in angles:
        tl1=m.radians(angle)
        if a>1.0:
            f = 1.0-m.cos(tl1)
        else:
            eps=1e-8
            delx=1.0
            x=2.0*tl1
            p=float(x)
            while delx >= eps:
                y = a*m.sin(x)+.5*b*m.sin(2.*x)
                dx=.5*(y-x+p)
                x=x+dx
                delx=abs(dx)
            f = (2.*y+p)/m.pi
        freq=freq-f
        lidf.append(freq)
        freq=float(f)
    lidf=list(reversed(lidf))
    return  lidf
    
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
    '''
    
    import numpy as np
    
    alpha=float(alpha)
    excent=np.exp(-1.6184e-5*alpha**3.+2.1145e-3*alpha**2.-1.2390e-1*alpha+3.2491)
    Delta_excent=np.exp(-1.6184e-5*alpha**3.+2.1145e-3*alpha**2.-1.2390e-1*alpha+3.2491)*(3*-1.6184e-5*alpha**2+
                    2.*2.1145e-3*alpha-1.2390e-1)
    print(excent)
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
    return Delta_lidf,lidf

def FourSAIL(lai,hotspot,lidf,tts,tto,psi,rho,tau,rsoil):
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
    '''

    from numpy import cos, tan, radians, pi, sqrt, log, exp, isnan, size
    cts   = cos(radians(tts))
    cto   = cos(radians(tto))
    ctscto  = cts*cto
    #sts   = sin(radians(tts))
    #sto   = sin(radians(tto))
    tants = tan(radians(tts))
    tanto = tan(radians(tto))
    cospsi  = cos(radians(psi))
    dso = sqrt(tants**2.+tanto**2.-2.*tants*tanto*cospsi)
    #Calculate geometric factors associated with extinction and scattering 
    #Initialise sums
    ks=0.
    ko=0.
    bf=0.
    sob=0.
    sof=0.
    # Weighted sums over LIDF
    n_angles=len(lidf)
    angle_step=float(90.0/n_angles)
    litab=[float(angle)*angle_step+(angle_step/2.0) for angle in range(n_angles)]
    i=0
    for ili in litab:
        ttl=float(ili)
        cttl=cos(radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s,chi_o,frho,ftau]=volscatt(tts,tto,psi,ttl)
        # Extinction coefficients
        ksli=chi_s/cts
        koli=chi_o/cto
        # Area scattering coefficient fractions
        sobli=frho*pi/ctscto
        sofli=ftau*pi/ctscto
        bfli=cttl**2.
        ks=ks+ksli*float(lidf[i])
        ko=ko+koli*float(lidf[i])
        bf=bf+bfli*float(lidf[i])
        sob=sob+sobli*float(lidf[i])
        sof=sof+sofli*float(lidf[i])
        i=i+1
    # Geometric factors to be used later with rho and tau
    sdb=0.5*(ks+bf)
    sdf=0.5*(ks-bf)
    dob=0.5*(ko+bf)
    dof=0.5*(ko-bf)
    ddb=0.5*(1.+bf)
    ddf=0.5*(1.-bf)
    # Here rho and tau come in
    sigb=ddb*rho+ddf*tau
    sigf=ddf*rho+ddb*tau
    if size(sigf)>1:
        sigf[sigf == 0.0]=1e-36
        sigb[sigb == 0.0]=1e-36
    else:
        sigf=max(1e-36,sigf)
        sigb=max(1e-36,sigb)
    att=1.-sigf
    m=sqrt(att**2.-sigb**2.)
    sb=sdb*rho+sdf*tau
    sf=sdf*rho+sdb*tau
    vb=dob*rho+dof*tau
    vf=dof*rho+dob*tau
    w =sob*rho+sof*tau
    # Here the LAI comes in
    if lai<=0:
        tss = 1
        too= 1
        tsstoo= 1
        rdd= 0
        tdd=1
        rsd=0
        tsd=0
        rdo=0
        tdo=0
        rso=0
        rsos=0
        rsod=0
        rddt= rsoil
        rsdt= rsoil
        rdot= rsoil
        rsodt= 0
        rsost= rsoil
        rsot= rsoil
        gammasdf=0
        gammaso=0
        gammasdb=0
        
        return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
            rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]
            
    e1=exp(-m*lai)
    e2=e1**2.
    rinf=(att-m)/sigb
    rinf2=rinf**2.
    re=rinf*e1
    denom=1.-rinf2*e2
    J1ks=Jfunc1(ks,m,lai)
    J2ks=Jfunc2(ks,m,lai)
    J1ko=Jfunc1(ko,m,lai)
    J2ko=Jfunc2(ko,m,lai)
    Pss=(sf+sb*rinf)*J1ks
    Qss=(sf*rinf+sb)*J2ks
    Pv=(vf+vb*rinf)*J1ko
    Qv=(vf*rinf+vb)*J2ko
    tdd=(1.-rinf2)*e1/denom
    rdd=rinf*(1.-e2)/denom
    tsd=(Pss-re*Qss)/denom
    rsd=(Qss-re*Pss)/denom
    tdo=(Pv-re*Qv)/denom
    rdo=(Qv-re*Pv)/denom
    # Thermal "sd" quantities
    gammasdf=(1.+rinf)*(J1ks-re*J2ks)/denom
    gammasdb=(1.+rinf)*(-re*J1ks+J2ks)/denom
    tss=exp(-ks*lai)
    too=exp(-ko*lai)
    z=Jfunc2(ks,ko,lai)
    g1=(z-J1ks*too)/(ko+m)
    g2=(z-J1ko*tss)/(ks+m)
    Tv1=(vf*rinf+vb)*g1
    Tv2=(vf+vb*rinf)*g2
    T1=Tv1*(sf+sb*rinf)
    T2=Tv2*(sf*rinf+sb)
    T3=(rdo*Qss+tdo*Pss)*rinf
    # Multiple scattering contribution to bidirectional canopy reflectance
    rsod=(T1+T2-T3)/(1.-rinf2)
    # Thermal "sod" quantity
    T4=Tv1*(1.+rinf)
    T5=Tv2*(1.+rinf)
    T6=(rdo*J2ks+tdo*J1ks)*(1.+rinf)*rinf
    gammasod=(T4+T5-T6)/(1.-rinf2)
    #Treatment of the hotspot-effect
    alf=1e36
    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0. : alf=(dso/hotspot)*2./(ks+ko)
    if alf == 0. : 
        # The pure hotspot
        tsstoo=tss
        sumint=(1.-tss)/(ks*lai)
    else :
        # Outside the hotspot
        fhot=lai*sqrt(ko*ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1=0.
        y1=0.
        f1=1.
        fint=(1.-exp(-alf))*.05
        sumint=0.
        for istep in range(1,21):
            if istep < 20 :
                x2=-log(1.-istep*fint)/alf
            else :
                x2=1.
            y2=-(ko+ks)*lai*x2+fhot*(1.-exp(-alf*x2))/alf
            f2=exp(y2)
            sumint=sumint+(f2-f1)*(x2-x1)/(y2-y1)
            x1=x2
            y1=y2
            f1=f2
        tsstoo=f1
    if isnan(sumint) : sumint=0.
    # Bidirectional reflectance
    # Single scattering contribution
    rsos=w*lai*sumint
    gammasos=ko*lai*sumint
    # Total canopy contribution
    rso=rsos+rsod
    gammaso=gammasos+gammasod
    #Interaction with the soil
    dn=1.-rsoil*rdd
    if size(dn)>1:
        dn[dn < 1e-36]=1e-36
    else:
        dn=max(1e-36,dn)
    rddt=rdd+tdd*rsoil*tdd/dn
    rsdt=rsd+(tsd+tss)*rsoil*tdd/dn
    rdot=rdo+tdd*rsoil*(tdo+too)/dn
    rsodt=((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*rsoil/dn
    rsost=rso+tsstoo*rsoil
    rsot=rsost+rsodt
    
    return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
          rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]

def FourSAIL_wl(lai,hotspot,lidf,tts,tto,psi,rho,tau,rsoil):
    '''Runs 4SAIL canopy radiative transfer model for a single wavelenght.
    
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
    '''
    from math import cos, tan, radians, pi, sqrt, log, exp, isnan
    cts   = cos(radians(tts))
    cto   = cos(radians(tto))
    ctscto  = cts*cto
    tants = tan(radians(tts))
    tanto = tan(radians(tto))
    cospsi  = cos(radians(psi))
    dso = sqrt(tants**2.+tanto**2.-2.*tants*tanto*cospsi)
    #Calculate geometric factors associated with extinction and scattering 
    #Initialise sums
    ks=0.
    ko=0.
    bf=0.
    sob=0.
    sof=0.
    # Weighted sums over LIDF
    n_angles=len(lidf)
    angle_step=float(90.0/n_angles)
    litab=[float(angle)*angle_step+(angle_step/2.0) for angle in range(n_angles)]
    i=0
    for ili in litab:
        ttl=float(ili)
        cttl=cos(radians(ttl))
        # SAIL volume scattering phase function gives interception and portions to be multiplied by rho and tau
        [chi_s,chi_o,frho,ftau]=volscatt(tts,tto,psi,ttl)
        # Extinction coefficients
        ksli=chi_s/cts
        koli=chi_o/cto
        # Area scattering coefficient fractions
        sobli=frho*pi/ctscto
        sofli=ftau*pi/ctscto
        bfli=cttl**2.
        ks=ks+ksli*float(lidf[i])
        ko=ko+koli*float(lidf[i])
        bf=bf+bfli*float(lidf[i])
        sob=sob+sobli*float(lidf[i])
        sof=sof+sofli*float(lidf[i])
        i=i+1
    # Geometric factors to be used later with rho and tau
    sdb=0.5*(ks+bf)
    sdf=0.5*(ks-bf)
    dob=0.5*(ko+bf)
    dof=0.5*(ko-bf)
    ddb=0.5*(1.+bf)
    ddf=0.5*(1.-bf)
    # Here rho and tau come in
    sigb=ddb*rho+ddf*tau
    sigf=ddf*rho+ddb*tau
    att=1.-sigf
    try:
        m=sqrt(att**2-sigb**2)
    except:
        m=0.0
    sb=sdb*rho+sdf*tau
    sf=sdf*rho+sdb*tau
    vb=dob*rho+dof*tau
    vf=dof*rho+dob*tau
    w =sob*rho+sof*tau
    # Here the LAI comes in
    if lai<=0:
        tss = 1
        too= 1
        tsstoo= 1
        rdd= 0
        tdd=1
        rsd=0
        tsd=0
        rdo=0
        tdo=0
        rso=0
        rsos=0
        rsod=0
        rddt= rsoil
        rsdt= rsoil
        rdot= rsoil
        rsodt= 0
        rsost= rsoil
        rsot= rsoil
        gammasdf=0
        gammaso=0
        gammasdb=0
        
        return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
            rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]

    e1=exp(-m*lai)
    e2=e1**2.
    try:
        rinf=(att-m)/sigb
    except:
        rinf=1e36
    rinf2=rinf**2.
    re=rinf*e1
    denom=1.-rinf2*e2
    if denom < 1e-36: denom=1e-36
    J1ks=Jfunc1_wl(ks,m,lai)
    J2ks=Jfunc2_wl(ks,m,lai)
    J1ko=Jfunc1_wl(ko,m,lai)
    J2ko=Jfunc2_wl(ko,m,lai)
    Pss=(sf+sb*rinf)*J1ks
    Qss=(sf*rinf+sb)*J2ks
    Pv=(vf+vb*rinf)*J1ko
    Qv=(vf*rinf+vb)*J2ko
    tdd=(1.-rinf2)*e1/denom
    rdd=rinf*(1.-e2)/denom
    tsd=(Pss-re*Qss)/denom
    rsd=(Qss-re*Pss)/denom
    tdo=(Pv-re*Qv)/denom
    rdo=(Qv-re*Pv)/denom
    # Thermal "sd" quantities
    gammasdf=(1.+rinf)*(J1ks-re*J2ks)/denom
    gammasdb=(1.+rinf)*(-re*J1ks+J2ks)/denom
    tss=exp(-ks*lai)
    too=exp(-ko*lai)
    z=Jfunc2_wl(ks,ko,lai)
    g1=(z-J1ks*too)/(ko+m)
    g2=(z-J1ko*tss)/(ks+m)
    Tv1=(vf*rinf+vb)*g1
    Tv2=(vf+vb*rinf)*g2
    T1=Tv1*(sf+sb*rinf)
    T2=Tv2*(sf*rinf+sb)
    T3=(rdo*Qss+tdo*Pss)*rinf
    # Multiple scattering contribution to bidirectional canopy reflectance
    if rinf2>=1:rinf2=1-1e-16
    rsod=(T1+T2-T3)/(1.-rinf2)
    # Thermal "sod" quantity
    T4=Tv1*(1.+rinf)
    T5=Tv2*(1.+rinf)
    T6=(rdo*J2ks+tdo*J1ks)*(1.+rinf)*rinf
    gammasod=(T4+T5-T6)/(1.-rinf2)
    #Treatment of the hotspot-effect
    alf=1e36
    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0. : alf=(dso/hotspot)*2./(ks+ko)
    if alf == 0. : 
        # The pure hotspot
        tsstoo=tss
        sumint=(1.-tss)/(ks*lai)
    else :
        # Outside the hotspot
        fhot=lai*sqrt(ko*ks)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1=0.
        y1=0.
        f1=1.
        fint=(1.-exp(-alf))*.05
        sumint=0.
        for istep in range(1,21):
            if istep < 20 :
                x2=-log(1.-istep*fint)/alf
            else :
                x2=1.
            y2=-(ko+ks)*lai*x2+fhot*(1.-exp(-alf*x2))/alf
            f2=exp(y2)
            sumint=sumint+(f2-f1)*(x2-x1)/(y2-y1)
            x1=x2
            y1=y2
            f1=f2
        tsstoo=f1
    if isnan(sumint) : sumint=0.
    # Bidirectional reflectance
    # Single scattering contribution
    rsos=w*lai*sumint
    gammasos=ko*lai*sumint
    # Total canopy contribution
    rso=rsos+rsod
    gammaso=gammasos+gammasod
    #Interaction with the soil
    dn=1.-rsoil*rdd
    if dn == 0.0 : dn=1e-36
    rddt=rdd+tdd*rsoil*tdd/dn
    rsdt=rsd+(tsd+tss)*rsoil*tdd/dn
    rdot=rdo+tdd*rsoil*(tdo+too)/dn
    rsodt=((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*rsoil/dn
    rsost=rso+tsstoo*rsoil
    rsot=rsost+rsodt

    return [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,
            rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]

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

    from math import sin, cos, acos, radians, pi 
    cts=cos(radians(tts))
    cto=cos(radians(tto))
    sts=sin(radians(tts))
    sto=sin(radians(tto))
    cospsi=cos(radians(psi))
    psir=radians(psi)
    cttl=cos(radians(ttl))
    sttl=sin(radians(ttl))
    cs=cttl*cts
    co=cttl*cto
    ss=sttl*sts
    so=sttl*sto  
    cosbts=5.
    if abs(ss) > 1e-6 : cosbts=-cs/ss
    cosbto=5.
    if abs(so) > 1e-6 : cosbto=-co/so
    if abs(cosbts) < 1.0:
        bts=acos(cosbts)
        ds=ss
    else:
        bts=pi
        ds=cs
    chi_s=2./pi*((bts-pi*0.5)*cs+sin(bts)*ss)
    if abs(cosbto) < 1.0:
        bto=acos(cosbto)
        do_=so
    else:
        if tto < 90.:
            bto=pi
            do_=co
        else:
            bto=0.0
            do_=-co
    chi_o=2.0/pi*((bto-pi*0.5)*co+sin(bto)*so)
    btran1=abs(bts-bto)
    btran2=pi-abs(bts+bto-pi)
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
    if bt2 > 0.: t2=sin(bt2)*(2.*ds*do_+ss*so*cos(bt1)*cos(bt3))
    denom=2.*pi**2
    frho=((pi-bt2)*t1+t2)/denom
    ftau=(-bt2*t1+t2)/denom
    if frho < 0. : frho=0.
    if ftau < 0. : ftau=0.
   
    return [chi_s,chi_o,frho,ftau]    

def Jfunc1(k,l,t) :
    ''' J1 function with avoidance of singularity problem.'''
    from numpy import exp,zeros,size
    nb=size(l)
    del_=(k-l)*t
    if nb > 1:
        result=zeros(nb)
        result[abs(del_) > 1e-3]=(exp(-l[abs(del_)> 1e-3]*t)-exp(-k*t))/(k-l[abs(del_)> 1e-3])
        result[abs(del_)<= 1e-3]=0.5*t*(exp(-k*t)+exp(-l[abs(del_)<= 1e-3]*t))*(1.-(del_[abs(del_)<= 1e-3]**2.)/12.)
    else:
        if abs(del_) > 1e-3 :
            result=(exp(-l*t)-exp(-k*t))/(k-l)
        else:
            result=0.5*t*(exp(-k*t)+exp(-l*t))*(1.-(del_**2.)/12.)
    return result

def Jfunc2(k,l,t) :
    '''J2 function.'''
    from numpy import exp
    return (1.-exp(-(k+l)*t))/(k+l)

def Jfunc1_wl(k,l,t) :
    '''J1 function with avoidance of singularity problem.'''
    from math import exp
    del_=(k-l)*t
    if abs(del_) > 1e-3 :
      result=(exp(-l*t)-exp(-k*t))/(k-l)
    else:
      result=0.5*t*(exp(-k*t)+exp(-l*t))*(1.-(del_**2.)/12.)
    return result

def Jfunc2_wl(k,l,t) :
    '''J2 function.'''
    from math import exp
    return (1.-exp(-(k+l)*t))/(k+l)

def Get_SunAngles(lat,lon,doy,hour,stdlon):
    '''Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).
    
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
     
    '''
    from math import pi, sin, cos, asin, acos, radians, degrees
    # Calculate declination
    declination=0.409*sin((2.0*pi*doy/365.0)-1.39)
    EOT=0.258*cos(declination)-7.416*sin(declination)-3.648*cos(2.0*declination)-9.228*sin(2.0*declination)
    LC=(stdlon-lon)/15.
    time_corr=(-EOT/60.)+LC
    solar_time=hour-time_corr
    # Get the hour angle
    w=(solar_time-12.0)*15.
    # Get solar elevation angle
    sin_thetha=cos(radians(w))*cos(declination)*cos(radians(lat))+sin(declination)*sin(radians(lat))
    sun_elev=asin(sin_thetha)
    # Get solar zenith angle
    sza=pi/2.0-sun_elev
    sza=degrees(sza)
    # Get solar azimuth angle
    cos_phi=(sin(declination)*cos(radians(lat))-cos(radians(w))*cos(declination)*sin(radians(lat)))/cos(sun_elev)
    if w <= 0.0:
        saa=degrees(acos(cos_phi))
    else:
        saa=360.-degrees(acos(cos_phi))

    return sza,saa
