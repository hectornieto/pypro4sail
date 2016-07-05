# -*- coding: utf-8 -*-
'''
Created on Tue Apr 14 13:36:24 2015

@author: ector Nieto (hnieto@ias.csic.es)

Modified on Apr 14 2016
@author: Hector Nieto (hnieto@ias.csic.es)

DESCRIPTION
===========
This package contains several cost/merit functions for inverting :doc:`Prospect5`
and :doc:`FourSAIL`. It requires the import of the following modules.

* :doc:`FourSAIL` for simulating the canopy reflectance and transmittance factors.
* :doc:`Prospect5` for simulating the lambertian reflectance and transmittance of a leaf.

PACKAGE CONTENTS
================

* :func:`FCost_RMSE_ProSail_wl` Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean Square Error of observed vs. modeled reflectances.
* :func:`FCostScaled_RMSE_ProSail_wl` Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean Square Error of observed vs. modeled reflectances and scaled [0,1] parameters (recommended over :func:`FCost_RMSE_ProSail_wl`).
* :func:`FCostScaled_RMSE_PROSPECT5_wl` Cost Function for inverting PROSPEC5 based on the Relative Root MeanSquare Error of observed vs. modeled reflectances and scaled [0,1] parameters.
* :func:`FCostScaled_RRMSE_PROSPECT5_wl` Cost Function for inverting PROSPEC5 based on the Relative Root MeanSquare Error of observed vs. modeled reflectances and scaled [0,1] parameters.

'''   
import FourSAIL
import Prospect5
import FourSAILJacobian
import Prospect5Jacobian

def FCost_ProSail_wl(x0,ObjParam,FixedValues,n_obs,rho_canopy,vza,sza,psi,skyl,rsoil,wls,scale):
    ''' Cost Function for inverting PROSPEC5 + 4SAIL based on the Mean
    Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSAIL values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    N_obs : int
        the total number of observations used for the inversion. N_Obs=1.
    rho_canopy : 2D-array
        observed surface reflectances. The size of this list be N_obs x n_wls.
    vza : list 
        View Zenith Angle for each one of the observations. The size must be equal to N_obs.
    sza : list 
        Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
    psi : list
        Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
    skyl : 2D-array
        ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs x wls.
    rsoil : 1D-array
        background (soil) reflectance. The size must be equal to n_wls.
    wls : list 
        wavebands used in the inversion. The size must be equal to n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_obs*n_wl)
    #Calculate LIDF
    lidf=FourSAIL.CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
    i=0
    for obs in range(n_obs):
        j=0
        for wl in wls:
            [l,r,t]=Prospect5.Prospect5_wl(wl,input_parameters['N_leaf'],
                input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
                input_parameters['Cw'],input_parameters['Cm'])
            [_,_,_,_,_,_,_,_,_,_,_,_,_,_,rdot,
                 _,_,rsot,_,_,_]=FourSAIL.FourSAIL_wl(input_parameters['LAI'],
                 input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
                float(psi[obs]),r,t,float(rsoil[j]))
            r2=rdot*float(skyl[obs,j])+rsot*(1-float(skyl[obs,j]))
            error[i]=(r2-rho_canopy[obs,j])**2
            i+=1
            j+=1
    mse=0.5*np.mean(error)
    return mse

def FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy,vza,sza,psi,skyl,rsoil,wls,scale):
    ''' Cost Function and for inverting PROSPEC5 + 4SAIL based on the Mean
    Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSAIL values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    N_obs : int
        the total number of observations used for the inversion. N_Obs=1.
    rho_canopy : 2D-array
        observed surface reflectances. The size of this list be N_obs x n_wls.
    vza : list 
        View Zenith Angle for each one of the observations. The size must be equal to N_obs.
    sza : list 
        Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
    psi : list
        Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
    skyl : 2D-array
        ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs x wls.
    rsoil : 1D-array
        background (soil) reflectance. The size must be equal to n_wls.
    wls : list 
        wavebands used in the inversion. The size must be equal to n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
        
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']

    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_obs*n_wl)
    #Calculate LIDF
    lidf=FourSAIL.CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
    for obs in range(n_obs):
        [l,r,t]=Prospect5.Prospect5(input_parameters['N_leaf'],
                input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
                input_parameters['Cw'],input_parameters['Cm'])
        k=[k for k,wl in enumerate(l) if float(wl) in wls]
        r=r[k]
        t=t[k]
        [_,_,_,_,_,_,_,_,_,_,_,_,_,_,rdot,
             _,_,rsot,_,_,_]=FourSAIL.FourSAIL(input_parameters['LAI'],
             input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
            float(psi[obs]),r,t,rsoil)
        r2=rdot*np.asarray(skyl[obs])+rsot*(1.-np.asarray(skyl[obs]))
        error[obs*n_wl:(obs+1)*n_wl]=(r2-rho_canopy[obs])**2
    mse=0.5*np.mean(error)

    return mse

def FCostJac_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy,vza,sza,psi,skyl,rsoil,wls,scale):
    ''' Cost Function and its Jacobian for inverting PROSPEC5 + 4SAIL based on the Mean
    Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSAIL values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    N_obs : int
        the total number of observations used for the inversion. N_Obs=1.
    rho_canopy : 1D-array
        observed surface reflectances. The size of this list be wls*N_obs.
    vza : list 
        View Zenith Angle for each one of the observations. The size must be equal to N_obs.
    sza : list 
        Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
    psi : list
        Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
    skyl : 2D-array
        ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs x wls.
    rsoil : 1D-array
        background (soil) reflectance. The size must be equal to n_wls.
    wls : list 
        wavebands used in the inversion. The size must be equal to n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    param_index=[]
    for k,param in enumerate(param_list):
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            param_index.append(k)
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_obs*n_wl)
    #Calculate LIDF
    Jac_lidf,lidf=FourSAILJacobian.JacCalcLIDF_Campbell(float(input_parameters['leaf_angle']))
    for obs in range(n_obs):
        l,Jac_r,Jac_t,r,t=Prospect5Jacobian.Prospect5(input_parameters['N_leaf'],
                input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
                input_parameters['Cw'],input_parameters['Cm'])
        k=[k for k,wl in enumerate(l) if float(wl) in wls]
        r=r[k]
        t=t[k]
        [_,_,_,_,_,_,_,_,_,
            _,_,_,_,_,Delta_rdot,_,_,Delta_rsot,_,_,_,
            _,_,_,_,_,_,_,_,_,
            _,_,_,_,_,rdot,_,_,rsot,_,_,_]=FourSAILJacobian.JacFourSAIL(input_parameters['LAI'],
             input_parameters['hotspot'],(Jac_lidf,lidf),float(sza[obs]),float(vza[obs]),
            float(psi[obs]),r,t,rsoil,Jac_r,Jac_t)
        r2=rdot*np.array(skyl[obs])+rsot*(1.-np.array(skyl[obs]))
        Delta_r2=Delta_rdot[param_index]*np.array(skyl[obs])+Delta_rsot[param_index]*(1.-np.array(skyl[obs]))
        error[obs*n_wl:(obs+1)*n_wl]=(r2-rho_canopy[obs])**2
        Delta_error=2*(r2-rho_canopy[obs])*Delta_r2
    mse=0.5*np.mean(error)
    Jac_mse=0.5*np.mean(Delta_error,axis=1)
    return mse,Jac_mse


def FCost_PROSPECT5_wl(x0,ObjParam,FixedValues,rho_leaf,wls,scale):
    ''' Cost Function for inverting PROSPECT5  the Root MeanSquare Error of 
    observed vs. modeled reflectances and scaled [0,1] parameters.
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSPECT5 values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    rho_leaf : 1D-array
        observed leaf reflectance. The size of this list be n_wlss.
    wls : list 
        wavebands used in the inversion. The size is n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm']
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_wl)
    for i,wl in enumerate(wls):
        [l,r,t]=Prospect5.Prospect5_wl(wl,input_parameters['N_leaf'],
            input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
            input_parameters['Cw'],input_parameters['Cm'])
        error[i]=(r-rho_leaf[i])**2
    mse=0.5*np.mean(error)

    return mse

def FCost_PROSPECT5(x0,ObjParam,FixedValues,rho_leaf,wls,scale):
    ''' Cost Function for inverting PROSPECT5  the Root MeanSquare Error of 
    observed vs. modeled reflectances and scaled [0,1] parameters.
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSPECT5 values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    rho_leaf : 1D-array
        observed leaf reflectance. The size of this list be n_wls.
    wls : list 
        wavebands used in the inversion. The size is n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm']
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing   
    [l,r,t]=Prospect5.Prospect5(input_parameters['N_leaf'],
            input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
            input_parameters['Cw'],input_parameters['Cm'])
    k=[k for k,wl in enumerate(l) if float(wl) in wls]
    error=(r[k]-rho_leaf)**2
    mse=0.5*np.mean(error)
    return mse
    
def FCostJac_PROSPECT5(x0,ObjParam,FixedValues,rho_leaf,wls,scale):
    ''' Cost Function for inverting PROSPECT5  the Root MeanSquare Error of 
    observed vs. modeled reflectances and scaled [0,1] parameters.
        
    Parameters
    ----------
    x0 : list
        Scaled (0-1) a priori PROSPECT5 values to be retrieved during the inversion.
    ObjParam : list 
        PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
    FixedValues' : dict
        Values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
    rho_leaf : 1D-array
        observed leaf reflectance. The size of this list be n_wls.
    wls : list 
        wavebands used in the inversion. The size is n_wls.
    scale : list
        minimum and scale tuple (min,scale) for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm']
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    param_index=[]
    for k,param in enumerate(param_list):
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[i][1])+float(scale[i][0])
            param_index.append(k)
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    l,Delta_r,Delta_t,r,t=Prospect5Jacobian.JacProspect5(input_parameters['N_leaf'],
            input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
            input_parameters['Cw'],input_parameters['Cm'])
    k=[k for k,wl in enumerate(l) if float(wl) in wls]
    error=(r[k]-rho_leaf)**2
    Delta_r=Delta_r[param_index]
    Delta_error=2*(r[k]-rho_leaf)*Delta_r[:,k]
    mse=0.5*np.mean(error)
    Jac_mse=0.5*np.mean(Delta_error,axis=1)
    return mse,Jac_mse