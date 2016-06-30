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
from FourSAIL import *
from Prospect5 import *

<<<<<<< HEAD

=======
>>>>>>> First commit for computing cost function jacobians
def FCostScaled_RMSE_ProSail_wl(x0,*args):
    ''' Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean
    Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
        
    Parameters
    ----------
    x0 : list
        a priori PROSAIL values to be retrieved during the inversion.
    args : list
        Additional arguments to be parsed in the inversion.
        
            * 'ObjParam': list of the PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
            * 'FixedValues' : dictionary with the values of the parameters that are fixedduring the inversion. The dictionary must complement the list from ObjParam.
            * 'N_obs' : integer with the total number of observations used for the inversion. N_Obs=1.
            * 'rho_canopy': list with the observed surface reflectances. The size of this list be wls*N_obs.
            * 'vza' : list with the View Zenith Angle for each one of the observations. The size must be equal to N_obs.
            * 'sza' : list with the Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
            * 'psi' : list with the Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
            * 'skyl' : list with the ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs.
            * 'rsoil' : list with the background (soil) reflectance. The size must be equal to the size of wls.
            * 'wls' : list with wavebands used in the inversion.
            * scale : minimum and scale tuple [min,scale] for each objective parameter.
    
    Returns
    -------
    mse : float
        Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
    #Extract all the values needed
    ObjParam=args[0]
    FixedValues=args[1]
    n_obs=args[2]
    rho_canopy=args[3]
    vza=args[4]
    sza=args[5]
    psi=args[6]
    skyl=args[7]
    rsoil=args[8]
    wls=args[9]
    scale=args[10]
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[1][i])+float(scale[0][i])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_obs*n_wl)
    #Calculate LIDF
    lidf=CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
    i=0
    for obs in range(0,n_obs):
        j=0
        for wl in wls:
            [l,r,t]=Prospect5_wl(wl,input_parameters['N_leaf'],
                input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
                input_parameters['Cw'],input_parameters['Cm'])
            [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
                 rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]=FourSAIL_wl(input_parameters['LAI'],
                 input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
                float(psi[obs]),r,t,float(rsoil[j]))
            r2=rdot*float(skyl[i])+rsot*(1-float(skyl[i]))
            error[i]=(rho_canopy[i]-r2)**2
            i+=1
            j+=1
    mse=0.5*(np.mean(error,dtype=np.float64))
    return mse

#==============================================================================
# def JacobianFCostScaled_RMSE_ProSail_wl(x0,*args):
#     ''' Jacobian of the Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean
#     Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
#         
#     Parameters
#     ----------
#     x0 : list
#         a priori PROSAIL values to be retrieved during the inversion.
#     args : list
#         Additional arguments to be parsed in the inversion.
#         
#             * 'ObjParam': list of the PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
#             * 'FixedValues' : dictionary with the values of the parameters that are fixedduring the inversion. The dictionary must complement the list from ObjParam.
#             * 'N_obs' : integer with the total number of observations used for the inversion. N_Obs=1.
#             * 'rho_canopy': list with the observed surface reflectances. The size of this list be wls*N_obs.
#             * 'vza' : list with the View Zenith Angle for each one of the observations. The size must be equal to N_obs.
#             * 'sza' : list with the Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
#             * 'psi' : list with the Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
#             * 'skyl' : list with the ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs.
#             * 'rsoil' : list with the background (soil) reflectance. The size must be equal to the size of wls.
#             * 'wls' : list with wavebands used in the inversion.
#             * Bounds : minimum and maximum tuple [min,max] for each objective parameter.
#     
#     Returns
#     -------
#     rmse : float
#         Root Mean Square Error of observed vs. modelled surface reflectance
#         This is the function to be minimized.'''
#     
#     import numpy as np
#     param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
#     #Extract all the values needed
#     ObjParam=args[0]
#     FixedValues=args[1]
#     n_obs=args[2]
#     rho_canopy=args[3]
#     vza=args[4]
#     sza=args[5]
#     psi=args[6]
#     skyl=args[7]
#     rsoil=args[8]
#     wls=args[9]
#     scaler=args[10]
#     # Get the a priori parameters and fixed parameters for the inversion
#     input_parameters=dict()
#     i=0
#     j=0
#     for param in param_list:
#         if param in ObjParam:
#             #Transform the random variables (0-1) into biophysical variables 
#             input_parameters[param]=x0[i]*float((bounds[i][1]-bounds[i][0]))+float(bounds[i][0])
#             i=i+1
#         else:
#             input_parameters[param]=FixedValues[j]
#             j=j+1
#     # Start processing    
#     n_wl=len(wls)
#     error= np.zeros(n_obs*n_wl)
#     #Calculate LIDF
#     lidf=CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
#     i=0
#     for obs in range(n_obs):
#         j=0
#         for wl in wls:
#             [l,r,t]=Prospect5_wl(wl,input_parameters['N_leaf'],
#                 input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
#                 input_parameters['Cw'],input_parameters['Cm'])
#             [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
#                  rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]=FourSAIL_wl(input_parameters['LAI'],
#                  input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
#                 float(psi[obs]),r,t,float(rsoil[j]))
#             r2=rdot*skyl[obs]+rsot*(1-skyl[obs])
#             error[i]=(rho_canopy[i]-r2)
#             i+=1
#             j+=1
#     Jac=np.zeros(len(ObjParam))
#     for i in range(len(ObjParam)):
#         Jac[i]=np.mean(error*x0[i],dtype=np.float64)
#     return Jac
#==============================================================================

def FCostScaled_RMSE_ProSail(x0,*args):
    ''' Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean
    Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
        
    Parameters
    ----------
    x0 : list
        a priori PROSAIL values to be retrieved during the inversion.
    args : list
        Additional arguments to be parsed in the inversion.
        
            * 'ObjParam': list of the PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
            * 'FixedValues' : dictionary with the values of the parameters that are fixedduring the inversion. The dictionary must complement the list from ObjParam.
            * 'N_obs' : integer with the total number of observations used for the inversion. N_Obs=1.
            * 'rho_canopy': list with the observed surface reflectances. The size of this list be wls*N_obs.
            * 'vza' : list with the View Zenith Angle for each one of the observations. The size must be equal to N_obs.
            * 'sza' : list with the Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
            * 'psi' : list with the Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
            * 'skyl' : list with the ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs.
            * 'rsoil' : list with the background (soil) reflectance. The size must be equal to the size of wls.
            * 'wls' : list with wavebands used in the inversion.
            * scale : minimum and scale tuple [min,scale] for each objective parameter.
    
    Returns
    -------
    rmse : float
        Root Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
    #Extract all the values needed
    ObjParam=args[0]
    FixedValues=args[1]
    n_obs=args[2]
    rho_canopy=args[3]
    vza=args[4]
    sza=args[5]
    psi=args[6]
    skyl=np.asarray(args[7])
    rsoil=np.asarray(args[8])
    wls=np.asarray(args[9])
    scale=args[10]
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[1][i])+float(scale[0][i])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_obs*n_wl)
    #Calculate LIDF
    lidf=CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
    for obs in range(n_obs):
        [l,r,t]=Prospect5(input_parameters['N_leaf'],
                input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
                input_parameters['Cw'],input_parameters['Cm'])
        k=[k for k,wl in enumerate(l) if float(wl) in wls]
        r=r[k]
        t=t[k]
        [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
             rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]=FourSAIL(input_parameters['LAI'],
             input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
            float(psi[obs]),r,t,rsoil)
        r2=rdot*skyl[obs]+rsot*(1.-skyl[obs])
        error[obs*n_wl:(obs+1)*n_wl]=(rho_canopy[obs*n_wl:(obs+1)*n_wl]-r2)**2
    mse=0.5*(np.mean(error,dtype=np.float64))

    return mse
#==============================================================================
# def JacobianFCostScaled_RMSE_ProSail(x0,*args):
#     ''' Cost Function for inverting PROSPEC5 + 4SAIL based on the Root Mean
#     Square Error of observed vs. modeled reflectances and scaled [0,1] parameters
#         
#     Parameters
#     ----------
#     x0 : list
#         a priori PROSAIL values to be retrieved during the inversion.
#     args : list
#         Additional arguments to be parsed in the inversion.
#         
#             * 'ObjParam': list of the PROSAIL parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot'].
#             * 'FixedValues' : dictionary with the values of the parameters that are fixedduring the inversion. The dictionary must complement the list from ObjParam.
#             * 'N_obs' : integer with the total number of observations used for the inversion. N_Obs=1.
#             * 'rho_canopy': list with the observed surface reflectances. The size of this list be wls*N_obs.
#             * 'vza' : list with the View Zenith Angle for each one of the observations. The size must be equal to N_obs.
#             * 'sza' : list with the Sun Zenith Angle for each one of the observations. The size must be equal to N_obs.
#             * 'psi' : list with the Relative View-Sun Angle for each one of the observations. The size must be equal to N_obs.
#             * 'skyl' : list with the ratio of diffuse radiation for each one of the observations. The size must be equal to N_obs.
#             * 'rsoil' : list with the background (soil) reflectance. The size must be equal to the size of wls.
#             * 'wls' : list with wavebands used in the inversion.
#             * Bounds : minimum and maximum tuple [min,max] for each objective parameter.
#     
#     Returns
#     -------
#     rmse : float
#         Root Mean Square Error of observed vs. modelled surface reflectance
#         This is the function to be minimized.'''
#     
#     import numpy as np
#     param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm', 'LAI', 'leaf_angle','hotspot']
#     #Extract all the values needed
#     ObjParam=args[0]
#     FixedValues=args[1]
#     n_obs=args[2]
#     rho_canopy=args[3]
#     vza=args[4]
#     sza=args[5]
#     psi=args[6]
#     skyl=args[7]
#     rsoil=np.asarray(args[8])
#     wls=np.asarray(args[9])
#     bounds=args[10]
#     # Get the a priori parameters and fixed parameters for the inversion
#     input_parameters=dict()
#     i=0
#     j=0
#     for param in param_list:
#         if param in ObjParam:
#             #Transform the random variables (0-1) into biophysical variables 
#             input_parameters[param]=x0[i]*float((bounds[i][1]-bounds[i][0]))+float(bounds[i][0])
#             i=i+1
#         else:
#             input_parameters[param]=FixedValues[j]
#             j=j+1
#     # Start processing    
#     n_wl=len(wls)
#     error= np.zeros(n_obs*n_wl)
#     #Calculate LIDF
#     lidf=CalcLIDF_Campbell(float(input_parameters['leaf_angle']))
#     for obs in range(n_obs):
#         [l,r,t]=Prospect5(input_parameters['N_leaf'],
#                 input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
#                 input_parameters['Cw'],input_parameters['Cm'])
#         k=[k for k,wl in enumerate(l) if float(wl) in wls]
#         r=r[k]
#         t=t[k]
#         [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
#              rsodt,rsost,rsot,gammasdf,gammasdb,gammaso]=FourSAIL(input_parameters['LAI'],
#              input_parameters['hotspot'],lidf,float(sza[obs]),float(vza[obs]),
#             float(psi[obs]),r,t,rsoil)
#         r2=rdot*skyl[obs]+rsot*(1.-skyl[obs])
#         error[obs*n_wl:(obs+1)*n_wl]=(rho_canopy[obs*n_wl:(obs+1)*n_wl]-r2)
# 
#     Jac=np.zeros(len(ObjParam))
#     for i in range(len(ObjParam)):
#         Jac[i]=np.mean(error*x0[i],dtype=np.float64)
#     return Jac
# 
#==============================================================================

def FCostScaled_RMSE_PROSPECT5_wl(x0,*args):
    ''' Cost Function for inverting PROSPECT5  the Root MeanSquare Error of 
    observed vs. modeled reflectances and scaled [0,1] parameters.
        
    Parameters
    ----------
    x0 : list
        a priori PROSAIL values to be retrieved during the inversion.
    args : list 
        additional arguments to be parsed in the inversion.
        
            * ObjParam': list of the PROSPECT5 parameters to be retrieved during the inversion, sorted in the same order as in the param list. ObjParam'=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm'].
            * FixedValues : dictionary with the values of the parameters that are fixed during the inversion. The dictionary must complement the list from ObjParam.
            * n_obs : integer with the total number of observations used for the inversion. N_Obs=1.
            * rho_leaf: list with the observed surface reflectances. The size of this list be wls*N_obs.
            * wls : list with wavebands used in the inversion.
            * scale : minimum and scale tuple [min,scale] for each objective parameter, used to unscale the values.
            
    Returns
    -------
    rmse : float
        Root Mean Square Error of observed vs. modelled surface reflectance
        This is the function to be minimized.'''
    
    import numpy as np
    param_list=['N_leaf','Cab','Car','Cbrown', 'Cw','Cm']
    #Extract all the values needed
    ObjParam=args[0]
    FixedValues=args[1]
    rho_leaf=args[2]
    wls=args[3]
    scale=args[4]
    # Get the a priori parameters and fixed parameters for the inversion
    input_parameters=dict()
    i=0
    j=0
    for param in param_list:
        if param in ObjParam:
            #Transform the random variables (0-1) into biophysical variables 
            input_parameters[param]=x0[i]*float(scale[1][i])+float(scale[0][i])
            i=i+1
        else:
            input_parameters[param]=FixedValues[j]
            j=j+1
    # Start processing    
    n_wl=len(wls)
    error= np.zeros(n_wl)
    for i,wl in enumerate(wls):
        [l,r,t]=Prospect5_wl(wl,input_parameters['N_leaf'],
            input_parameters['Cab'],input_parameters['Car'],input_parameters['Cbrown'], 
            input_parameters['Cw'],input_parameters['Cm'])
        error[i]=(rho_leaf[i]-r)**2
    mse=0.5*(np.mean(error,dtype=np.float64))

    return mse
