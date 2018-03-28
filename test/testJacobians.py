# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:01:39 2016

@author: hector
"""
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from pyPro4Sail.ProspectD import ProspectD
from pyPro4Sail.ProspectDJacobian import JacProspectD

import pyPro4Sail.FourSAIL as FourSAIL
import pyPro4Sail.FourSAILJacobian as FourSAILJacobian
#from pyPro4Sail.cost_functions import FCostJac_ProSail

plt.close('all')
Params=('N_leaf','Cab','Car','Cbrown','Cw','Cm','Ant','LAI','hotspot','leaf_angle')
colors=('black', 'green', 'orange', 'brown', 'blue', 'yellow', 'red', 'purple', 'cyan', 'grey')

leaf_params = ('N_leaf','Cab','Car','Cbrown','Cw','Cm','Ant')
canopy_params = ('LAI','hotspot','leaf_angle')

ObjBounds=dict([('N_leaf',[1.,4.]),('Cab',[0.0,100.0]),('Car',[0.0,40.0]),('Cbrown',[0.0,1.0]),
             ('Cw',[0.001,0.05]),('Cm',[0.001,0.02]),('Ant',[0.0,40.]),('LAI',[0.0,6.0]),
            ('hotspot',[0.01,0.1]),('leaf_angle',[50.0,70.0])])

bounds=dict([('vza',[0.0,90.0]),('sza',[0.0,90.0]),('psi',[0.0,90.0]),('skyl',[0.0,1.0])])


step=dict()
eps=1e-6

for param in Params:
    step[param]=eps*(ObjBounds[param][1]-ObjBounds[param][0])
            
soil_file='/home/hector/Data/Scripts/Python/PROSPECT4SAIL/SoilSpectralLibrary/jhu.becknic.soil.inceptisol.xerumbrept.coarse.87P325.spectrum.txt'
rsoil=np.genfromtxt(soil_file)
wl_soil=rsoil[:,0]
rsoil=np.array(rsoil[:,1])
repeats=100

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()
fig9, ax9 = plt.subplots()

for k in range(repeats):
    print(k)
    # Generate the simulated observed reflectance
    N_leaf=rnd.uniform(ObjBounds['N_leaf'][0],ObjBounds['N_leaf'][1])
    Cab=rnd.uniform(ObjBounds['Cab'][0],ObjBounds['Cab'][1])
    Car=rnd.uniform(ObjBounds['Car'][0],ObjBounds['Car'][1])
    Cbrown=rnd.uniform(ObjBounds['Cbrown'][0],ObjBounds['Cbrown'][1])
    Cw=rnd.uniform(ObjBounds['Cw'][0],ObjBounds['Cw'][1])
    Cm=rnd.uniform(ObjBounds['Cm'][0],ObjBounds['Cm'][1])
    Ant=rnd.uniform(ObjBounds['Ant'][0],ObjBounds['Ant'][1])
    LAI=rnd.uniform(ObjBounds['LAI'][0],ObjBounds['LAI'][1])
    leaf_angle=rnd.uniform(ObjBounds['leaf_angle'][0],ObjBounds['leaf_angle'][1])
    hotspot=rnd.uniform(ObjBounds['hotspot'][0],ObjBounds['hotspot'][1])
    vza=rnd.uniform(bounds['vza'][0],bounds['vza'][1])
    sza=rnd.uniform(bounds['sza'][0],bounds['sza'][1])
    psi=rnd.uniform(bounds['psi'][0],bounds['psi'][1])
    skyl=np.ones(2101)*rnd.uniform(bounds['skyl'][0],bounds['skyl'][1])
    
    
    [l, rho_leaf, tau_leaf, Jac_rho_leaf, Jac_tau_leaf] = JacProspectD(N_leaf,Cab,Car,Cbrown,Cw,Cm,Ant)
    
    Jac_rho_leaf_numerical = np.zeros((len(leaf_params),2101))
    Jac_tau_leaf_numerical = np.zeros((len(leaf_params),2101))
#==============================================================================
#     for i,param in enumerate(leaf_params):
#         leaf_args = [N_leaf,Cab,Car,Cbrown,Cw,Cm,Ant] 
#         leaf_args[i] = leaf_args[i] + step[param]  
#         l, rho_leaf_up, tau_leaf_up, *_ = JacProspectD(*leaf_args)
#         Jac_rho_leaf_numerical[i] = (rho_leaf_up - rho_leaf) / step[param]
#         Jac_tau_leaf_numerical[i] = (tau_leaf_up - tau_leaf) / step[param]
#   
#     for i,param in enumerate(leaf_params):
#         ax1.plot(l, 
#                  np.abs((Jac_rho_leaf[i]-Jac_rho_leaf_numerical[i])/Jac_rho_leaf_numerical[i]), 
#                          color=colors[i], label = param)
#         ax2.plot(l,Jac_rho_leaf_numerical[i], 
#                          color=colors[i], label = param)
#         ax3.plot(l,Jac_rho_leaf[i], 
#                          color=colors[i], label = param)
# 
#==============================================================================
    lidf=FourSAIL.CalcLIDF_Campbell(leaf_angle,n_elements=18)
    [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,
         rsot,gammasdf,gammasdb,gammaso]=FourSAIL.FourSAIL(LAI,hotspot,lidf,sza,vza,psi,rho_leaf,tau_leaf,rsoil)
    rho_canopy=skyl*rdot+(1-skyl)*rsot
    
    Jac_rho_canopy_numerical = np.zeros((len(Params),2101))
    
    for j,param in enumerate(Params):
        canopy_args = [N_leaf, Cab, Car, Cbrown, Cw, Cm, Ant, LAI, hotspot, leaf_angle] 
        canopy_args[j] = canopy_args[j] + step[param]  
        l,rho_leaf_up, tau_leaf_up = ProspectD(*canopy_args[0:7])
        lidf_up = FourSAIL.CalcLIDF_Campbell(canopy_args[9],n_elements=18)
        [_,_,_,_,_,_,_,_,_,_,_,_,_,_,
                 rdot,_,_,rsot,_,_,_]=FourSAIL.FourSAIL(canopy_args[7],
                                                        canopy_args[8],
                                                        lidf_up,
                                                        sza,
                                                        vza,
                                                        psi,
                                                        rho_leaf_up,
                                                        tau_leaf_up,
                                                        rsoil)
        rho_canopy_up=skyl*rdot+(1-skyl)*rsot  
    
        Jac_rho_canopy_numerical[j]=(np.asarray(rho_canopy_up)-np.asarray(rho_canopy))/step[param]
    
   
    lidf, Delta_lidf = FourSAILJacobian.JacCalcLIDF_Campbell(leaf_angle,n_elements=18)
    [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,
     rsost,rsot,gammasdf,gammasdb,gammaso,
         Delta_tss,Delta_too,Delta_tsstoo,Delta_rdd,
         Delta_tdd,Delta_rsd,Delta_tsd,Delta_rdo,Delta_tdo,
         Delta_rso,Delta_rsos,Delta_rsod,Delta_rddt,Delta_rsdt,Delta_rdot,
         Delta_rsodt,Delta_rsosJac_rho_leaf_numericalt,Delta_rsot,Delta_gammasdf,Delta_gammasdb,
         Delta_gammaso] = FourSAILJacobian.JacFourSAIL(LAI,
                                                hotspot,
                                                [lidf, Delta_lidf],
                                                sza,
                                                vza,
                                                psi,
                                                rho_leaf,
                                                tau_leaf,
                                                rsoil,
                                                Delta_rho=Jac_rho_leaf,
                                                Delta_tau=Jac_tau_leaf)
    
    rho_canopy_obs_2=skyl*rdot+(1-skyl)*rsot
    Jac_rho_canopy_analytical = skyl*Delta_rdot+(1-skyl)*Delta_rsot
    
    for i,param in enumerate(leaf_params):
        ax1.plot(l, 
                 np.abs((Jac_rho_canopy_analytical[i]-Jac_rho_canopy_numerical[i])/Jac_rho_canopy_numerical[i]), 
                         color=colors[i], label = param)
        
        ax2.plot(l,Jac_rho_canopy_numerical[i], 
                         color=colors[i], label = param)
        
        ax3.plot(l,Jac_rho_canopy_analytical[i], 
                     color=colors[i], label = param)

    for j,param in enumerate(canopy_params):
        i = j+len(leaf_params)
        ax4.plot(l, 
                 np.abs((Jac_rho_canopy_analytical[i]-Jac_rho_canopy_numerical[i])/Jac_rho_canopy_numerical[i]), 
                         color=colors[i], label = param)
        
        ax5.plot(l,Jac_rho_canopy_numerical[i], 
                         color=colors[i], label = param)
        
        ax6.plot(l,Jac_rho_canopy_analytical[i], 
                     color=colors[i], label = param)



    #==============================================================================
    # # Cost Function Jacobian
    #==============================================================================
    DefaultParams={'N_leaf':1.5,'Cab':40,'Car':15,'Cbrown':0,
                'Cw':0.01,'Cm':0.005,'Ant':0.,'LAI':0.5,'leaf_angle':57.3,
                'hotspot':0.01}
    
    wls=np.arange(400,2501)
    n_obs=1
    
    scalerOutput=[]
    FixedValues=[]
    guess = []

    for param in Params:
        scalerOutput.append((ObjBounds[param][0],ObjBounds[param][1]-ObjBounds[param][0]))
        guess.append(rnd.uniform(ObjBounds[param][0],ObjBounds[param][1]))
    
    scalerOutput=np.asarray(scalerOutput)
    x0 = (np.array(guess)-scalerOutput[:,0])/scalerOutput[:,1]
    
    mse, Jac_mse = FCostJac_ProSail(x0, 
                                    Params, 
                                    FixedValues, 
                                    n_obs, 
                                    [rho_canopy], 
                                    [vza], 
                                    [sza],
                                    [psi],
                                    [skyl],
                                    rsoil,
                                    wls,
                                    scalerOutput)
    
    Jac_mse_numerical = np.zeros(len(Params))                             
    for j,param in enumerate(Params):
        canopy_args = guess[:] 
        canopy_args[j] = canopy_args[j] + step[param]  

        x0_up = (np.array(canopy_args)-scalerOutput[:,0])/scalerOutput[:,1]
        
        mse_up, _ = FCostJac_ProSail(x0_up, 
                                    Params, 
                                    FixedValues, 
                                    n_obs, 
                                    [rho_canopy], 
                                    [vza], 
                                    [sza],
                                    [psi],
                                    [skyl],
                                    rsoil,
                                    wls,
                                    scalerOutput)
        
        Jac_mse_numerical[j]=(np.asarray(mse_up)-np.asarray(mse))/step[param]
    
 
    ax7.plot(range(len(Params)), 
             np.abs((Jac_mse-Jac_mse_numerical)/Jac_mse_numerical), 'b.')
  
    ax8.plot(range(len(Params)),Jac_mse_numerical, 'b.')
    
    ax9.plot(range(len(Params)),Jac_mse, 'b.')

