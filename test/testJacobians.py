# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:01:39 2016

@author: hector
"""
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from collections import OrderedDict
import Prospect5
import FourSAIL
import FourSAILJacobian
import CostFunctionsPROSPECT4SAIL as FCost

Params=['N_leaf','Cab','Car','Cbrown','Cw','Cm','LAI','hotspot','leaf_angle']
ObjParam=['N_leaf','Cab','Car','Cbrown','Cw','Cm','LAI','hotspot','leaf_angle']
PotObjParams=FourSAILJacobian.paramsPro4SAIL
ObjBounds=OrderedDict([('N_leaf',[1.,4.]),('Cab',[0.0,100.0]),('Car',[0.0,40.0]),('Cbrown',[0.0,1.0]),
            ('Cw',[0.001,0.05]),('Cm',[0.001,0.02]),('LAI',[0.0,6.0]),
            ('hotspot',[0.01,0.1]),('leaf_angle',[50.0,70.0])])

bounds=dict([('N_leaf',[1.,4.]),('Cab',[0.0,100.0]),('Car',[0.0,40.0]),('Cbrown',[0.0,1.0]),
            ('Cw',[0.001,0.05]),('Cm',[0.001,0.02]),('LAI',[0.0,6.0]),
            ('hotspot',[0.01,0.1]),('leaf_angle',[50.0,70.0]),
            ('vza',[0.0,90.0]),('sza',[0.0,90.0]),('psi',[0.0,90.0]),('skyl',[0.0,1.0])])


DefaultParams={'N_leaf':1.5,'Cab':40,'Car':15,'Cbrown':0,
            'Cw':0.01,'Cm':0.005,'LAI':0.5,'leaf_angle':57.3,
            'hotspot':0.01}

scalerOutput=[]
FixedValues=[]
guess=[]
step=dict()
eps=1e-6

for param in PotObjParams:
    step[param]=eps*(ObjBounds[param][1]-ObjBounds[param][0])
    if param in ObjParam:
        scalerOutput.append((ObjBounds[param][0],ObjBounds[param][1]-ObjBounds[param][0]))
        guess.append(DefaultParams[param])
    else:
        FixedValues.append(DefaultParams[param])
    
scalerOutput=np.asarray(scalerOutput)
            
soil_file='/home/hector/Data/Scripts/Python/PROSPECT4SAIL/SoilSpectralLibrary/jhu.becknic.soil.inceptisol.xerumbrept.coarse.87P325.spectrum.txt'
rsoil=np.genfromtxt(soil_file)
wl_soil=rsoil[:,0]
rsoil=np.array(rsoil[:,1])
wl=2500
n_obs=1
wls=np.arange(400,2501)
# Generate the simulated observed reflectance
N_leaf=rnd.uniform(bounds['N_leaf'][0],bounds['N_leaf'][1])
Cab=rnd.uniform(bounds['Cab'][0],bounds['Cab'][1])
Car=rnd.uniform(bounds['Car'][0],bounds['Car'][1])
Cbrown=rnd.uniform(bounds['Cbrown'][0],bounds['Cbrown'][1])
Cw=rnd.uniform(bounds['Cw'][0],bounds['Cw'][1])
Cm=rnd.uniform(bounds['Cm'][0],bounds['Cm'][1])
LAI=rnd.uniform(bounds['LAI'][0],bounds['LAI'][1])
leaf_angle=rnd.uniform(bounds['leaf_angle'][0],bounds['leaf_angle'][1])
hotspot=rnd.uniform(bounds['hotspot'][0],bounds['hotspot'][1])
vza=rnd.uniform(bounds['vza'][0],bounds['vza'][1])
sza=rnd.uniform(bounds['sza'][0],bounds['sza'][1])
psi=rnd.uniform(bounds['psi'][0],bounds['psi'][1])
skyl=np.ones(2101)*rnd.uniform(bounds['skyl'][0],bounds['skyl'][1])

l,rho_leaf,tau_leaf=Prospect5.Prospect5(N_leaf,Cab,Car,Cbrown,Cw,Cm)
lidf=FourSAIL.CalcLIDF_Campbell(leaf_angle,n_elements=18)
[tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,
     rsot,gammasdf,gammasdb,gammaso]=FourSAIL.FourSAIL(LAI,hotspot,lidf,sza,vza,psi,rho_leaf,tau_leaf,rsoil)
rho_canopy_obs=[skyl*rdot+(1-skyl)*rsot]

   
# Generate the simulated reflectance
N_leaf=rnd.uniform(bounds['N_leaf'][0],bounds['N_leaf'][1])
Cab=rnd.uniform(bounds['Cab'][0],bounds['Cab'][1])
Car=rnd.uniform(bounds['Car'][0],bounds['Car'][1])
Cbrown=rnd.uniform(bounds['Cbrown'][0],bounds['Cbrown'][1])
Cw=rnd.uniform(bounds['Cw'][0],bounds['Cw'][1])
Cm=rnd.uniform(bounds['Cm'][0],bounds['Cm'][1])
LAI=rnd.uniform(bounds['LAI'][0],bounds['LAI'][1])
hotspot=rnd.uniform(bounds['hotspot'][0],bounds['hotspot'][1])
leaf_angle=rnd.uniform(bounds['leaf_angle'][0],bounds['leaf_angle'][1])

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw,Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up=np.zeros(len(ObjParam))
vza,sza,psi,skyl=[vza],[sza],[psi],[skyl]
mse=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)
mse_2,Jac_mse_analytical=FCost.FCostJac_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf+step['N_leaf'],Cab,Car,Cbrown,Cw,Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[0]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab+step['Cab'],Car,Cbrown,Cw,Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[1]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car+step['Car'],Cbrown,Cw,Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[2]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown+step['Cbrown'],Cw,Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[3]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw+step['Cw'],Cm,LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[4]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw,Cm+step['Cm'],LAI,hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[5]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw,Cm,LAI+step['LAI'],hotspot,leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[6]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw,Cm,LAI,hotspot+step['hotspot'],leaf_angle])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[7]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

x0=np.array([N_leaf,Cab,Car,Cbrown,Cw,Cm,LAI,hotspot,leaf_angle+step['leaf_angle']])
x0=(x0-scalerOutput[:,0])/scalerOutput[:,1]

mse_up[8]=FCost.FCost_ProSail(x0,ObjParam,FixedValues,n_obs,rho_canopy_obs,vza,sza,psi,skyl,rsoil,wls,scalerOutput)

Jac_mse_numerical=np.zeros(Jac_mse_analytical.shape)
for i,param in enumerate(PotObjParams):
    Jac_mse_numerical[i]=(np.asarray(mse_up[i])-np.asarray(mse))/step[param]

plt.close("all")
Jac_mse_numerical=np.asarray(Jac_mse_numerical)
Jac_mse_analytical=np.asarray(Jac_mse_analytical)
print('Relative differnece between numerical and analytical gradient: \n[%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s]\n'%PotObjParams, 
      abs((Jac_mse_analytical-Jac_mse_numerical)/Jac_mse_numerical))
#==============================================================================
# for i,param in enumerate(Params):
#     plt.figure()
#     plt.plot(Jac_mse_analytical[i],label='Jacobian')
#     plt.plot(Jac_mse_numerical[i],label='Numerical')
#     plt.legend()
#     plt.title(param)
#==============================================================================

