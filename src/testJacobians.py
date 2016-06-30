# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 18:01:39 2016

@author: hector
"""
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
Params=['N_leaf','Cab','Car','Cbrown','Cw','Cm']
bounds={'N_leaf':[1,2.5],'Cab':[1e-3,80.0],'Car':[1e-3,40.0],'Cbrown':[0.0,1.0],
        'Cw':[1e-6,0.04],'Cm':[1e-6,0.05],'LAI':[0.0,8.0],'leaf_angle':[30.0,85.0],
        'hotspot':[0.01,0.50]}

bounds={'N_leaf':[1,2.5],'Cab':[1e-3,80.0],'Car':[1e-3,40.0],'Cbrown':[0.0,1.0],
        'Cw':[1e-6,0.04],'Cm':[1e-6,0.05]}


N_leaf=rnd.uniform(bounds['N_leaf'][0],bounds['N_leaf'][1])
Cab=rnd.uniform(bounds['Cab'][0],bounds['Cab'][1])
Car=rnd.uniform(bounds['Car'][0],bounds['Car'][1])
Cbrown=rnd.uniform(bounds['Cbrown'][0],bounds['Cbrown'][1])
Cw=rnd.uniform(bounds['Cw'][0],bounds['Cw'][1])
Cm=rnd.uniform(bounds['Cm'][0],bounds['Cm'][1])

eps=1e-6

step=dict()
for param,bound in bounds.items():
    step[param]=eps*(bound[1]-bound[0])
    
r90_analytical,t90_analytical=JacProspect5(N_leaf,Cab,Car,Cbrown,Cw,Cm)

t90_numerical=[]
r90_numerical= []
wls,r90,t90=Prospect5(N_leaf,Cab,Car,Cbrown,Cw,Cm)

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf+step['N_leaf'],Cab,Car,Cbrown,Cw,Cm)

r90_numerical.append((delta_r90_up-r90)/step['N_leaf'])
t90_numerical.append((delta_t90_up-t90)/step['N_leaf'])

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf,Cab+step['Cab'],Car,Cbrown,Cw,Cm)

r90_numerical.append((delta_r90_up-r90)/step['Cab'])
t90_numerical.append((delta_t90_up-t90)/step['Cab'])

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf,Cab,Car+step['Car'],Cbrown,Cw,Cm)

r90_numerical.append((delta_r90_up-r90)/step['Car'])
t90_numerical.append((delta_t90_up-t90)/step['Car'])

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf,Cab,Car,Cbrown+step['Cbrown'],Cw,Cm)

r90_numerical.append((delta_r90_up-r90)/step['Cbrown'])
t90_numerical.append((delta_t90_up-t90)/step['Cbrown'])

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf,Cab,Car,Cbrown,Cw+step['Cw'],Cm)

r90_numerical.append((delta_r90_up-r90)/step['Cw'])
t90_numerical.append((delta_t90_up-t90)/step['Cw'])

_,delta_r90_up,delta_t90_up=Prospect5(N_leaf,Cab,Car,Cbrown,Cw,Cm+step['Cm'])

r90_numerical.append((delta_r90_up-r90)/step['Cm'])
t90_numerical.append((delta_t90_up-t90)/step['Cm'])
plt.close("all")
r90_numerical=np.asarray(r90_numerical)
t90_numerical=np.asarray(t90_numerical)
for i,param in enumerate(Params):
    plt.figure()
    plt.plot(wls,r90_analytical[i,:],label='Jacobian')
    plt.plot(wls,r90_numerical[i,:],label='Numerical')
    plt.title(param+' r90') 
    plt.legend()
    plt.figure()
    plt.plot(wls,t90_analytical[i,:],label='Jacobian')
    plt.plot(wls,t90_numerical[i,:],label='Numerical')
    plt.title(param+' t90')
    plt.legend()    
    
