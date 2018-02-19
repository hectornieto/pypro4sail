# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:56:25 2016

@author: hector
"""
import numpy as np
from sknn.mlp import Regressor, Layer
import pickle
from sknn.platform import gpu32
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pyPro4Sail.ProspectD as ProspectD 
import pyPro4Sail.FourSAIL as FourSAIL 
from numpy.random import rand,randint
from scipy.ndimage.filters import gaussian_filter1d


def TrainANN(X_array,Y_array, dropout_rate=0.25,learning_rate=0.3, momentum=0.99,hidden_layers=['Sigmoid'],
             inputScale=True, outputScale=True,reducePCA=True,outfile=None):


    Y_array=np.asarray(Y_array)
    X_array=np.asarray(X_array)

    pca=None
    scalerInput=None
    scalerOutput=None
    
    # Normalize the input
    if type(inputScale)==type(list()):
        inputScale=np.asarray(inputScale)
        X_array=(X_array-inputScale[:,0])/inputScale[:,1]
        inputScale=list(inputScale)
    elif inputScale==True:   
        scalerInput=StandardScaler()
        scalerInput.fit(X_array)
        X_array=scalerInput.transform(X_array)
        if outfile:
            fid=open(outfile+'_scalerInput','wb')
            pickle.dump(scalerInput,fid,-1)
            fid.close()

    if reducePCA:
        #Reduce input variables using Principal Component Analisys
        pca = PCA(n_components=10)
        pca.fit(X_array)
        X_array=pca.transform(X_array)
        if outfile:
            fid=open(outfile+'_PCA','wb')
            pickle.dump(pca,fid,-1)
            fid.close()

    # Normalize the output
    if type(outputScale)==type(list()):
        outputScale=np.asarray(outputScale)
        Y_array=(Y_array-outputScale[:,0])/outputScale[:,1]
        scalerOutput=list(outputScale)
    elif outputScale==True:   
        scalerOutput=StandardScaler()
        scalerOutput.fit(Y_array)
        Y_array=scalerOutput.transform(Y_array)
        if outfile:
            fid=open(outfile+'_scalerOutput','wb')
            pickle.dump(scalerOutput,fid,-1)
            fid.close()

    # Get the number of bands to set the ANN structure
    n_bands=X_array.shape[1] 
    n_outputs=Y_array.shape[1] 
    layers=[]
    # Include the hidden layers
    if type(hidden_layers)==type(list()):
        for factor,actFunc in enumerate(hidden_layers):
            layers.append(Layer(actFunc, units=int(n_bands/(factor+1)),dropout=dropout_rate))
    elif type(hidden_layers)==type(dict()) or type(hidden_layers)==type(OrderedDict()):
        for layerid,[actFunc,nodes] in hidden_layers.items():
            layers.append(Layer(actFunc, units=int(nodes),dropout=dropout_rate))
    else:
        raise TypeError('hidden_layers must be either a list or a dictionary')
    # Include the output layer
    layers.append(Layer('Linear', units=n_outputs))
    ann=Regressor(layers,verbose=True,learning_momentum=momentum,learning_rate=learning_rate,
                  regularize='dropout',batch_size=10000,learning_rule='sgd')
    
    ANN=ann.fit(X_array, Y_array)
    if outfile:    
        fid=open(outfile,'wb')
        pickle.dump(ANN,fid,-1)
        fid.close()
    
    return ANN, scalerInput,scalerOutput,pca

def TestANN(X_array,Y_array, annObject, scalerInput=None,scalerOutput=None,pca=None, 
            outfile=None,ObjParamName=None):
    
    
    X_array=np.asarray(X_array)
    Y_array=np.asarray(Y_array)
    
    # Estimate error measuremenents
    RMSE=[]
    bias=[]
    cor=[]
    
    if scalerInput:
        if type(scalerInput)==type(list()):
            scalerInput=np.asarray(scalerInput)
            X_array=(X_array-scalerInput[:,0])/scalerInput[:,1]
            scalerInput=list(scalerInput)
        else:   
            X_array=scalerInput.transform(X_array)
    if pca:
        X_array=pca.transform(X_array)
    
    Y_test=annObject.predict(X_array)
    
    if scalerOutput:
        if type(scalerOutput)==type(list()):
            scalerOutput=np.asarray(scalerOutput)
            Y_test=(Y_test*scalerOutput[:,1])+scalerOutput[:,0]
            scalerOutput=list(scalerOutput)
        else:   
            Y_test=scalerOutput.inverse_transform(Y_test)
    n_outputs=Y_test.shape[1] 

    if not ObjParamName:
        ObjParamName=[str(i) for i in range(n_outputs)]
    f,axarr=plt.subplots(n_outputs)
    for i,param in enumerate(ObjParamName):
        RMSE.append(np.sqrt(np.sum((Y_array[:,i]-Y_test[:,i])**2)/np.size(Y_array[:,i])))
        bias.append(np.mean(Y_array[:,i]-Y_test[:,i]))
        cor.append(pearsonr(Y_array[:,i],Y_test[:,i])[0])
        axarr[i].scatter(Y_array[:,i],Y_test[:,i], color='black', s=5,alpha=0.1, marker='.')
        absline=np.asarray([[np.amin(Y_array[:,i]),np.amax(Y_array[:,i])],[np.amin(Y_array[:,i]),np.amax(Y_array[:,i])]])
        axarr[i].plot(absline[0],absline[1],color='red')
        axarr[i].set_title(param)
    if outfile:        
        f.savefig(outfile)
    plt.close()
    
    return RMSE,bias,cor
    
def SimulateProSAIL_LUT(n_simulations,wls_sim,rsoil,skyl=0.1,sza=37,vza=0,psi=0,
        fwhm=None,outfile=None,
        ObjParam=['N_leaf','Cab','Car','Cbrown','Cm','Cw', 'Ant','LAI', 'leaf_angle','hotspot','fAPAR'],
        param_bounds={'N_leaf':[1.,3.],'Cab':[0.0,100.0],'Car':[0.0,40.0],'Cbrown':[0.0,1.0],
        'Cw':[0.001,0.04],'Cm':[0.001,0.05],'Ant':[0.00,100],'LAI':[0.0,8.0],'leaf_angle':[34.0,73.0],
        'hotspot':[0.0,1.0]}):
            
    
    print ('Starting Simulations... 0% done')
    progress=n_simulations/10
    percent=10
    X_array=[]
    Y_array=[]
    # Get the number of soil spectra
    rsoil=np.asarray(rsoil)
    n_soils=rsoil.shape
    if len(n_soils)==1:
        n_soils=1
    else:
        n_soils=n_soils[0]
    for case in range(n_simulations):
        if case>=progress:
            print(str(percent) +'% done')
            percent+=10
            progress+=n_simulations/10
        input_param=dict()
        for param in param_bounds:
            rnd=rand()
            input_param[param]=param_bounds[param][0]+rnd*(param_bounds[param][1]-param_bounds[param][0])
        if n_soils>1:
            rho_soil=rsoil[randint(0,n_soils),:]
        else:
            rho_soil=np.asarray(rsoil)
        # Calculate the lidf
        lidf=FourSAIL.CalcLIDF_Campbell(input_param['leaf_angle'])
        #for i,wl in enumerate(wls_wim):
        [wls,r,t]=ProspectD.ProspectD(input_param['N_leaf'],
                input_param['Cab'],input_param['Car'],input_param['Cbrown'], 
                input_param['Cw'],input_param['Cm'],input_param['Ant'])
        #Convolve the simulated spectra to a gaussian filter per band
        if fwhm:
            sigma=FWHM2Sigma(fwhm)
            r=gaussian_filter1d(r,sigma)
            t=gaussian_filter1d(t,sigma)
        else:
            r=np.asarray(r)
            t=np.asarray(t)
        rho_leaf=[]
        tau_leaf=[]
        for wl in wls_sim:
            rho_leaf.append(float(r[wls==wl]))
            tau_leaf.append(float(t[wls==wl]))
        rho_leaf=np.asarray(rho_leaf)
        tau_leaf=np.asarray(tau_leaf)
                        
        [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
                 rsodt,rsost,rsot,gammasdf,gammasdb,
                 gammaso]=FourSAIL.FourSAIL(input_param['LAI'],input_param['hotspot'],
                    lidf,sza,vza,psi,rho_leaf,tau_leaf,rho_soil)
        if 'fAPAR' in ObjParam:
            par_index=wls_sim<=700
            fAPAR,fIPAR=CalcfAPAR_4SAIL (skyl[par_index],input_param['LAI'],lidf,
                                         input_param['hotspot'],sza,rho_leaf[par_index],
                                        tau_leaf[par_index],rho_soil[par_index])
            if fAPAR==np.nan:fAPAR=0
        r2=rdot*skyl+rsot*(1-skyl)
        X_array.append(r2)
        Y=[]
        for param in ObjParam:
            if param=='fAPAR':
                Y.append(fAPAR)
            else:
                Y.append(input_param[param])
        Y_array.append(Y)
   
    X_array=np.asarray(X_array)
    Y_array=np.asarray(Y_array)
    if outfile:
        fid=open(outfile+'_rho','wb')
        pickle.dump(X_array,fid,-1)
        fid.close()
        fid=open(outfile+'_param','wb')
        pickle.dump(Y_array,fid,-1)
        fid.close()
    return X_array,Y_array

def CalcfAPAR_4SAIL (skyl,LAI,lidf,hotspot,sza,rho_leaf,tau_leaf,rsoil):
    '''Estimates the fraction of Absorbed PAR using the 4SAIL 
         Radiative Transfer Model.
        
    Parameters
    ----------
    skyl : float
        Ratio of diffuse to total PAR radiation
    LAI : float
        Leaf (Plant) Area Index
    lidf : list
        Leaf Inclination Distribution Function, 5 degrees step
    hotspot : float
        hotspot parameters, use 0 to ignore the hotspot effect (turbid medium)
    sza : float
        Sun Zenith Angle (degrees)
    rho_leaf : list
        Narrowband leaf bihemispherical reflectance, 
            it might be simulated with PROSPECT (400-700 @1nm)
    tau_leaf : list
        Narrowband leaf bihemispherical transmittance, 
            it might be simulated with PROSPECT (400-700 @1nm)
    rsoil : list
        Narrowband soil bihemispherical reflectance (400-700 @1nm)
    
    Returns
    -------
    fAPAR : float
        Fraction of Absorbed Photosynthetically Active Radiation'''
    
    #Input the angle step to perform the hemispherical integration
    stepvza=10
    steppsi=30
    Es=1.0-skyl
    Ed=skyl
    #Initialize values
    So_sw=np.zeros(len(rho_leaf))
    # Start the hemispherical integration
    vzas_psis=((vza,psi) for vza in range(5,90,stepvza) for psi in range(0,360,steppsi))
    for vza,psi in vzas_psis:
        [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt, 
             rsdt, rdot, rsodt, rsost, rsot,gammasdf,gammasdb,gammaso]=FourSAIL.FourSAIL(LAI,
            hotspot,lidf,sza,vza,psi,rho_leaf,tau_leaf,rsoil)
        # Downwelling solar beam radiation at ground level (beam transmissnion)
        Es_1=tss*Es
        # Upwelling diffuse shortwave radiation at ground level
        Ed_up_1=(rsoil*(Es_1+tsd*Es+tdd*Ed))/(1.-rsoil*rdd)
        # Downwelling diffuse shortwave radiation at ground level            
        Ed_down_1=tsd*Es+tdd*Ed+rdd*Ed_up_1
        # Upwelling shortwave (beam and diffuse) radiation towards the observer (at angle psi/vza)
        Eo_0=(rso*Es+rdo*Ed+tdo*Ed_up_1+too*Ed_up_1)
        # Calculate the reflectance factor and project into the solid angle
        cosvza   = np.cos(np.radians(vza))
        sinvza   = np.sin(np.radians(vza))
        # Spectral flux at the top of the canopy        
        So=Eo_0*cosvza*sinvza*np.radians(stepvza)*np.radians(steppsi)
        # Spectral flux at ground lnevel
        Sn_soil=(1.-rsoil)*(Es_1+Ed_down_1) # narrowband net soil shortwave radiation
        # Get the to of the canopy flux (as it is not Lambertian)
        So_sw_dir=So/np.pi
        # Add the top of the canopy flux to the integral and continue through the hemisphere
        So_sw=So_sw+So_sw_dir
    # Calculate the vegetation (sw/lw) net radiation (divergence) as residual of top of the canopy and net soil radiation
    Esw=Es+Ed    
    Rn_sw_veg_nb=Esw-So_sw-Sn_soil  # narrowband net canopy sw radiation
    fAPAR=sum(Rn_sw_veg_nb)/len(Rn_sw_veg_nb)   # broadband net canopy sw radiation
    fIPAR=1.0-(Es_1+Ed_down_1)/Esw
    fIPAR=sum(fIPAR)/len(fIPAR)
    return fAPAR,fIPAR


def FWHM2Sigma(fwhm):

    sigma=fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
    return sigma