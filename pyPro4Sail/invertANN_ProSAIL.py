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
import numpy.random as rnd
from scipy.ndimage.filters import gaussian_filter1d

UNIFORM_DIST=1
GAUSSIAN_DIST=2

def TrainANN(X_array,
             Y_array, 
             dropout_rate=0.25,
             learning_rate=0.3, 
             momentum=0.99,
             hidden_layers=['Sigmoid'],
             inputScale=True, 
             outputScale=True,
             reducePCA=True,
             outfile=None):


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
    ann=Regressor(layers,verbose=False,learning_momentum=momentum,learning_rate=learning_rate,
                  batch_size=10000,learning_rule='sgd',
                  #regularize='dropout'
                  )
    
    ANN=ann.fit(X_array, Y_array)
    if outfile:    
        fid=open(outfile,'wb')
        pickle.dump(ANN,fid,-1)
        fid.close()
    
    return ANN, scalerInput,scalerOutput,pca

def TestANN(X_array,
            Y_array, 
            annObject, 
            scalerInput=None,
            scalerOutput=None,
            pca=None, 
            outfile=None,
            ObjParamName=None):
    
    
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
    if n_outputs==1:
        RMSE.append(np.sqrt(np.sum((Y_array[:,0]-Y_test[:,0])**2)/np.size(Y_array[:,0])))
        bias.append(np.mean(Y_array[:,0]-Y_test[:,0]))
        cor.append(pearsonr(Y_array[:,0],Y_test[:,0])[0])
        axarr.scatter(Y_array[:,0],Y_test[:,0], color='black', s=5,alpha=0.1, marker='.')
        absline=np.asarray([[np.amin(Y_array[:,0]),np.amax(Y_array[:,0])],[np.amin(Y_array[:,0]),np.amax(Y_array[:,0])]])
        axarr.plot(absline[0],absline[1],color='red')
        axarr.set_title(ObjParamName[0])
    else:   
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

def build_prospect_database(n_simulations,
                           param_bounds={'N_leaf':(1.2,2.2),
                                         'Cab':(0.0,100.0),
                                         'Car':(0.0,40.0),
                                         'Cbrown':(0.0,1.0),
                                         'Cw':(0.003,0.011),
                                         'Cm':(0.003,0.011),
                                         'Ant':(0.00,40.)},
                           moments={'N_leaf':(1.5,0.3),
                                         'Cab':(45.0,30.0),
                                         'Car':(20.0,10.0),
                                         'Cbrown':(0.0,0.3),
                                         'Cw':(0.005,0.005),
                                         'Cm':(0.005,0.005),
                                         'Ant':(0.0,10.0)},
                            distribution={'N_leaf':GAUSSIAN_DIST,
                                         'Cab':GAUSSIAN_DIST,
                                         'Car':GAUSSIAN_DIST,
                                         'Cbrown':GAUSSIAN_DIST,
                                         'Cw':GAUSSIAN_DIST,
                                         'Cm':GAUSSIAN_DIST,
                                         'Ant':GAUSSIAN_DIST},
                            apply_covariate={'N_leaf':False,
                                         'Car':False,
                                         'Cbrown':False,
                                         'Cw':False,
                                         'Cm':False,
                                         'Ant':False},
                            covariate={'N_leaf':((1.2,2.2),(1.3,1.8)),
                                         'Car':((0,40),(20,40)),
                                         'Cbrown':((0,1),(0,0.2)),
                                         'Cw':((0.003,0.011),(0.005,0.011)),
                                         'Cm':((0.003,0.011),(0.005,0.011)),
                                         'Ant':((0,40),(0,40))},
                            outfile=None):
            
    
    print ('Build ProspectD database')
    input_param=dict()
    for param in param_bounds:
        if distribution[param]==UNIFORM_DIST:
            input_param[param]=param_bounds[param][0]+rnd.rand(n_simulations)*(param_bounds[param][1]-param_bounds[param][0])
        elif distribution[param]==GAUSSIAN_DIST:
            input_param[param]=moments[param][0]+ rnd.randn(n_simulations) * moments[param][1]
            input_param[param]=np.clip(input_param[param],param_bounds[param][0],param_bounds[param][1])
            
    # Apply covariates where needed
    LAI_range=param_bounds['Cab'][1]-param_bounds['Cab'][0]
    for param in apply_covariate:
        if apply_covariate:
            Vmin_lai=covariate[param][0][0]+input_param['Cab']*(covariate[param][1][0]-covariate[param][0][0])/LAI_range
            Vmax_lai=covariate[param][0][1]+input_param['Cab']*(covariate[param][1][1]-covariate[param][0][1])/LAI_range
            input_param[param]=np.clip(input_param[param], Vmin_lai, Vmax_lai)
            
            
    return input_param


def simulate_prospectD_LUT(input_param,
                        wls_sim,
                        srf=None,
                        outfile=None,
                        ObjParam=('N_leaf',
                           'Cab',
                           'Car',
                           'Cbrown',
                           'Cm',
                           'Cw',
                           'Ant')):            
    
    [wls,r,t]=ProspectD.ProspectD_vec(input_param['N_leaf'],
                input_param['Cab'], input_param['Car'],
                input_param['Cbrown'],input_param['Cw'],
                input_param['Cm'],input_param['Ant'])
    #Convolve the simulated spectra to a gaussian filter per band
    rho_leaf=[]
    tau_leaf=[]

    if srf:
        if type(srf)==float or type(srf)==int:
            wls=np.asarray(wls_sim)
            #Convolve spectra by full width half maximum
            sigma=FWHM2Sigma(srf)
            r=gaussian_filter1d(r,sigma)
            t=gaussian_filter1d(t,sigma)
            for wl in wls_sim:
                rho_leaf.append(float(r[wls==wl]))
                tau_leaf.append(float(t[wls==wl]))

        elif type(srf)==list or type(srf)==tuple:
            for weight in srf:
               rho_leaf.append(float(np.sum(weight*r)/np.sum(weight)))
               tau_leaf.append(float(np.sum(weight*t)/np.sum(weight)))

    else:
        rho_leaf=np.copy(r)                
        tau_leaf=np.copy(t)
        
    rho_leaf=np.asarray(rho_leaf)
    tau_leaf=np.asarray(tau_leaf)
                        
    
    if outfile:
        fid=open(outfile+'_rho','wb')
        pickle.dump(rho_leaf,fid,-1)
        fid.close()
        fid=open(outfile+'_param','wb')
        pickle.dump(input_param,fid,-1)
        fid.close()

    return rho_leaf,input_param


def build_prosail_database(n_simulations,
                           param_bounds={'N_leaf':(1.2,2.2),
                                         'Cab':(0.0,100.0),
                                         'Car':(0.0,40.0),
                                         'Cbrown':(0.0,1.0),
                                         'Cw':(0.003,0.011),
                                         'Cm':(0.003,0.011),
                                         'Ant':(0.00,40.),
                                         'LAI':(0.0,10.0),
                                         'leaf_angle':(30.0,80.0),
                                         'hotspot':(0.1,0.5)},
                           moments={'N_leaf':(1.5,0.3),
                                         'Cab':(45.0,30.0),
                                         'Car':(20.0,10.0),
                                         'Cbrown':(0.0,0.3),
                                         'Cw':(0.005,0.005),
                                         'Cm':(0.005,0.005),
                                         'Ant':(0.0,10.0),
                                         'LAI':(2.0,3.0),
                                         'leaf_angle':(60.0,30.0),
                                         'hotspot':(0.2,0.5)},
                            distribution={'N_leaf':GAUSSIAN_DIST,
                                         'Cab':GAUSSIAN_DIST,
                                         'Car':GAUSSIAN_DIST,
                                         'Cbrown':GAUSSIAN_DIST,
                                         'Cw':GAUSSIAN_DIST,
                                         'Cm':GAUSSIAN_DIST,
                                         'Ant':GAUSSIAN_DIST,
                                         'LAI':GAUSSIAN_DIST,
                                         'leaf_angle':GAUSSIAN_DIST,
                                         'hotspot':GAUSSIAN_DIST},
                            apply_covariate={'N_leaf':False,
                                         'Cab':False,
                                         'Car':False,
                                         'Cbrown':False,
                                         'Cw':False,
                                         'Cm':False,
                                         'Ant':False,
                                         'leaf_angle':False,
                                         'hotspot':False},
                            covariate={'N_leaf':((1.2,2.2),(1.3,1.8)),
                                         'Cab':((0,100),(45,100)),
                                         'Car':((0,40),(20,40)),
                                         'Cbrown':((0,1),(0,0.2)),
                                         'Cw':((0.003,0.011),(0.005,0.011)),
                                         'Cm':((0.003,0.011),(0.005,0.011)),
                                         'Ant':((0,40),(0,40)),
                                         'leaf_angle':((30,80),(55,65)),
                                         'hotspot':((0.1,0.5),(0.1,0.5))},
                                    
                            outfile=None):
            
    
    print ('Build ProspectD+4SAIL database')
    input_param=dict()
    for param in param_bounds:
        if distribution[param]==UNIFORM_DIST:
            input_param[param]=param_bounds[param][0]+rnd.rand(n_simulations)*(param_bounds[param][1]-param_bounds[param][0])
        elif distribution[param]==GAUSSIAN_DIST:
            input_param[param]=moments[param][0]+ rnd.randn(n_simulations) * moments[param][1]
            input_param[param]=np.clip(input_param[param],param_bounds[param][0],param_bounds[param][1])
            
    # Apply covariates where needed
    LAI_range=param_bounds['LAI'][1]-param_bounds['LAI'][0]
    for param in apply_covariate:
        if apply_covariate:
            Vmin_lai=covariate[param][0][0]+input_param['LAI']*(covariate[param][1][0]-covariate[param][0][0])/LAI_range
            Vmax_lai=covariate[param][0][1]+input_param['LAI']*(covariate[param][1][1]-covariate[param][0][1])/LAI_range
            input_param[param]=np.clip(input_param[param], Vmin_lai, Vmax_lai)
            
            
    return input_param
    
def SimulateProSAIL_LUT(input_param,
                        wls_sim,
                        rsoil,
                        skyl=0.1,
                        sza=37,
                        vza=0,
                        psi=0,
                        srf=None,
                        outfile=None,
                        calc_FAPAR=False,
                        reduce_4sail=False,
                        ObjParam=('N_leaf',
                           'Cab',
                           'Car',
                           'Cbrown',
                           'Cm',
                           'Cw',
                           'Ant',
                           'LAI', 
                           'leaf_angle',
                           'hotspot')):            
    
    print ('Starting Simulations... 0% done')
    # Convert input 
    n_simulations=input_param[ObjParam[0]].shape[0]
    progress=n_simulations/10
    percent=10
    X_array=[]
    # Get the number of soil spectra
    rsoil=np.asarray(rsoil)
    n_soils=rsoil.shape
    if len(n_soils)==1:
        n_soils=1
    else:
        n_soils=n_soils[0]
    if calc_FAPAR:
        FAPAR_array=[]
    for case in range(n_simulations):
        if case>=progress:
            print(str(percent) +'% done')
            percent+=10
            progress+=n_simulations/10

        if n_soils>1:
            rho_soil=rsoil[rnd.randint(0,n_soils),:]
        else:
            rho_soil=np.asarray(rsoil)
        # Calculate the lidf
        lidf=FourSAIL.CalcLIDF_Campbell(input_param['leaf_angle'][case])
        #for i,wl in enumerate(wls_wim):
        [wls,r,t]=ProspectD.ProspectD(input_param['N_leaf'][case],
                input_param['Cab'][case], input_param['Car'][case],
                input_param['Cbrown'][case],input_param['Cw'][case],
                input_param['Cm'][case],input_param['Ant'][case])
        #Convolve the simulated spectra to a gaussian filter per band
        rho_leaf=[]
        tau_leaf=[]

        if srf and reduce_4sail:
            if type(srf)==float or type(srf)==int:
                #Convolve spectra by full width half maximum
                sigma=FWHM2Sigma(srf)
                r=gaussian_filter1d(r,sigma)
                t=gaussian_filter1d(t,sigma)
                for wl in wls_sim:
                    rho_leaf.append(float(r[wls==wl]))
                    tau_leaf.append(float(t[wls==wl]))

            elif type(srf)==list or type(srf)==tuple:
                skyl_rho=[]
                for weight in srf:
                   rho_leaf.append(float(np.sum(weight*r)/np.sum(weight)))
                   tau_leaf.append(float(np.sum(weight*t)/np.sum(weight)))
                   skyl_rho.append(float(np.sum(weight*skyl)/np.sum(weight)))
                   
                skyl_rho=np.asarray(skyl_rho)
                
        elif reduce_4sail:
            r=np.asarray(r)
            t=np.asarray(t)
            for wl in wls_sim:
                rho_leaf.append(float(r[wl==wls]))
                tau_leaf.append(float(t[wl==wls]))
        else:
            rho_leaf=np.asarray(r)
            tau_leaf=np.asarray(t)
            skyl_rho=np.asarray(skyl)



        rho_leaf=np.asarray(rho_leaf)
        tau_leaf=np.asarray(tau_leaf)
                        
        [tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,
                 rsodt,rsost,rsot,gammasdf,gammasdb,
                 gammaso]=FourSAIL.FourSAIL(input_param['LAI'][case],
                                            input_param['hotspot'][case],
                                            lidf,sza,vza,psi,rho_leaf,
                                            tau_leaf,rho_soil)
        
        if type(skyl)==float:
            skyl_rho=skyl*np.ones(len(wls_sim))
                    
        r2=rdot*skyl_rho+rsot*(1-skyl_rho)  
        
        if calc_FAPAR:
            par_index=wls_sim<=700
            fAPAR,fIPAR=CalcfAPAR_4SAIL (skyl_rho[par_index],input_param['LAI'][case],lidf,
                                         input_param['hotspot'][case],sza,rho_leaf[par_index],
                                        tau_leaf[par_index],rho_soil[par_index])
            if fAPAR==np.nan:
                fAPAR=0
                
            FAPAR_array.append(fAPAR)
        
        rho_canopy=[]      
        if srf and not reduce_4sail:
            if type(srf)==float or type(srf)==int:
                #Convolve spectra by full width half maximum
                sigma=FWHM2Sigma(srf)
                r2=gaussian_filter1d(r2,sigma)
                for wl in wls_sim:
                    rho_canopy.append(float(r2[wls==wl]))

            elif type(srf)==list or type(srf)==tuple:
                for band in srf:
                   rho_canopy.append(float(np.sum(srf[band]*r2)/np.sum(srf[band])))
        elif reduce_4sail:
            rho_canopy=np.asarray(r2)
            
        else:
            for wl in wls_sim:
                rho_canopy.append(float(r2[wls==wl]))

        rho_canopy=np.asarray(rho_canopy)

        X_array.append(rho_canopy)            
    
    #Append FAPAR to dependent parameters array
    if calc_FAPAR:
        input_param['fAPAR']=FAPAR_array
 
    X_array=np.asarray(X_array)
    
    
    if outfile:
        fid=open(outfile+'_rho','wb')
        pickle.dump(X_array,fid,-1)
        fid.close()
        fid=open(outfile+'_param','wb')
        pickle.dump(input_param,fid,-1)
        fid.close()

    return X_array,input_param

def inputdict2array(input_param, 
                    ObjParam=('N_leaf',
                           'Cab',
                           'Car',
                           'Cbrown',
                           'Cm',
                           'Cw',
                           'Ant',
                           'LAI', 
                           'leaf_angle',
                           'hotspot')):
    
    Y_array=[]
    for param in ObjParam:
        Y_array.append(input_param[param])

    Y_array=np.asarray(Y_array).T
    
    return Y_array


def CalcfAPAR_4SAIL (skyl,
                     LAI,
                     lidf,
                     hotspot,
                     sza,
                     rho_leaf,
                     tau_leaf,
                     rsoil):
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