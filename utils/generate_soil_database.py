# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:49:17 2018

@author: hnieto
"""

import os
import os.path as pth
import numpy as np
import pickle
import glob

workdir=os.getcwd()

input_dir=pth.join(workdir,'SoilSpectralLibrary')

soil_files=glob.glob(pth.join(input_dir,'*.txt'))

out_file=pth.join(workdir,'pyPro4Sail','soil_spectra')


soil_dict={}
soil_classes=[]
soil_types=[]
soil_texture=[]
soil_id=[]
soil_spectra=[]
soil_origin=[]
for spectrum in soil_files:
    
    filename=pth.basename(spectrum)
    fields=filename.split('.')
    
    soil_origin.append('.'.join(fields[0:2]))
    
    soil_classes.append(fields[3])
    soil_types.append(fields[4])
    soil_texture.append(fields[5])
    soil_id.append(fields[6])
    
    spectrum=np.genfromtxt(spectrum,names=['wl','rho'],dtype=None)
    soil_spectra.append(spectrum['rho'])
    
database=zip(tuple(soil_classes),soil_types,soil_id,soil_texture,soil_origin)
soil_spectra=np.asarray(soil_spectra)

np.savetxt(out_file+'txt',soil_spectra)
fid=open(pth.join(out_file),'wb')
pickle.dump(soil_spectra,fid,-1)
fid.close()
    
    

