# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:08:44 2018

@author: hnieto
"""

# Import python libraries
import numpy as np # matrix calculations
import gdal # geospatial library
import os.path as pth # file management
import glob # recursive search
from scipy import stats as st # statistical analysis
from scipy import optimize as op # numerical optimization
from matplotlib import pyplot as plt # plotting

    
# Set the working folder
workdir = pth.join("D:", "POTENTIAL", "Drone image")

plot_fit = True

# Set the folder where the different bands are stored
blue_folder = pth.join(workdir, "blue mask", "potato")
green_folder = pth.join(workdir, "green mask", "potato")
red_folder = pth.join(workdir, "red mask", "potato")
rededge_folder = pth.join(workdir, "rededge mask", "potato")
nir_folder = pth.join(workdir, "nir mask", "potato")

# Set the output folder where the imgaes are saved
output_folder = pth.join(workdir, "VIs", "potato")


baresoil_red_file = pth.join(workdir, "red mask", "potato", "2018_05_11_potato_red.tif")
baresoil_nir_file = pth.join(workdir, "nir mask", "potato", "2018_05_11_potato_nir.tif")

def main():
    
    ### extract the indices of the bare soil, since 2018/05/11, 
    baresoil_red, _, _ = read_image(baresoil_red_file)
    baresoil_nir, _, _ = read_image(baresoil_nir_file)
    
    # Remove all NaNs
    baresoil_red = baresoil_red[np.finite(baresoil_red)]
    baresoil_nir = baresoil_red[np.finite(baresoil_nir)]
    
    ## it is total soil, so extract top 0.1% pixel to represent the soil indices
    bs_red_max = np.percentile(baresoil_red, 0.001)
    bs_nir_max = np.percentile(baresoil_nir, 0.001)
    
    # Find all the NIR images
    nir_files = glob.glob(pth.join(nir_folder, "*.tif"))
    
    # Loop all the NIR images and start processing
    for nir_file in nir_files:
        # Extract the acquisition date
        nir_filename = pth.splitext(pth.basename(nir_file))[0]
        datestr = nir_filename[0:10]
        
        print('Processing date %s'%datestr)
        
        # Get all the bands for that date
        nir, prj, geo = read_image(nir_filename)
        
        filename = nir_file.replace('nir', 'red')
        red, _, _ = read_image(filename)
    
        filename = nir_file.replace('nir', 'green')
        green, _, _ = read_image(filename)
    
        filename = nir_file.replace('nir', 'blue')
        blue, _, _ = read_image(filename)
    
        filename = nir_file.replace('nir', 'rededge')
        rededge, _, _ = read_image(filename)
    
        #### calculate the indices
        rvi, ndvi, ndre = calc_indices(red, green, blue, rededge, nir)
        
        ### run the optimization model
        params = calibrate_fipar(rvi, red, nir, bs_red_max, bs_nir_max, plot_curve=plot_fit)
        fipar_est = rvi2fipar(rvi, a=params[0], b=params[1], c=params[2])
        
        # save images
        output_file = pth.join(output_folder , nir_filename.replace('nir', 'fipar'))
        saveImg(fipar_est, geo, prj, output_file, noDataValue=np.nan)

        output_file = pth.join(output_folder , nir_filename.replace('nir', 'rvi'))
        saveImg(rvi, geo, prj, output_file, noDataValue=np.nan)

        output_file = pth.join(output_folder , nir_filename.replace('nir', 'ndvi'))
        saveImg(ndvi, geo, prj, output_file, noDataValue=np.nan)

        output_file = pth.join(output_folder , nir_filename.replace('nir', 'ndre'))
        saveImg(ndre, geo, prj, output_file, noDataValue=np.nan)

def read_image(image_filename):
    ''' Reads a sigle band raster image and returns the values 
    and projection information'''
    
    ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    prj = ds.GetProjection()
    geo = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    
    return data, prj, geo

# save the data to geotiff or memory
def saveImg(data, geotransform, proj, outPath, noDataValue=np.nan, fieldNames=[]):

    driver = gdal.GetDriverByName("GTiff")
    driverOpt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']


    shape = data.shape
    if len(shape) > 2:
        ds = driver.Create(outPath, shape[1], shape[0], shape[2], gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        for i in range(shape[2]):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
            ds.GetRasterBand(i+1).SetNoDataValue(noDataValue)
    else:
        ds = driver.Create(outPath, shape[1], shape[0], 1, gdal.GDT_Float32, driverOpt)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetNoDataValue(noDataValue)


    print('Saved ' + outPath)

    return ds

def calc_indices(red, green, blue, rededge, nir):
    '''Calculates a bunch of vegetation indices from the Red Edge Camera'''
    
    ndvi = (nir - red) / (nir + red)
    rvi = nir / red
    ndre = (nir - rededge) / (nir + rededge)
    
    return rvi, ndvi, ndre

def rvi2fipar(rvi, a=1, b=1, c=1):
    fipar = a + b * rvi**c
    return fipar

def calibrate_fipar(rvi, red, nir, bs_red_max, bs_nir_max, plot_curve = False):
    
    def fipar_merit_function(params, rvi, fipar_obs):
        fipar_est = rvi2fipar(rvi, a=params[0], b=params[1], c=params[2])
        mse = 0.5 * np.sum((fipar_est - fipar_obs)**2)
        return mse
    
    #### Extract 0.01% RVI value and use which function to get the position
    rvi_max = rvi[np.finite(rvi)]
    rvi_max = np.percentile(rvi_max, 0.0001)
    
    ### calculate the mean  value at the 0.01% position
    red_max = np.mean(red[rvi > rvi_max])
    nir_max = np.mean(nir[rvi > rvi_max])
    
    #### calculate the indices
    n_red = (red_max - bs_red_max) / (bs_red_max - 1./red_max)
    n_nir = (nir_max - bs_nir_max) / (bs_nir_max - 1./nir_max)
    
    ### the by(interval matters a lot)
    fipar_origin = np.arange(0, 1.01, 0.01) 
    rvi_fipar = ((nir_max + (n_nir/nir_max) * (1-(fipar_origin))) 
                * (1+n_red*((1-(fipar_origin))^2))) \
                / ((red_max+(n_red/red_max)*((1-(fipar_origin))^2))
                *(1+n_nir*(1-(fipar_origin))))
    
    ####initialize values
    params0 = [1, 1, 1] #a0=1;b0=1;c0=1

    result = op.minimize(fipar_merit_function,
                         params0,
                         method = 'L-BFGS-B',
                         jac = False, 
                         args = [rvi_fipar, fipar_origin],
                         options = { 'maxiter': 100000}) 
        
    params = result.x
    
    ### calculate teh Fipar
    fipar_pred = rvi2fipar(rvi_fipar, a=params[0], b=params[1], c=params[2])
    
    if plot_curve:
        ### check the result
        slope, intercept, rvalue, pvalue, stderr = st.linregress(fipar_origin , fipar_pred)
        print('slope %s'%slope)
        print('intercept %s'%intercept)
        print('rvalue %s'%rvalue)
        print('pvalue %s'%pvalue)
        print('stderr %s'%stderr)
        plt.figure()
        plt.plot(fipar_pred, fipar_origin, 'ko')
        plt.figure()
        plt.plot(rvi_fipar, fipar_origin, 'k..')
        plt.plot(rvi_fipar, fipar_pred, 'r-')
    
    return params
    
if __name__ == '__main__':
    main()