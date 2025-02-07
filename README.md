# pyPro4Sail
Vectorized vesions of the ProspectD and 4SAIL Radiative Transfer Models for simulating the transmission of radiation in leaves and canopies.

## Synopsis

This project contains *Python* code for *Prospect* and *4SAIL* Radiative Transfer Models (**RTM**)
for simulating the transmission of optical and thermal electromagnetic radiation through 
leaves and vegetated canopies.

The package also include helpers for inverting the models, either using gradient-based optimization algorithms or regression-based approaches based of forward simulations.

The project consists of: 

1. lower-level modules with the basic functions needed in *Prospec5* and *4SAIL* RTMs.

2. higher-level scripts for easily running ProSAIL in both forward and inverse mode.

## Installation

Download the project to your local system, enter the download directory and then type.

`python setup.py install` 

if you want to install pyTSEB and its low-level modules in your Python distribution. 

The following Python library ir required for running Prospect and 4SAIL:

- Numpy

In addition, the inversion of both RTMS requires.

- Scipy

- cma [Optional]


## Code Example
### High-level example
You can automatically run the coupled leaf+canopyt Prospect5+4SAIL RTM with *pyPro4Sail.py* module.

```python
[N, chloro, caroten, brown, EWT, LMA, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth, LIDF]=[1.5,40,8,0.0,0.01,0.009,3,0.01,30,180,10,180,(-0.35,-0.15)]
import pyPro4SAIL
wl,rho=pyPro4SAIL.run(N, chloro, caroten, brown, EWT, LMA, LAI, hot_spot, solar_zenith, solar_azimuth, view_zenith, view_azimuth, LIDF, skyl=0.2, soilType=pyPro4SAIL.DEFAULT_SOIL)
```

Also it is possible to simulate the surface land-leaving thermal radiance with the function `run_TIR`.

### Low-level example
#### Prospect5 RTM
You can run *Prospect* by importing the module Prospect5.py and then either calling the function `Prospect5` 
for simulating the full optical spectrum (400-2500nm), or the function `Prospect5_wl` for simulating
the leaf reflectance and transmittance for a given wavelength.

```python
# Running Prospect5
import Prospect5
# Simulate leaf full optical spectrum (400-2500nm) 
wl, rho_leaf, tau_leaf = Prospect5.Prospect5(N, chloro, caroten, brown, EWT, LMA)

```

You can type
`help(Prospect5.Prospect5)`
to understand better the inputs needed and the outputs returned

#### 4SAIL RTM
You can run *4SAIL* by importing the module FourSAIL.py and then either calling the function `FourSAIL` 
for simulating the reflectance and transmittance factor of a given canopy given a list of leaf reflectances 
and trasmittances, or you can call the function `FourSAIL_wl` for simulating the leaf reflectance and transmittance 
factor of a given canopy at for a single wavelenght.

```python
# Running the coupled Prospect and 4SAIL
import Prospect5, FourSAIL
# Simulate leaf full optical spectrum (400-2500nm) 
wl, rho_leaf, tau_leaf = Prospect5.Prospect5(N, chloro, caroten, brown, EWT, LMA)
# Estimate the Leaf Inclination Distribution Function of a canopy
LIDF = FourSAIL.CalcLIDF_Campbell(alpha)
# Simulate leaf reflectance and transmittance factors of a canopy 
tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammasowl = FourSAIL.FourSAIL(LAI,hot_spot,LIDF,solar_zenith,view_zenith,solar_azimuth-view_azimuth,rho_leaf,tau_leaf,rho_soil)
# Simulate the canopy reflectance factor for a given difuse/total radiation condition (skyl)
rho_canopy = rdot*skyl+rsot*(1-skyl)
``` 

You can type
`help(FourSAIL.FourSAIL)`
to understand better the inputs needed and the outputs returned
   
## Basic Contents
### High-level modules
- *.src/pyPro4SAIL.py*
> Runs the coupled Prospect5+4SAIL to estimate the canopy directional reflectance factor and 4SAIL to estimate the land-leaving broadband thermal radiance.

### Low-level modules
The low-level modules in this project are aimed at providing customisation and more flexibility in running TSEB. 
The following modules are included.

- *.src/Prospect5.py*
> core functions for running Prospect5 Leaf Radiative Transfer Model. 

- *.src/Prospect5Jacobian.py*
> core functions for computing the Jacobian of Prospect5 Leaf Radiative Transfer Model. 

- *.src/FourSAIL.py*
> core functions for running 4SAIL Canopy Radiative Transfer Model.

- *.src/FourSAILJacobian.py*
> core functions for computing the Jacobian of 4SAIL Canopy Radiative Transfer Model.

- *.src/CostFunctionsPROSPECT4SAIL.py*
> merit functions used to invert Prospect and/or 4SAIL from a given spectrum

- *.src/cma.py*
> Covariance Matrix Adaptation Evolution Strategy optimization method for inverting Prospect5 and/or 4SAIL.


## API Reference
http://pyPro4Sail.readthedocs.org/en/latest/index.html

## Main Scientific References
- S. Jacquemoud, F. Baret, PROSPECT: A model of leaf optical properties spectra, Remote Sensing of Environment, Volume 34, Issue 2, November 1990, Pages 75-91, ISSN 0034-4257, http://dx.doi.org/10.1016/0034-4257(90)90100-Z.
- Jean-Baptiste Feret, Christophe François, Gregory P. Asner, Anatoly A. Gitelson, Roberta E. Martin, Luc P.R. Bidel, Susan L. Ustin, Guerric le Maire, Stéphane Jacquemoud, PROSPECT-4 and 5: Advances in the leaf optical properties model separating photosynthetic pigments, Remote Sensing of Environment, Volume 112, Issue 6, 16 June 2008, Pages 3030-3043, ISSN 0034-4257, http://dx.doi.org/10.1016/j.rse.2008.02.012.
- W. Verhoef, Light scattering by leaf layers with application to canopy reflectance modeling: The SAIL model, Remote Sensing of Environment, Volume 16, Issue 2, October 1984, Pages 125-141, ISSN 0034-4257, http://dx.doi.org/10.1016/0034-4257(84)90057-9.
- W. Verhoef, L. Jia, Q. Xiao and Z. Su, Unified Optical-Thermal Four-Stream Radiative Transfer Theory for Homogeneous Vegetation Canopies, IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 6, pp. 1808-1822, June 2007. http://dx.doi.org/10.1109/TGRS.2007.895844.

## Tests
*to be included*

## Contributors
- **Hector Nieto** <hnieto@ias.csic.es> <hector.nieto.solana@gmail.com> main developer
- **Radoslaw Guzinski** 
- **Robin Wilson** <robin@rtwilson.com> main developer of pyProSail <https://github.com/robintw/PyProSAIL>

## License
pyPro4Sail: a Python Two Source Energy Balance Model

Copyright 2016 Hector Nieto and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
