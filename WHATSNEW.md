# What is new from [PyProSAIL](https://github.com/robintw/PyProSAIL).
* `Prospect5` and `4SAIL` were fully translated into Python.
* `PyPro4Sail.run` allows using different types of soil spectrum.
* the folder *SoilSpectralLibrary* contains all the soil spectra from the [ASTER Spectral Library](http://speclib.jpl.nasa.gov/) processed to be used directly in `PyProSAIL` and `FourSAIL`.
* the ratio of diffuse radiation `skyl` is an input in `PyPro4Sail.run`.
* `PyPro4Sail.run_TIR` allows simulating the thermal component of 4SAIL.

# What is new from the Fortran version of PROSPECT_5B and PROSAIL_5B at http://teledetection.ipgp.jussieu.fr/prosail/.
* use of the exponential integral `expn` of `scipy` in `Prospect5`.
* Allow the simulation of a full spectrum using `numpy` of single wavelengths. The later might be used in inversion problems enhancing the computation speed in observations with few bands.

# What is new from the Fortran version of PROSAIL_5B at http://teledetection.ipgp.jussieu.fr/prosail/.
* the ratio of diffuse radiation `skyl` is an optional input, instead of being computed internally based on solar zenith angle.
* the canopy reflectance factor is computed as  `rho_canopy=rdot*skyl+rsot*(1-skyl)` instead of 
```
PARdifo=skyl*Ed
PARdiro=(1-skyl)*Es
rho_canopy=(rdot*PARdifo+rsot*PARdiro)/(PARdifo+PARdiro)
```
the latter somehow considers twice the ratio of diffuse radiation since `skyl=Ed/(Es+Ed)`.
*  `Prospect5` and `4SAIL` in Python provides more flexibility in extracting the internal parameters of both models, allowing an easier framework for estimating additional variables, such as albedo, fraction of intercepted and absorbed radiation, etc.
* A link to the Two Source Energy Balance Model [pyTSEB](https://github.com/hectornieto/pyTSEB) for using Prospect and 4SAIL in the retrieval of net radiation and/or the inversion of dual-angle radiometric temperature for the retrieval of soil and canopy temperatures.
