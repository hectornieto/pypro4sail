#!/usr/bin/env python
"""Spectral libraries for PROSPECT + SAIL
Adapted from prosail Python version by 
J Gomez-Dans (NCEO & UCL) j.gomez-dans@ucl.ac.uk
https://github.com/jgomezdans/prosail/blob/master/prosail/spectral_library.py

"""
import pkgutil
from collections import namedtuple
from io import BytesIO

import numpy as np

ProspectDSpectra = namedtuple('ProspectDSpectra','wl' 
                                'nr kab kcar kbrown kw km kant')

def get_spectra():
    """Reads the spectral information and stores is for future use."""

    # PROSPECT-D
    prospect_d_spectraf = pkgutil.get_data('pyPro4Sail', 'prospect_d_spectra.txt')
    wl, nr, kab, kcar, kant, kbrown, kw, km= np.loadtxt(
        BytesIO(prospect_d_spectraf), unpack=True)
    prospect_d_spectra = ProspectDSpectra(wl, nr, kab, kcar, kbrown, kw, km, kant)

    return prospect_d_spectra
