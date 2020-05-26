import sys
import pandas as pd
import os
import numpy as np
from analyseSpectralData_functions import *
from binary_synthesis import *
import directories
np.random.seed(1)

#  ___ Variables _______________________________________________________________ Variables
RUNNAME = 'Galah_wholerange_dwarfs' # For the genereated stars
READ_FITS = True

#  ___ Directory paths _________________________________________________________ Directory paths
dirs = directories.directories()
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]

# Check for the files
FILENAME_SPECTRA = (dirs.single_spectra + 'synthetic_single_spectra_{}.npy'.format(SELECTED_RANGES))
if os.path.isfile(FILENAME_SPECTRA) == True:
    print('File exists already, skipping this step..')
    exit()

else:
    pass

#  ___ Read and cut fits _______________________________________________________ Read and cut fits
# Read the fits files and cut them in the selected waveranges
if READ_FITS == True:
    synthetic_single_fluxes, single_waveranges, corrupt_files = read_spectra_fits(
        dirs.single_synth_spectra, SELECTED_RANGES, run_name=RUNNAME)
print('='*40)

del single_waveranges
del corrupt_files

print('Fits files read and cut succesfully, storing results in a separate file...')

np.save(dirs.single_spectra + 'synthetic_single_spectra_{}.npy'.format(SELECTED_RANGES),
        np.vstack(synthetic_single_fluxes))
