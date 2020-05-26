import os
import sys
import directories

#  ___ Variables _______________________________________________________________ Variables
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]
FILE_TO_REMOVE = sys.argv[3]

#  ___ Directories _____________________________________________________________ Directories
dirs = directories.directories()

filename_synthetic_spectra = (dirs.single_spectra +
             'synthetic_single_spectra_{}.npy'.format(SELECTED_RANGES))


#  ___ Delete files if exist ___________________________________________________ File delete

if FILE_TO_REMOVE == 'spectra':
    # Delete the cut synthetic spectra file
    if os.path.isfile(filename_synthetic_spectra) == True:
        os.remove(filename_synthetic_spectra)
    print('Deleting spectra..')

print('Removed the {} file from the disk'.format(FILE_TO_REMOVE))
print('=' * 40)
