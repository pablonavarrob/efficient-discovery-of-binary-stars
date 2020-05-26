import multiprocessing
import sys
import os
import pandas as pd
import numpy as np
from analyseSpectralData_functions import *
from binary_synthesis import *
import directories
np.random.seed(1)

#  ___ Directory paths _________________________________________________________ Directory paths
# Initialize directories
dirs = directories.directories()

#  ___ Variables _______________________________________________________________ Variables
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]

FILENAME_SAMPLE = (dirs.samples_to_tsne + 'sample_to_tsne_range_{}.npy'.format(SELECTED_RANGES))
if os.path.isfile(FILENAME_SAMPLE) == True:
    print('File exists already, skipping..')
    exit()

else:
    pass

#  ___ Triggers ________________________________________________________________ Triggers
LOAD_SINGLES_DATA = True
GENERATE_BINARIES = True # Needs to be true, otherwise the binaries do not match the ranges
LOAD_BINARIES_DATA = True

# _____ Load data ______________________________________________________________ Load data
# Specify the ranges of interest - also for the binary generation
binary_population = pd.read_csv(dirs.data + ('BinaryPopulation_5000_classicrelations'
'_allStarsDifferent_DucheneKraus13_q_IMF_exponential0,3-----FINALPOPULATION.csv'))

# Load the stellar parameters for ALL the stars, both singles and binaries
synthetic_stellar_parameters = pd.read_csv(
    dirs.data + 'stellar_parameters_duchenekrauspopulation.csv')

if LOAD_SINGLES_DATA == True:
    print('=' * 40)
    print('Loading synthetic single star data..')
    synthetic_single_fluxes = np.load(
        dirs.single_spectra + 'synthetic_single_spectra_{}.npy'.format(SELECTED_RANGES))
    number_single_synth_stars = len(synthetic_single_fluxes)
    # synthetic_single_parameters = pd.read_csv(
    #     dirs.data + 'single_synth_stars_parameters.csv')
    # print('=' * 40)
    # print('Parameters of single stars loaded')

    print('=' * 40)
    print('All of the synthetic single star data loaded')

# _____ Generate binaries ______________________________________________________ Generate binaries
if GENERATE_BINARIES == True:
    synthetic_binary_fluxes = []
    v_rad_difference = np.load(dirs.data + 'v_rad_difference_gaussian.npy')

    print('=' * 40)
    print('---> Synthesizing binary spectra...')
    for i in range(len(binary_population)):
        # Masks
        mask_a = synthetic_stellar_parameters['id_A'][0:number_single_synth_stars] \
            == binary_population['id_A'][i]
        mask_b = synthetic_stellar_parameters['id_A'][0:number_single_synth_stars] \
            == binary_population['id_B'][i]

        # Generate the actual binary fluxes
        synthetic_binary_f, v_rad_diff, binary_waveranges \
            = generate_spectra_binary(synthetic_single_fluxes[mask_a, :],
                                      synthetic_single_fluxes[
                mask_b, :],
                'beta', binary_population['lum ratio'][i], SELECTED_RANGES,
                normalize_main_star=True, input_v_rad=v_rad_difference[i])
        synthetic_binary_fluxes.append(synthetic_binary_f)
        # v_rad_difference.append(v_rad_diff)

    # Stack data and rearrange
    synthetic_binary_fluxes = np.vstack(synthetic_binary_fluxes)
    print('---> Binary spectra synthesis done')
    # binary_population['rad_vel'] = v_rad_difference

# _____ Load binary data _______________________________________________________ Load binary data
else:  # Load the binary data
    synthetic_binary_fluxes = np.load(
        dirs.data + 'synthetic_binary_spectra_duchenekrauspopulation_40nm.npy')
    print('=' * 40)
    print('Binary spectra loaded')

    # v_rad_difference = np.load(dirs.data + 'v_rad_difference_gaussian.npy')
    # binary_population['rad_vel'] = v_rad_difference
    #
    # # _____ Create parameter dataset ___________________________________________ Create dataset
    # synthetic_stellar_parameters = pd.concat(
    #     [synthetic_single_parameters, binary_population], axis=0, ignore_index=True, sort=False)
    # synthetic_stellar_parameters['binarity'] = np.concatenate((
    #     np.full(number_single_synth_stars, 0), np.full(len(v_rad_difference), 1)))

# _____ Create t-SNE data ______________________________________________________ Create t-SNE data
# Corrects the lenght of the single spectra arrays to match
# that of the binary stars, which is shorter due to interpolation
print('=' * 40)
print('Creating final t-SNE sample')
indexes_list = 0
cut_single_spectra = []
for range in SELECTED_RANGES:
    waverange_single = get_waverange(range)
    spectra_chunk = synthetic_single_fluxes[:, indexes_list + 125:
                       indexes_list + len(waverange_single) - 125]
    indexes_list += len(waverange_single)
    cut_single_spectra.append(spectra_chunk)

del synthetic_single_fluxes
cut_single_spectra = np.array(cut_single_spectra)[0]

sample_tsne = np.round(np.vstack((cut_single_spectra, synthetic_binary_fluxes)), 3)

del cut_single_spectra
del synthetic_binary_fluxes

np.save(
    dirs.samples_to_tsne + 'sample_to_tsne_range_{}'.format(SELECTED_RANGES), sample_tsne)

# The stellar parameters are the SAME for all the generated
# samples, also the radial velocities, in order to be able to
# compare later the results from tSNE

print('=' * 40)
print('Sample t-SNE succesfully created')
