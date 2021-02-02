import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import directories as directories
import matplotlib
import dbscan_ratio
from dbscan_ratio import niceplot
from analyseSpectralData_functions import add_noise
from analyseSpectralData_functions import get_waverange
matplotlib.use("agg") # For astrolabe
import matplotlib.pyplot as plt

dirs = directories.directories()

################################################################################
# Triggers and parameters ######################################################

load_tsne_dbscan_data = 0
load_data_spectra = 0

# Need to add noise before plotting - not on the saved file
# The data-sets contain spectra for both singles and binaries
SELECTED_RANGES = [[(525.0, 550.0)], [(675.0, 700.0)]]
SNR = [100]

# Spectra parameters
resolution              = 2.8e4
sampling                = 0.004 #((wave_top - wave_base)/resolution)/5
wave_b = 450
wave_t = 900
waverange = (np.arange(wave_b, wave_t, sampling)*10) # In Armstrongs

################################################################################
# Import the stellar data ######################################################
stellar_parameters = pd.read_csv(dirs.data +
        'stellar_parameters_duchenekrauspopulation.csv')

mask_binary = stellar_parameters['binarity'] == 1
mask_single = stellar_parameters['binarity'] == 0

################################################################################
# Plot simple binary stars #####################################################

# Get 3 different spectra of binaries depending on their radial velocities

fig, ax = plt.subplots(6, 2, figsize=[20, 24], tight_layout=True, sharex='col')
for j in range(2):
    # Load the data for the plots, LOL
    print(SELECTED_RANGES[j], j)
    synthetic_spectra = np.load(dirs.samples_to_tsne + 'sample_to_tsne_range_{}.npy'.format(SELECTED_RANGES[j]))
    # Noise is added at each iteration then
    print('Loaded')
    synthetic_spectra_noise = add_noise(100, synthetic_spectra)

    print("Added noise {} to sample in range {}".format(int(SNR[0]),
                                                        SELECTED_RANGES[j]))

    del synthetic_spectra
    binary_parameters = stellar_parameters[mask_binary].sort_values('rad_vel')
    binary_spectra = synthetic_spectra_noise[mask_binary]
    del synthetic_spectra_noise

    waverange_binaries = np.arange(SELECTED_RANGES[j][0][0],
                                   SELECTED_RANGES[j][0][1],
                                   sampling)[125:-125]

    binary_params = stellar_parameters[mask_binary].sort_values('rad_vel')
    # Array to sort the spectra from lowest to highest rad vel
    index_sort = binary_params.index.values
    # Now sort binary spectra
    binary_spectra_to_plot = binary_spectra[index_sort - 99986]
    del binary_spectra

    mask_luminosity = binary_params['lum ratio'] >= 0.8

    binary_spectra_to_plot = binary_spectra_to_plot[mask_luminosity]
    binaries_plot = (binary_params[mask_luminosity]).reset_index()
    indexes_binaries = np.linspace(0, 884, 6, dtype=int)
    plots_letters = ['a', 'b', 'c', 'd', 'e', 'f']
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    for i in range(6):
        index = indexes_binaries[i]
        print(i, j, SELECTED_RANGES[j], binaries_plot.iloc[index]['rad_vel'], index)
        ax[i, j].plot(waverange_binaries, binary_spectra_to_plot[index].T,
         lw=1.25, color='k')
        if j == 0:
            ax[i, 0].set_xlim(535, 540)
            ax[i, 0].set_ylim(0.1, 1.1)
            ax[i, 0].text(0.52, 0.165,
                (r'$\Delta v_r$ = {:.1f} km/s,'
                r' $L_B / L_A$ = {:.2f}').format(binaries_plot.iloc[index]['rad_vel'],
                binaries_plot.iloc[index]['lum ratio']),
                transform=ax[i, 0].transAxes,
                fontsize=18, verticalalignment='top', bbox=props)
        if j == 1:
            ax[i, 1].set_xlim(680, 685)
            ax[i, 1].set_ylim(0.675, 1.05)
            ax[i, 1].text(0.9, 0.165, '{})'.format(plots_letters[i]),
                transform=ax[i, 1].transAxes,
                fontsize=18, verticalalignment='top', bbox=props)

ax[0, 0].set_ylabel('Normalized Flux')
ax[1, 0].set_ylabel('Normalized Flux')
ax[2, 0].set_ylabel('Normalized Flux')
ax[3, 0].set_ylabel('Normalized Flux')
ax[4, 0].set_ylabel('Normalized Flux')
ax[5, 0].set_ylabel('Normalized Flux')
ax[5, 0].set_xlabel(r'Wavelength [nm]')
ax[5, 1].set_xlabel(r'Wavelength [nm]')
matplotlib.rcParams.update({'font.size': 24})
plt.savefig('synthetic_binaries_noise.png', dpi = 250)
