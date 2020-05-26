# Parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import directories
import dbscan_ratio

# Manage directories ###########################################################
dirs = directories.directories()

# Import the data ##############################################################
stellar_parameters = pd.read_csv(dirs.data +
        'stellar_parameters_duchenekrauspopulation.csv')

# Import the data ##############################################################
# Parameters
TOP = np.arange(475, 925, 25)
BOTTOM = np.arange(450, 900, 25)
PERPLEXITIES = [5, 15, 30, 100]
SIGNALTONOISE = [10, 25, 50, 100, 500]

# Generate dataframe with all the information from the parameter space exploration
# include as well the arrays themselves in the dataframe for the labels
parameter_space_exploration = []
for i, j in zip(BOTTOM, TOP):
    SELECTED_RANGES = [(float(i), float(j))]
    print(SELECTED_RANGES)
    for PERPLEXITY in PERPLEXITIES:
        for SNR in SIGNALTONOISE:
            tsne_data = np.load(('/Users/pablonavarrobarrachina/Desktop/Results FFTW/' +
                'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy').format(SELECTED_RANGES,
                PERPLEXITY, SNR))

            # ---> Initialize DBSCAN
            dbscan = dbscan_ratio.dbscan_method(tsne_data[:, 0], tsne_data[:, 1],
                stellar_parameters['binarity'], stellar_parameters, SELECTED_RANGES,
                SNR, PERPLEXITY)
            dbscan.normalize_data_tSNE()
            dbscan.parameter_space = pd.read_csv(('/Users/pablonavarrobarrachina/Desktop/' +
                'Results FFTW/DBSCAN_parameterspace_range_{}_perplexity_' +
                '{}_SNRof{}_ratio_{}_iterations_{}.csv').format(SELECTED_RANGES,
                PERPLEXITY, SNR, dbscan.ratio, dbscan.iterations))
            dbscan.get_variables_from_imported_data()

            # Append dictionaries with the results from the exploration
            exploration_entry = {'spectral_range': SELECTED_RANGES,
                                'perplexity': PERPLEXITY,
                                'snr': SNR,
                                'eps': dbscan.optimized_eps,
                                'minpts': dbscan.optimized_minpts,
                                'recovery': dbscan.optimized_recovery_ratio,
                                'ratio_labels': [int(i) for i in dbscan.ratio_labels]}

            parameter_space_exploration.append(exploration_entry)

parameter_space_exploration = pd.DataFrame(parameter_space_exploration)
parameter_space_exploration.to_csv('parameter_space_exploration_run_25mm_intervals_2.csv',
    index=False)
