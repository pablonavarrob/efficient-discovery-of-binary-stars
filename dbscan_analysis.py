# Main script for the project to explore t-SNE'd data
# The main script for running everything will be a bash script
import dbscan_ratio
import spectra
from sklearn.cluster import DBSCAN
import directories
import numpy as np
import pandas as pd
import os
import sys

#  ___ Variables _______________________________________________________________ Variables
# These are neede for the naming scheme and for the run itself
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]
SNR = int(sys.argv[3])
PERPLEXITY = int(sys.argv[4])

#  ___ Directory paths _________________________________________________________ Directory paths
dirs = directories.directories()

# ___ Import the t-SNE data ____________________________________________________ Import t-SNE results
print('='*40)
print('Loading data..')

tsne_data = np.load(
    dirs.tsne_results + 'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy'.format(SELECTED_RANGES,
                                                                         PERPLEXITY, SNR))
stellar_parameters = pd.read_csv(
    dirs.data + 'stellar_parameters_duchenekrauspopulation.csv')

# DBSCAN parameters ____________________________________________________________
minEpsilon = 0.1
maxEpsilon = 0.75
min_minSamples = 25
max_minSamples = 125

# Run the main DBSCAN
dbscan = dbscan_ratio.dbscan_method(tsne_data[:, 0], tsne_data[:, 1],
                                    stellar_parameters['binarity'], minEpsilon,
                                    maxEpsilon, min_minSamples, max_minSamples,
                                    stellar_parameters, SELECTED_RANGES, SNR, PERPLEXITY)

FILENAME_DBSCAN_RESULT = (dirs.dbscan_results +
 'DBSCAN_parameterspace_range_{}_perplexity_{}_SNRof{}_ratio_{}_iterations_{}.csv'.format(SELECTED_RANGES,
                                                        PERPLEXITY, SNR, dbscan.ratio, dbscan.iterations))


if os.path.isfile(FILENAME_DBSCAN_RESULT) == True:
    print('File exists already, skipping this analysis..')
    exit()

else:
    dbscan.normalize_data_tSNE()
    dbscan.iterations = 10
    dbscan.ratio = 0.9  # To make sure is set to 0.9 - a bit redundant
    dbscan.explore_parameter_space()
    # dbscan.parameter_space = pd.read_csv(
    #         dirs.data + 'DBSCAN_parameterspace_range_{}_perplexity_{}_SNRof{}_ratio_{}.csv'.format(SELECTED_RANGES,
    #                                                                                 PERPLEXITY, SNR, dbscan.ratio))
    # Save the results from the parameter space exploration
    (dbscan.parameter_space).to_csv(
        dirs.dbscan_results + 'DBSCAN_parameterspace_range_{}_perplexity_{}_SNRof{}_ratio_{}_iterations_{}.csv'.format(SELECTED_RANGES,
                                                                                  PERPLEXITY, SNR, dbscan.ratio, dbscan.iterations),
                                                                                  index=False)

# print("="*40)
# print("Plotting...")
# dbscan.plot_tsne_maps()
# dbscan.plot_parameter_space()
# dbscan.plot_histograms()
# print("="*40)
# print("Done!")
