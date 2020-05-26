import numpy as np
import pandas as pd
import directories as directories
import dbscan_ratio
import matplotlib.pyplot as plt

#  ___ Directory paths _________________________________________________________ Directory paths
dirs = directories.directories()
dirs.name_analysis_run = 'run_intervals_25nm'

SELECTED_RANGES = [(450.0, 475.0)]
for value in [10, 25, 50, 100, 500]:
    for perp in [5, 15, 30, 100]:
        SNR = value
        PERPLEXITY = perp

        #  ___ Load data _______________________________________________________________ Load the data
        tsne_data = np.load(
            dirs.tsne_results + 'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy'.format(SELECTED_RANGES,
                                                                                 PERPLEXITY, SNR))
        plt.figure(figsize=[15, 15])
        plt.scatter(tsne_data[:,0][0:99986], tsne_data[:,1][0:99986], s=1, c='k', alpha=0.5)
        plt.scatter(tsne_data[:,0][99986:104986], tsne_data[:,1][99986:104986], s=1, c='r')
        plt.savefig(dirs.figures + 'tsne{}_test_snr{}_perp{}.png'.format(SELECTED_RANGES, SNR, PERPLEXITY), dpi=300)
        plt.close()

SELECTED_RANGES = [(850.0, 875.0)]
for value in [10, 25, 50, 100, 500]:
    for perp in [5, 15, 30, 100]:
        SNR = value
        PERPLEXITY = perp

        #  ___ Load data _______________________________________________________________ Load the data
        tsne_data = np.load(
            dirs.tsne_results + 'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy'.format(SELECTED_RANGES,
                                                                                 PERPLEXITY, SNR))
        plt.figure(figsize=[15, 15])
        plt.scatter(tsne_data[:,0][0:99986], tsne_data[:,1][0:99986], s=1, c='k', alpha=0.5)
        plt.scatter(tsne_data[:,0][99986:104986], tsne_data[:,1][99986:104986], s=1)
        plt.savefig(dirs.figures + 'tsne{}_test_snr{}_perp{}.png'.format(SELECTED_RANGES, SNR, PERPLEXITY), dpi=300)
        plt.close()

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
dbscan.normalize_data_tSNE()
dbscan.parameter_space = pd.read_csv(
        dirs.dbscan_results + 'DBSCAN_parameterspace_range_{}_perplexity_{}_SNRof{}_ratio_{}_iterations_{}.csv'.format(SELECTED_RANGES,
                                                                                PERPLEXITY, SNR, dbscan.ratio, dbscan.iterations))
