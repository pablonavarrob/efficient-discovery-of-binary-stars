# Parameters
import numpy as np
import pandas as pd
import matplotlib as mpl
# matplotlib.use("agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 22})
# matplotlib.rcParams['text.usetex'] = True
import directories
import dbscan_ratio
from dbscan_ratio import niceplot
import scaling_relations

# Manage directories ###########################################################
dirs = directories.directories()

################################################################################
# Import the data ##############################################################
stellar_parameters = pd.read_csv(dirs.data +
        'stellar_parameters_duchenekrauspopulation.csv')
mask_singles = stellar_parameters['binarity'] == 0
mask_binaries = stellar_parameters['binarity'] == 1

################################################################################
# Import the parameter space exploration #######################################
parameter_space_exploration = pd.read_csv(dirs.data +
    'parameter_space_exploration_run_25mm_intervals.csv')

TOP = np.arange(475, 925, 25)
BOTTOM = np.arange(450, 900, 25)

################################################################################
# Define the baseline model ####################################################
# These two parameters will serve as diagnostics
PERPLEXITY = 30
SNR = 100

################################################################################
# Triggers - define collpasables  ##############################################

plot_parameter_space_vertical_baseline = 0
plot_metallicity_binary_pop = 0
plot_histograms_binary_pop = 0
plot_single_parameter_space = 0
plot_tsne_example = 0
plot_tsne_parameters_example = 0
plot_dbscan_intermediate = 0
plot_bars_parameter_space = 0
plot_binaryvsbinary_params = 0

################################################################################
# Plot the parmaeter space for the baseline model ##############################
# Plot parameter space in a vertical figure

if plot_parameter_space_vertical_baseline == 1:
    mpl.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(6, 3, figsize=[33, 55], tight_layout=False)
    i = 0
    for x in range(6):
        for y in range(3):
            SELECTED_RANGES = [(float(BOTTOM[i]), float(TOP[i]))]
            print(x, y, SELECTED_RANGES)

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

            # x = min pts, y = eps
            X, Y = np.meshgrid(np.linspace(dbscan.min_eps,
                                        dbscan.max_eps,
                                        dbscan.iterations),
                               np.linspace(dbscan.min_minpts,
                                        dbscan.max_minpts,
                                        dbscan.iterations))

            Z = (dbscan.parameter_space['recovery_ratio'].values).reshape(X.shape)
            mask_recovery = dbscan.parameter_space['recovery_ratio'] == np.max(
            dbscan.parameter_space['recovery_ratio'])

            textstr = r'{} - {} nm'.format(BOTTOM[i], TOP[i])
            contour = ax[x, y].contourf(X, Y, Z.T, 85, cmap='plasma', vmin=0, vmax=0.8)
            ax[x, y].autoscale(False)
            ax[x, y].scatter(dbscan.optimized_eps, dbscan.optimized_minpts,
             s=600, linewidths=8, marker='x', c='k', alpha=0.85, zorder=1)
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
            # place a text box in upper left in axes coords
            ax[x, y].text(0.7, 0.2, textstr, transform=ax[x, y].transAxes,
                                 fontsize=20, verticalalignment='top', bbox=props)

            ax[x, y].set_xlabel(r'$\epsilon$', fontsize=30)
            ax[x, y].set_ylabel(r'$minPts$', fontsize=30)
            i += 1
    plt.subplots_adjust(hspace=0.35)
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
     ax=ax.ravel().tolist(), orientation='horizontal', pad=0.05)
    cbar.ax.set_xlabel(r'Recovery',  labelpad=20)
    mpl.rcParams.update({'font.size': 30})
    plt.savefig('parameterspacebaselinemodel.png', dpi=200)

################################################################################
# Plot metallicity of the binary population ####################################

if plot_metallicity_binary_pop == 1:
    pop = stellar_parameters[mask_binaries]

    fig = plt.figure(figsize=[10, 8], tight_layout=True)
    ax_hist = fig.add_subplot(111, label='hist')
    ax_line = fig.add_subplot(111, label='line', frame_on=False)

    ax_hist.hist(pop['feh_A'], histtype='stepfilled',
     bins=20, color='grey', alpha=0.5, label=r'[Fe/H]$_{A}$')
    ax_hist.hist(pop['feh_B'], histtype='step',
     bins=20, color='k', label=r'[Fe/H]$_{B}$')
    ax_hist.set_xlabel('[Fe/H]')
    ax_hist.set_yticklabels([])
    ax_hist.legend(loc='upper left')

    ax_line.scatter(pop['feh_A'], pop['feh_B'], s=2, c='orange', alpha=0.35)
    ax_line.set_xlabel(r'[Fe/H]$_{A}$')
    ax_line.set_ylabel(r'[Fe/H]$_{B}$')
    ax_line.xaxis.tick_top()
    ax_line.yaxis.tick_right()
    ax_line.xaxis.set_label_position('top')
    ax_line.yaxis.set_label_position('right')

    niceplot(fig)

    plt.savefig('metallicity_binarypopulation.png', dpi=150)

################################################################################
# Plot histograms of the binary population #####################################

if plot_histograms_binary_pop == 1:
    pop = stellar_parameters[mask_binaries]
    fig, ax = plt.subplots(2, 2, figsize=[13, 18], tight_layout=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.suptitle(r'Binary Population from Eggleton', y=0.94)

    ax[0, 0].hist(pop['teff_A'], bins=20, color='grey', alpha=0.5,
     label=r'$T_{\mathrm{{eff}}, A}$', histtype='stepfilled')
    ax[0, 0].hist(pop['teff_B'], bins=20, color='k', alpha=0.75,
     label=r'$T_{\mathrm{{eff}}, B}$', histtype='step')
    ax[0, 0].set_xlabel(r'T$_{\mathrm{eff}} \,$ [K]')
    # ax[0, 0].set_xticks([5000, 5500, 6000, 6500, 7000])
    ax[0, 0].legend()

    ax[0, 1].hist(pop['logg_A'], bins=20, color='grey',alpha=0.5,
     label=r'$log \, g_{A}$', histtype='stepfilled')
    ax[0, 1].hist(pop['logg_B'], bins=20, color='k', alpha=0.75,
     label=r'$log \, g_{B}$', histtype='step')
    ax[0, 1].set_xlabel(r'$log \, g$')
    ax[0, 1].legend(loc='upper left')

    ax[1, 0].hist(pop['mass_A'], bins=20, color='grey', alpha=0.5,
     label=r'$M_{A}$', histtype='stepfilled')
    ax[1, 0].hist(pop['mass_B'], bins=20, color='k', alpha=0.75,
     label=r'$M_{B}$', histtype='step')
    ax[1, 0].set_xlabel(r'$M/M_{{\odot}}$')
    ax[1, 0].legend()

    ax[1, 1].hist(pop['mass ratio'], bins=20, color='grey', alpha=0.5,
     label=r'$q$', histtype='stepfilled')
    ax[1, 1].hist(pop['lum ratio'], bins=20, color='k', alpha=0.75,
     label=r'$L_{B} / L_{A}$', histtype='step')
    ax[1, 1].set_xlabel(r'$L_{B}/L_{A}$ and $q$')
    ax[1, 1].legend(loc='upper left')

    niceplot(fig)

    plt.savefig('binarypopulation2.png', dpi=150)

################################################################################
# Plot the parameter space for a single model ##################################

if plot_single_parameter_space == 1:
    SELECTED_RANGES = [(800.0, 825.0)]

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

    # x = min pts, y = eps
    X, Y = np.meshgrid(np.linspace(dbscan.min_eps,
                                dbscan.max_eps,
                                dbscan.iterations),
                       np.linspace(dbscan.min_minpts,
                                dbscan.max_minpts,
                                dbscan.iterations))

    Z = (dbscan.parameter_space['recovery_ratio'].values).reshape(X.shape)
    mask_recovery = dbscan.parameter_space['recovery_ratio'] == np.max(
    dbscan.parameter_space['recovery_ratio'])

    fig, ax = plt.subplots(figsize=[14, 10], tight_layout=True)
    contour = ax.contourf(X, Y, Z.T, 85, cmap='plasma', vmin=0, vmax=0.8)
    plt.autoscale(False)
    ax.scatter(dbscan.optimized_eps, dbscan.optimized_minpts, s=400,
                     marker='x', c='k', alpha=0.85, zorder=1)
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$minPts$')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.65)

    ax.text(0.6, 0.955, 'SNR {}, Perplexity {}'.format(100, 30),
     transform=ax.transAxes, fontsize=22, verticalalignment='top',
     bbox=props)

    ax.text(0.05, 0.955, r'{} - {} nm'.format(SELECTED_RANGES[0][0],
     int(SELECTED_RANGES[0][1])),
     transform=ax.transAxes, fontsize=22, verticalalignment='top',
     bbox=props)

    ax.text(0.05, 0.1,
     r'Recovery: {:.2f}'.format(np.max(Z)),
     transform=ax.transAxes, fontsize=22, verticalalignment='top',
     bbox=props)

    cbar = plt.colorbar(contour, ticks=[0, 0.1, 0.2, 0.3, 0.4,
     0.5, 0.6, 0.7, 0.8])
    cbar.ax.set_ylabel(r'Recovery',  labelpad=20, rotation=270)

    niceplot(fig, remove_firstandlast_tick=True)
    plt.savefig('parameterspace_example.png', dpi=200)

################################################################################
# Plot simple t-SNE for method #################################################

if plot_tsne_example == 1:
    SELECTED_RANGES = [(800.0, 825.0)]
    mpl.rcParams.update({'font.size': 22})
    tsne = np.load(('/Users/pablonavarrobarrachina/Desktop/Results FFTW/' +
            'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy').format(SELECTED_RANGES,
            PERPLEXITY, SNR))
    tsne_x = tsne[:, 0]
    tsne_y = tsne[:, 1]

    # Normalize the t-SNE data
    normalized_tsne_x = (2 * (tsne_x - np.min(tsne_x)) /
                              (np.max(tsne_x) - np.min(tsne_x)) - 1) * 10
    normalized_tsne_y = (2 * (tsne_y - np.min(tsne_y)) /
                              (np.max(tsne_y) - np.min(tsne_y)) - 1) * 10

    fig, ax = plt.subplots(figsize=[15, 13], tight_layout=True)
    ax.scatter(normalized_tsne_x[mask_singles],
                normalized_tsne_y[mask_singles],
                c='grey', s=2, label='Single', alpha=0.65)
    ax.scatter(normalized_tsne_x[mask_binaries],
                normalized_tsne_y[mask_binaries],
                c='red', s=2, label='Binary')

    legend1 = plt.legend(loc='lower right')
    legend1.legendHandles[0]._sizes = [30]
    legend1.legendHandles[1]._sizes = [30]
    plt.tight_layout()
    niceplot(fig)
    plt.savefig('tsneexample800825.png', dpi=200)

################################################################################
# Plot t-SNE with four stellar parameters ######################################

if plot_tsne_parameters_example == 1:
    SELECTED_RANGES = [(800.0, 825.0)]
    mpl.rcParams.update({'font.size': 20})
    tsne = np.load(('/Users/pablonavarrobarrachina/Desktop/Results FFTW/' +
            'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy').format(SELECTED_RANGES,
            PERPLEXITY, SNR))
    tsne_x = tsne[:, 0]
    tsne_y = tsne[:, 1]

    normalized_tsne_x = (2 * (tsne_x - np.min(tsne_x)) /
                              (np.max(tsne_x) - np.min(tsne_x)) - 1) * 10
    normalized_tsne_y = (2 * (tsne_y - np.min(tsne_y)) /
                              (np.max(tsne_y) - np.min(tsne_y)) - 1) * 10

    fig, ax = plt.subplots(2, 2, figsize=[30, 20], tight_layout=False)
    b = ax[0, 0].scatter(normalized_tsne_x[mask_singles], normalized_tsne_y[mask_singles],
                         c=stellar_parameters['teff_A'][mask_singles],
                         cmap='plasma',
                         s=2, label='Single')
    ax[0, 0].scatter(normalized_tsne_x[mask_binaries], normalized_tsne_y[mask_binaries],
                     c=stellar_parameters['teff_A'][mask_binaries],
                     cmap='plasma',
                     s=2, label='Single')
    cbar_2 = plt.colorbar(b, ax=ax[0, 0])
    cbar_2.ax.set_ylabel(r'T$_{eff} \, \, [K]$ primary',  labelpad=20,
                        fontsize=30)

    # ax[3].set_title('$log \, g \, \, [dex]$ primary')
    c = ax[0, 1].scatter(normalized_tsne_x[mask_singles],
                        normalized_tsne_y[mask_singles],
                        c=stellar_parameters['logg_A'][mask_singles],
                        cmap='plasma',
                        s=2, label='Single')
    ax[0, 1].scatter(normalized_tsne_x[mask_binaries],
                    normalized_tsne_y[mask_binaries],
                    c=stellar_parameters['logg_A'][mask_binaries],
                    cmap='plasma',
                    s=2, label='Single')
    cbar_3 = plt.colorbar(c, ax=ax[0, 1])
    cbar_3.ax.set_ylabel(r'$log \, g \, \, [dex]$ primary',  labelpad=20,
                        fontsize=30)


    ax[1, 0].scatter(normalized_tsne_x[mask_singles],
                    normalized_tsne_y[mask_singles],
                    cmap='plasma', c='grey', s=2, label='Single')
    a = ax[1, 0].scatter(normalized_tsne_x[mask_binaries],
                    normalized_tsne_y[mask_binaries],
                    c=stellar_parameters['lum ratio'][mask_binaries], s=2,
                    cmap='plasma', label='Single')
    cbar_1 = plt.colorbar(a, ax=ax[1, 0])
    cbar_1.ax.set_ylabel(r'$L_{B} \, / \, L_{A}$',  labelpad=20,
                        fontsize=30)


    # ax[1, 2].set_title('$|\Delta v _{rad}| \, \, [m/s] $')
    ax[1, 1].scatter(normalized_tsne_x[mask_singles],
                    normalized_tsne_y[mask_singles],
                    c='grey', s=2, label='Single')
    d = ax[1, 1].scatter(normalized_tsne_x[mask_binaries],
                    normalized_tsne_y[mask_binaries],
                    c=stellar_parameters['rad_vel'][mask_binaries],
                    cmap='plasma',
                    s=2, label='Single')
    cbar_4 = plt.colorbar(d, ax=ax[1, 1])
    cbar_4.ax.set_ylabel(r'$\Delta v _{rad} \, \, [km/s] $',  labelpad=5,
                        fontsize=30)

    niceplot(fig, labelsize=28)

    plt.savefig('tsneexample800825_parameters_2.png', dpi=150)

# ##############################################################################
# Plot bars for the parameter space results  ###################################

if plot_bars_parameter_space == 1:
    SNRS = [10, 25, 50, 100, 500]
    PERPLEXITIES = [5, 15, 30, 100]
    # Plot the recovery distribtion for each of the spectral regions
    for PERPLEXITY in PERPLEXITIES:
        tick_marker = []
        tick_labels = []
        plt.figure(figsize=[15, 7])
        for i in range(18):
            SELECTED_RANGES = ([(float(BOTTOM[i]), float(TOP[i]))])
            # print(PERPLEXITY, SELECTED_RANGES)
            mask = ((parameter_space_exploration['spectral_range'] ==  str(SELECTED_RANGES)) &
                   (parameter_space_exploration['perplexity'] == PERPLEXITY))

            # Create extra array to handle the plotting on the same point
            X = np.array([1,2,3,4,5]) + (i*6)
            ticks_positions = (X)
            recovery_values = parameter_space_exploration['recovery'][mask].values

            plt.bar(X, recovery_values, color=['#DADA94', '#9A9A68', '#666645',
                                                            '#5A5A3C', '#030302'])

            # Add the position/label to the list of ticks
            tick_marker.append(X[2])
            tick_labels.append(r'{} - {} nm'.format(BOTTOM[i], TOP[i]))

        plt.ylabel('Recovery')
        plt.yticks([0.2, 0.4, 0.6, 0.7])
        plt.xticks(tick_marker, tick_labels, rotation=35)

################################################################################
# Plot DBSCAN for the thesis method example  ###################################

if plot_dbscan_intermediate == 1:
    SELECTED_RANGES = [(800.0, 825.0)]

    tsne_data = np.load(('/Users/pablonavarrobarrachina/Desktop/Results FFTW/' +
            'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy').format(SELECTED_RANGES,
            PERPLEXITY, SNR))

    tsne_x = tsne_data[:, 0]
    tsne_y = tsne_data[:, 1]

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

    normalized_tsne_x = (2 * (tsne_x - np.min(tsne_x)) /
                              (np.max(tsne_x) - np.min(tsne_x)) - 1) * 10
    normalized_tsne_y = (2 * (tsne_y - np.min(tsne_y)) /
                              (np.max(tsne_y) - np.min(tsne_y)) - 1) * 10

    # Define the masks for the plots
    mask_noise = dbscan.labels_dbscan == -1
    mask_cluster = dbscan.labels_dbscan != -1
    mask_recovered_binaries = dbscan.ratio_labels == 1
    mask_non_recovered_binaries = dbscan.ratio_labels == 0
    mask_non_recovered_binaries_but_are_binaries = ((dbscan.ratio_labels == 0) &
                                      (stellar_parameters['binarity'] == 1))

    # Begin actual plotting of the DBSCAN results
    fig, ax = plt.subplots(1, 2, figsize=[22, 14], tight_layout=True)

    ax[0].scatter(normalized_tsne_x[mask_noise], normalized_tsne_y[mask_noise], s=1,
        c='grey', alpha=0.65)
    ax[0].scatter(normalized_tsne_x[mask_cluster], normalized_tsne_y[mask_cluster], s=1,
        c=dbscan.labels_dbscan[mask_cluster], cmap='plasma') #tab20
    ax[0].set_aspect('equal')

    ax[1].scatter(normalized_tsne_x[mask_recovered_binaries],
    normalized_tsne_y[mask_recovered_binaries],
     s=1, c='red', label='Recovered binary')
    ax[1].scatter(normalized_tsne_x[mask_non_recovered_binaries],
     normalized_tsne_y[mask_non_recovered_binaries],
     s=1, c='grey', alpha=0.65, label='Single star')
    ax[1].scatter(normalized_tsne_x[mask_non_recovered_binaries_but_are_binaries],
     normalized_tsne_y[mask_non_recovered_binaries_but_are_binaries],
     s=1, c='blue', label='Non-recovered binary')
    ax[1].set_aspect('equal')
    legend = ax[1].legend(loc='upper right')
    legend.legendHandles[0]._sizes = [50]
    legend.legendHandles[1]._sizes = [50]
    legend.legendHandles[2]._sizes = [50]

    niceplot(fig, labelsize=22, remove_firstandlast_tick=True)

    plt.savefig('dbscan_intermediatestep.png', dpi=150)

################################################################################
# Plot the binary parameters for comparison, add in the synthesis section  #####

if plot_binaryvsbinary_params == 1:
    binary_parameters = stellar_parameters[mask_binaries]

    fig, ax = plt.subplots(2, 2, figsize=[20, 14], tight_layout=True)
    ax[0, 0].scatter(binary_parameters['teff_A'],
     binary_parameters['teff_B'], c='k', alpha=0.35, s=10)
    ax[0, 0].set_xlabel(r'$T_{\mathrm{eff}, A} \, [K]$', fontsize=24)
    ax[0, 0].set_ylabel(r'$T_{\mathrm{eff}, B} \, [K]$', fontsize=24)

    ax[0, 1].scatter(binary_parameters['logg_A'],
     binary_parameters['logg_B'], c='k', alpha=0.35, s=10)
    ax[0, 1].set_xlabel(r'$log \, g_{A}$', fontsize=24)
    ax[0, 1].set_ylabel(r'$log \, g_{B}$', fontsize=24)

    ax[1, 0].scatter(binary_parameters['mass_A'],
     binary_parameters['mass_B'], c='k', alpha=0.35, s=10)
    ax[1, 0].set_xlabel(r'$M_{A} \, [M_{{\odot}}]$', fontsize=24)
    ax[1, 0].set_ylabel(r'$M_{B} \, [M_{{\odot}}]$', fontsize=24)

    ax[1, 1].scatter(scaling_relations.classic_MLR(binary_parameters['mass_A']),
     scaling_relations.classic_MLR(binary_parameters['mass_B']),
     c='k', alpha=0.35, s=10)
    ax[1, 1].set_xlabel(r'$L_{A} \, [L_{{\odot}}]$', fontsize=24)
    ax[1, 1].set_ylabel(r'$L_{B} \, [L_{{\odot}}]$', fontsize=24)

    niceplot(fig, labelsize=24, remove_firstandlast_tick=True)

    plt.savefig('binaryvsbinary_params.png', dpi = 200 )
