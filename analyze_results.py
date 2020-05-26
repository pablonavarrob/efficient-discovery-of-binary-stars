import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
import directories
import dbscan_ratio
from dbscan_ratio import niceplot

# Manage directories ###########################################################
dirs = directories.directories()

################################################################################
# Import the data ##############################################################
stellar_parameters = pd.read_csv(dirs.data +
        'stellar_parameters_duchenekrauspopulation.csv')
mask_singles = stellar_parameters['binarity'] == 0
mask_binaries = stellar_parameters['binarity'] == 1

################################################################################
# Triggers and such  ###########################################################

plot_bar_meanrecovery_per_perplexity = 1
plot_bar_meanrecovery_per_snr = 1
plot_tsne_maps = 0
plot_vertical_tsne = 0
plot_comparison_tsne_perplexity = 0
plot_comparison_tsne_noise = 0
plot_histograms_stellar_parameters = 0
make_table_variance = 0
plot_table_variance = 0
study_binaries = 0
plot_binaries_independent = 0
apendix_table_bonafidebinaries = 0

################################################################################
# Import the parameter space exploration #######################################
parameter_space_exploration = pd.read_csv(dirs.data +
    'parameter_space_exploration_run_25mm_intervals.csv')

TOP = np.arange(475, 925, 25)
BOTTOM = np.arange(450, 900, 25)
PERPLEXITIES = [5, 15, 30, 100]
SNRS = [10, 25, 50, 100, 500]

################################################################################
# Plot mean recovery per perplexity value  #####################################
# Calculate mean of the recovery per perplexity

if plot_bar_meanrecovery_per_perplexity == 1:
    plot_data = []
    for PERPLEXITY in PERPLEXITIES:
        for i in range(len(TOP)):
            SELECTED_RANGES = ([(float(BOTTOM[i]), float(TOP[i]))])
            mask_fixedperp_fixedrange = ((parameter_space_exploration['spectral_range'] ==
                                                                  str(SELECTED_RANGES)) &
                                (parameter_space_exploration['perplexity'] == PERPLEXITY))

            recovery_values_fixed_perplexity = parameter_space_exploration['recovery'][mask_fixedperp_fixedrange]
            mean_recovery_pp = np.mean(recovery_values_fixed_perplexity)

            plot_data.append({'range' : r'{} - {} nm'.format(BOTTOM[i], TOP[i]),
              'mean_recovery' : mean_recovery_pp,
              'min_recovery' : mean_recovery_pp - min(recovery_values_fixed_perplexity),
              'max_recovery' : max(recovery_values_fixed_perplexity) - mean_recovery_pp,
              'true_min': min(recovery_values_fixed_perplexity),
              'true_max' : max(recovery_values_fixed_perplexity),
              'perplexity' : PERPLEXITY,
              'error' : np.std(recovery_values_fixed_perplexity)})


    plot_data = pd.DataFrame(plot_data)
    plot_data['index'] = plot_data.index
    plot_data = plot_data.sort_values(['range', 'index'], ascending=[True, True])

    # Compute the positions of the bars and the tick labels
    data_position = []
    tick_labels   = []
    for i in range(len(TOP)):
        data_position.append(np.arange(1, 5, 1) + i*6)
        tick_labels.append(2.5 + i*6)
    data_position = np.hstack(np.asarray(data_position))
    yticks = np.round(np.arange(0, 1, 0.1), 1)
    # Calculate errors relative to mean value
    errors = np.array((plot_data['min_recovery'].values, plot_data['max_recovery'].values))
    mid_point = int(len(plot_data)/2)

    # Define color test
    plasma_colors = matplotlib.cm.plasma.colors
    # Begin actual plotting
    fig, ax = plt.subplots(2, 1, figsize=[20, 25])
    i = 0
    for PERPLEXITY in PERPLEXITIES:
        mask_p_low = (plot_data['perplexity'] == PERPLEXITY)[:mid_point]
        mask_p_high = (plot_data['perplexity'] == PERPLEXITY)[mid_point:]

        ax[0].bar((data_position[:mid_point])[mask_p_low],
            (plot_data['mean_recovery'][:mid_point])[mask_p_low],
            yerr = errors[:, :mid_point][:,mask_p_low], capsize=3,
            alpha=0.65, edgecolor='k', linewidth=1.5, color=plasma_colors[i],
            label=PERPLEXITY)

        ax[1].bar((data_position[mid_point:])[mask_p_high],
            (plot_data['mean_recovery'][mid_point:])[mask_p_high],
            yerr = errors[:, mid_point:][:,mask_p_high], capsize=3,
            alpha=0.65, edgecolor='k', linewidth=1.5, color=plasma_colors[i],
            label=PERPLEXITY)

        i += 65

    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, title='Perplexity',
    loc='center right', edgecolor='inherit', fontsize=16)
    legend.get_title().set_fontsize(16)
    legend.legendHandles[0]._sizes = [65]
    legend.legendHandles[1]._sizes = [65]
    legend.legendHandles[2]._sizes = [65]
    plt.subplots_adjust(right=0.92 )
    # plt.yscale('log')
    ax[0].set_xticks(tick_labels[:9])
    ax[0].set_xticklabels(np.unique(plot_data['range'])[:9], fontsize=17)
    ax[1].set_xticks(tick_labels[9:])
    ax[1].set_xticklabels(np.unique(plot_data['range'])[9:], fontsize=17)
    ax[0].set_yticklabels(yticks, fontsize=17)
    ax[1].set_yticklabels(yticks, fontsize=17)
    matplotlib.rcParams.update({'font.size': 17})

    plt.savefig('barplot_allperplexities_meanrecovery.png', dpi=200)

################################################################################
# Plot mean recovery per SNR value  #####################################
# Calculate mean of the recovery per SNR

if plot_bar_meanrecovery_per_snr == 1:
    matplotlib.rcParams.update({'font.size': 16})
    plot_data = []
    for SNR in SNRS:
        for i in range(len(TOP)):
            SELECTED_RANGES = ([(float(BOTTOM[i]), float(TOP[i]))])
            mask_fixedsnr_fixedrange = ((parameter_space_exploration['spectral_range'] ==
                                                                  str(SELECTED_RANGES)) &
                                (parameter_space_exploration['snr'] == SNR))

            recovery_values_fixed_snr = parameter_space_exploration['recovery'][mask_fixedsnr_fixedrange]
            mean_recovery_psnr = np.mean(recovery_values_fixed_snr)

            plot_data.append({'range' : r'{} - {} nm'.format(BOTTOM[i], TOP[i]),
              'mean_recovery' : mean_recovery_psnr,
              'min_recovery' : mean_recovery_psnr - min(recovery_values_fixed_snr),
              'max_recovery' : max(recovery_values_fixed_snr) - mean_recovery_psnr,
              # 'true_min': min(recovery_values_fixed_perplexity),
              # 'true_max' : max(recovery_values_fixed_perplexity),
              'snr' : SNR,
              'error' : np.std(recovery_values_fixed_snr)})

    plot_data = pd.DataFrame(plot_data)
    plot_data['index'] = plot_data.index
    plot_data = plot_data.sort_values(['range', 'index'], ascending=[True, True])

    # Compute the positions of the bars and the tick labels
    data_position = []
    tick_labels   = []
    for i in range(len(TOP)):
        data_position.append(np.arange(1, 6, 1) + i*7)
        tick_labels.append(3 + i*7)
    data_position = np.hstack(np.asarray(data_position))
    yticks = np.round(np.arange(0, 1, 0.1), 1)

    # Calculate errors relative to mean value
    errors = np.array((plot_data['min_recovery'].values,
                       plot_data['max_recovery'].values))
    mid_point = int(len(plot_data)/2)
    plasma_colors = matplotlib.cm.plasma.colors

    #  Begin plotting: separate in two groups of 9 spectral ranges (45 data points)
    i = 0
    fig, ax = plt.subplots(2, 1, figsize=[20, 25])
    for SNR in SNRS:
        mask_snr_low = (plot_data['snr'] == SNR)[:mid_point]
        mask_snr_high = (plot_data['snr'] == SNR)[mid_point:]

        ax[0].bar((data_position[:mid_point])[mask_snr_low],
            (plot_data['mean_recovery'][:mid_point])[mask_snr_low],
            yerr = errors[:, :mid_point][:,mask_snr_low], capsize=3,
            alpha=0.65, edgecolor='k', linewidth=1.5, color=plasma_colors[i],
            label=SNR)

        ax[1].bar((data_position[mid_point:])[mask_snr_high],
            (plot_data['mean_recovery'][mid_point:])[mask_snr_high],
            yerr = errors[:, mid_point:][:,mask_snr_high], capsize=3,
            alpha=0.65, edgecolor='k', linewidth=1.5, color=plasma_colors[i],
            label=SNR)
        i += 50

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='SNR',
     loc='center right', edgecolor='inherit')
    plt.subplots_adjust(right=0.92)
    # plt.yscale('log')
    ax[0].set_xticks(tick_labels[:9])
    ax[0].set_xticklabels(np.unique(plot_data['range'])[:9], fontsize=17)
    ax[1].set_xticks(tick_labels[9:])
    ax[1].set_xticklabels(np.unique(plot_data['range'])[9:], fontsize=17)
    ax[0].set_yticklabels(yticks, fontsize=17)
    ax[1].set_yticklabels(yticks, fontsize=17)
    plt.savefig('barplot_allsnr_meanrecovery.png', dpi=200)

################################################################################
# Plot t-SNE maps  #############################################################

if plot_tsne_maps == 1:
    SELECTED_RANGES = [(500.0, 525.0)]
    PERPLEXITY = 30
    SNR = 100
    matplotlib.rcParams.update({'font.size': 22})
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
    # dbscan.plot_tsne_maps_stellar_parameters()
    dbscan.plot_tsne_maps_raw()
    #dbscan.plot_histograms()
    # dbscan.plot_parameter_space()

################################################################################
# Plot vertical t-SNE for baseline model  ######################################

if plot_vertical_tsne == 1:
    # Define for the baseline model
    PERPLEXITY = 30
    SNR = 100
    matplotlib.rcParams.update({'font.size': 26})
    # Create figure and begin plotting
    fig, ax = plt.subplots(6, 3, figsize=[35, 55], sharex=True, sharey=True)
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

            mask_singles = stellar_parameters['binarity'] == 0
            mask_recovered_binaries = ((dbscan.ratio_labels == 1) &
                                          (stellar_parameters['binarity'] == 1))
            mask_non_recovered_binaries = ((dbscan.ratio_labels == 0) &
                                          (stellar_parameters['binarity'] == 1))

            recovered_binaries = len(dbscan.normalized_tsne_x[mask_recovered_binaries])

            # Begin actual plotting
            ax[x, y].scatter(dbscan.normalized_tsne_x[mask_singles],
             dbscan.normalized_tsne_y[mask_singles], s=2, color='grey', label='Single star')
            ax[x, y].scatter(dbscan.normalized_tsne_x[mask_recovered_binaries],
             dbscan.normalized_tsne_y[mask_recovered_binaries], s=2, color='red',
             label='Recovered binary')
            ax[x, y].scatter(dbscan.normalized_tsne_x[mask_non_recovered_binaries],
             dbscan.normalized_tsne_y[mask_non_recovered_binaries], s=2, color='blue',
             label='Non-recovered binary')

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.35)
            # place a text box in upper left in axes coords
            ax[x, y].text(0.075, 0.1, 'Recovery: {:.2f}'.format(recovered_binaries/5000),
             transform=ax[x, y].transAxes, fontsize=26, verticalalignment='top',
             bbox=props)

            ax[x, y].text(0.075, 0.95, r'{} - {} nm'.format(BOTTOM[i], TOP[i]),
             transform=ax[x, y].transAxes, fontsize=26, verticalalignment='top',
             bbox=props)
            i += 1

    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', edgecolor='inherit')
    legend.legendHandles[0]._sizes = [100]
    legend.legendHandles[1]._sizes = [100]
    legend.legendHandles[2]._sizes = [100]

    plt.subplots_adjust(bottom=0.045)
    niceplot(fig, labelsize = 26, tight_layout=False)
    plt.savefig('tsne_vertical_baseline.png', dpi=100)

################################################################################
# Plot t-SNE to compare spectral ranges  #######################################

if plot_comparison_tsne_perplexity == 1:
    # Define for the baseline model
    TOPS = [500.0, 750.0]
    BOTTOMS = [475.0, 725.0]
    matplotlib.rcParams.update({'font.size': 22})
    SNR = 50
    # Create figure and begin plotting
    fig, ax = plt.subplots(4, 2, figsize=[20, 35])
    for i in range(2):
        j = 0
        for PERPLEXITY in PERPLEXITIES:
            SELECTED_RANGES = ([(float(BOTTOMS[i]), float(TOPS[i]))])
            print(j, i, PERPLEXITY, SELECTED_RANGES)

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


            mask_singles = stellar_parameters['binarity'] == 0
            mask_recovered_binaries = ((dbscan.ratio_labels == 1) &
                                          (stellar_parameters['binarity'] == 1))
            mask_non_recovered_binaries = ((dbscan.ratio_labels == 0) &
                                          (stellar_parameters['binarity'] == 1))
            mask_params = ((parameter_space_exploration['spectral_range'] == str(SELECTED_RANGES)) &
                (parameter_space_exploration['snr'] == SNR) &
                (parameter_space_exploration['perplexity'] == PERPLEXITY))

            # Begin actual plotting
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_singles],
             dbscan.normalized_tsne_y[mask_singles], s=2, color='grey', label='Single star')
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_recovered_binaries],
             dbscan.normalized_tsne_y[mask_recovered_binaries], s=2, color='red',
             label='Recovered binary')
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_non_recovered_binaries],
             dbscan.normalized_tsne_y[mask_non_recovered_binaries], s=2, color='blue',
             label='Non-recovered binary')

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.35)
            # place a text box in upper left in axes coords
            ax[j, i].text(0.65, 0.95, 'Perplexity {}'.format(PERPLEXITY),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            ax[j, i].text(0.075, 0.95, r'{} - {} nm'.format(int(BOTTOMS[i]), int(TOPS[i])),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            ax[j, i].text(0.075, 0.1,
             r'Recovery: {:.2f}'.format(parameter_space_exploration[mask_params]['recovery'].values[0]),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            j += 1

    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', edgecolor='inherit')
    legend.legendHandles[0]._sizes = [100]
    legend.legendHandles[1]._sizes = [100]
    legend.legendHandles[2]._sizes = [100]

    plt.subplots_adjust(bottom=0.0675)
    niceplot(fig, labelsize = 22, tight_layout=False)
    plt.savefig('tsne_comparison_perplexityeffect.png', dpi=150)

################################################################################
# Plot t-SNE to compare spectral ranges  #######################################

if plot_comparison_tsne_noise == 1:
    # Define for the baseline model
    matplotlib.rcParams.update({'font.size': 22})
    PERPLEXITY = 30
    TOPS = [500.0, 750.0]
    BOTTOMS = [475.0, 725.0]
    # Create figure and begin plotting
    fig, ax = plt.subplots(5, 2, figsize=[18, 35])
    for i in range(2):
        j = 0
        for SNR in SNRS:
            SELECTED_RANGES = ([(float(BOTTOMS[i]), float(TOPS[i]))])
            print(j, i, SNR, SELECTED_RANGES)

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

            mask_singles = stellar_parameters['binarity'] == 0
            mask_recovered_binaries = ((dbscan.ratio_labels == 1) &
                                          (stellar_parameters['binarity'] == 1))
            mask_non_recovered_binaries = ((dbscan.ratio_labels == 0) &
                                          (stellar_parameters['binarity'] == 1))

            mask_params = ((parameter_space_exploration['spectral_range'] == str(SELECTED_RANGES)) &
                (parameter_space_exploration['snr'] == SNR) &
                (parameter_space_exploration['perplexity'] == PERPLEXITY))

            # Begin actual plotting
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_singles],
             dbscan.normalized_tsne_y[mask_singles], s=2, color='grey', label='Single star')
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_recovered_binaries],
             dbscan.normalized_tsne_y[mask_recovered_binaries], s=2, color='red',
             label='Recovered binary')
            ax[j, i].scatter(dbscan.normalized_tsne_x[mask_non_recovered_binaries],
             dbscan.normalized_tsne_y[mask_non_recovered_binaries], s=2, color='blue',
             label='Non-recovered binary')

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.35)
            # place a text box in upper left in axes coords
            ax[j, i].text(0.75, 0.95, 'SNR {}'.format(SNR),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            ax[j, i].text(0.075, 0.95, r'{} - {} nm'.format(int(BOTTOMS[i]), int(TOPS[i])),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            ax[j, i].text(0.075, 0.125,
             r'Recovery: {:.2f}'.format(parameter_space_exploration[mask_params]['recovery'].values[0]),
             transform=ax[j, i].transAxes, fontsize=22, verticalalignment='top',
             bbox=props)

            j += 1

    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', edgecolor='inherit')
    legend.legendHandles[0]._sizes = [100]
    legend.legendHandles[1]._sizes = [100]
    legend.legendHandles[2]._sizes = [100]

    plt.subplots_adjust(bottom=0.0675)
    niceplot(fig, labelsize = 23, tight_layout=False)
    plt.savefig('tsne_comparison_noiseeffect.png', dpi=150)

################################################################################
# Plot the histograms of the best/worst  #######################################

if plot_histograms_stellar_parameters == 1:
    SELECTED_RANGES = [(500.0, 525.0)]
    PERPLEXITY = 15
    SNR = 50
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
    dbscan.plot_histograms()

################################################################################
# Create table of variance between best and worst result   #####################

if make_table_variance == 1:
    # We create here a data-frame that we will import as latex code

    table_data = []
    for i in range(len(TOP)):
        SELECTED_RANGES = str(([(float(BOTTOM[i]), float(TOP[i]))]))
        mask_range = parameter_space_exploration['spectral_range'] == SELECTED_RANGES
        min_rec = np.min(parameter_space_exploration[mask_range]['recovery'])
        max_rec = np.max(parameter_space_exploration[mask_range]['recovery'])

        # Dictionary to append and later create a data-frame
        data_range = {'Spectral Range' : '{} - {} nm'.format(BOTTOM[i] ,TOP[i]),
        'Recovery max.' : max_rec,
        'Recovery min.' : min_rec,
        'Max. difference' : np.round(max_rec - min_rec, 3),
        'Deviation' : np.round(np.std(parameter_space_exploration[mask_range]['recovery']), 3),
        'Mean recovery' : np.round(np.mean(parameter_space_exploration[mask_range]['recovery']), 3)}
        table_data.append(data_range)

    table = pd.DataFrame(table_data)

if (make_table_variance == 1 and plot_table_variance == 1):
# Plot the result from the table in several nice plots to see the flows
# from the variations..?
    yerr = np.array((table['Mean recovery'].values - table['Recovery min.'].values,
                     table['Recovery max.'].values - table['Mean recovery'].values))
    positions = np.arange(1, len(TOP) + 1, 1)*5

    tick_labels = ['{} - {} nm'.format(BOTTOM[i], TOP[i]) for i in range(len(TOP))]

    fig, ax = plt.subplots(figsize=[20, 10], tight_layout=True)
    positions = np.arange(0, len(TOP), 1)
    ax.plot(positions, table['Recovery max.'], lw=1.5, c='k', alpha=0.85)
    ax.plot(positions, table['Recovery min.'], lw=1.5, c='k', alpha=0.85)
    ax.plot(positions, table['Mean recovery'], lw=1.5, c='red')
    ax.set_ylabel('Fraction Recovered Binaries', fontsize=22)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=35)
    ax.set_ylim(0, 0.85)
    niceplot(fig, labelsize=22, remove_firstandlast_tick=False)
    plt.savefig('statistics_all.png', dpi=200)

################################################################################
# Study what happens to each binary each time accross all params ###############

if study_binaries == 1:

    # Get the binary parameters
    binary_parameters = stellar_parameters[mask_binaries].reset_index()
    del binary_parameters['index']

    # Need to put 360 new columns in the binary parameter dataframe, as
    # a label result from each analysis
    labels_exploration_singles = []
    labels_exploration_binaries = []
    for i in range(len(parameter_space_exploration)):
        print(i)
        results_single_exploration = [int(label) for
         label in parameter_space_exploration['ratio_labels'][i].replace('[', '').replace(']', '').split(', ')]
        labels_exploration_singles.append(np.array(results_single_exploration)[mask_singles])
        labels_exploration_binaries.append(np.array(results_single_exploration)[mask_binaries])

    labels_exploration_singles = pd.DataFrame(labels_exploration_singles).T
    labels_exploration_binaries = pd.DataFrame(labels_exploration_binaries).T

    total_numbers = labels_exploration_binaries.sum(axis=1)
    binary_exploration = (pd.concat((total_numbers,
                                    binary_parameters),
                                    axis=1)).rename(columns={0:'total_recoveries'})

    binary_exploration.to_csv('binary_exploration.csv', index=False)

if plot_binaries_independent == 1:

    binary_exploration = pd.read_csv('binary_exploration.csv')
    recoveries = binary_exploration['total_recoveries']
    percentile90 = int(360*0.9)
    percentile75 = int(360*0.75)
    percentile50 = int(360*0.5)
    percentile5 = int(360*0.05)

    mask_percentile90 = (recoveries >= percentile90)
    mask_percentile75 = ((percentile90 > recoveries) &
                        (recoveries >= percentile75))
    mask_percentile50 = ((percentile75 > recoveries) &
                        (recoveries >= percentile50))
    mask_percentile_rest = ((percentile50 > recoveries) &
                        (recoveries >= percentile5))
    mask_percentile_lower5 = (recoveries < percentile5)

    print(len(recoveries[mask_percentile90]),
     len(recoveries[mask_percentile75]),
     len(recoveries[mask_percentile50]),
     len(recoveries[mask_percentile_rest]),
     len(recoveries[mask_percentile_lower5]))

    plasma_colors = matplotlib.cm.plasma.colors
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)

    fig, ax = plt.subplots(3, 2, figsize=[22, 28])
    ax[0, 0].scatter(binary_exploration['teff_A'][mask_percentile90],
                    abs(binary_exploration['teff_B'])[mask_percentile90],
                    c = plasma_colors[0], s=15, edgecolor='none', zorder=5,
                    alpha=0.65, label=r'$> 90\%$ recovery')
    ax[0, 0].scatter(binary_exploration['teff_A'][mask_percentile75],
                    abs(binary_exploration['teff_B'])[mask_percentile75],
                    c = plasma_colors[63],  s=15, edgecolor='none', zorder=4,
                    alpha=0.65, label=r'$90 - 75\%$ recovery')
    ax[0, 0].scatter(binary_exploration['teff_A'][mask_percentile50],
                    abs(binary_exploration['teff_B'])[mask_percentile50],
                    c = plasma_colors[150],  s=15, edgecolor='none', zorder=3,
                    alpha=0.55, label=r'$75 - 50\%$ recovery')
    ax[0, 0].scatter(binary_exploration['teff_A'][mask_percentile_rest],
                    abs(binary_exploration['teff_B'])[mask_percentile_rest],
                    c = plasma_colors[225],  s=15, edgecolor='none', zorder=2,
                    alpha=0.65, label=r'$50 - 5 \%$ recovery')
    ax[0, 0].scatter(binary_exploration['teff_A'][mask_percentile_lower5],
                    abs(binary_exploration['teff_B'])[mask_percentile_lower5],
                    c = 'k', alpha=0.5,  s=15, edgecolor='none', zorder=1,
                    label=r'$< 5 \%$ recovery')
    ax[0, 0].set_xlabel(r'T$_{eff} \, \, [K] $ (A)', fontsize=20)
    ax[0, 0].set_ylabel(r'T$_{eff} \, \, [K] $ (B)', fontsize=20)
    ax[0, 0].text(0.05, 0.925, 'a)', transform=ax[0, 0].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    ax[1, 0].scatter(binary_exploration['logg_A'][mask_percentile90],
                    abs(binary_exploration['logg_B'])[mask_percentile90],
                    c = plasma_colors[0], s=15, edgecolor='none', zorder=5,
                    alpha=0.65)
    ax[1, 0].scatter(binary_exploration['logg_A'][mask_percentile75],
                    abs(binary_exploration['logg_B'])[mask_percentile75],
                    c = plasma_colors[63],  s=15, edgecolor='none', zorder=4,
                    alpha=0.65)
    ax[1, 0].scatter(binary_exploration['logg_A'][mask_percentile50],
                    abs(binary_exploration['logg_B'])[mask_percentile50],
                    c = plasma_colors[150],  s=15, edgecolor='none', zorder=3,
                    alpha=0.55)
    ax[1, 0].scatter(binary_exploration['logg_A'][mask_percentile_rest],
                    abs(binary_exploration['logg_B'])[mask_percentile_rest],
                    c = plasma_colors[225],  s=15, edgecolor='none', zorder=2,
                    alpha=0.65)
    ax[1, 0].scatter(binary_exploration['logg_A'][mask_percentile_lower5],
                    abs(binary_exploration['logg_B'])[mask_percentile_lower5],
                    c = 'k', alpha=0.5,  s=15, edgecolor='none', zorder=1)
    ax[1, 0].set_xlabel(r'$log \, g \, \, [dex]$ (A)', fontsize=20)
    ax[1, 0].set_ylabel(r'$log \, g \, \, [dex]$ (B)', fontsize=20)
    ax[1, 0].text(0.05, 0.925, 'c)', transform=ax[1, 0].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    ax[2, 0].scatter(binary_exploration['lum ratio'][mask_percentile90],
                    abs(binary_exploration['rad_vel'])[mask_percentile90],
                    c = plasma_colors[0], s=15, edgecolor='none', zorder=5,
                    alpha=0.65)
    ax[2, 0].scatter(binary_exploration['lum ratio'][mask_percentile75],
                    abs(binary_exploration['rad_vel'])[mask_percentile75],
                    c = plasma_colors[63],  s=15, edgecolor='none', zorder=4,
                    alpha=0.65)
    ax[2, 0].scatter(binary_exploration['lum ratio'][mask_percentile50],
                    abs(binary_exploration['rad_vel'])[mask_percentile50],
                    c = plasma_colors[150],  s=15, edgecolor='none', zorder=3,
                    alpha=0.55)
    ax[2, 0].scatter(binary_exploration['lum ratio'][mask_percentile_rest],
                    abs(binary_exploration['rad_vel'])[mask_percentile_rest],
                    c = plasma_colors[225],  s=15, edgecolor='none', zorder=2,
                    alpha=0.65)
    ax[2, 0].scatter(binary_exploration['lum ratio'][mask_percentile_lower5],
                    abs(binary_exploration['rad_vel'])[mask_percentile_lower5],
                    c = 'k', alpha=0.5,  s=15, edgecolor='none', zorder=1)
    ax[2, 0].set_xlabel(r'$L_B / L_A$', fontsize=20)
    ax[2, 0].set_ylabel(r'$| \Delta v _{rad} | \, \, [km/s] $', fontsize=20)
    ax[2, 0].text(0.05, 0.925, 'e)', transform=ax[2, 0].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    a = ax[0, 1].scatter(binary_exploration['mass ratio'],
                    binary_exploration['total_recoveries'],
                    c=(binary_exploration['feh_A']),
                    cmap='plasma', s=20, edgecolor='none',
                    alpha=0.5)    # plt.colorbar(plot)
    ax[0, 1].set_xlabel(r'$q$', fontsize=20)
    ax[0, 1].set_ylabel(r'Total Recoveries', fontsize=20)
    cbar_1 = plt.colorbar(a, ax=ax[0, 1])
    cbar_1.ax.set_ylabel(r'[Fe/H]',  labelpad=20,
                        fontsize=20)
    ax[0, 1].text(0.05, 0.925, 'b)', transform=ax[0, 1].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    b = ax[1, 1].scatter(binary_exploration['lum ratio'],
                    binary_exploration['total_recoveries'],
                    c=abs(binary_exploration['rad_vel']),
                    cmap='plasma', s=20, edgecolor='none',
                    alpha=0.5)    # plt.colorbar(plot)
    ax[1, 1].set_xlabel(r'$L_B / L_A$', fontsize=20)
    ax[1, 1].set_ylabel(r'Total Recoveries ', fontsize=20)
    cbar_2 = plt.colorbar(b, ax=ax[1, 1])
    cbar_2.ax.set_ylabel(r'$ | \Delta v _{rad} | \, \, [km/s]$',  labelpad=20,
                        fontsize=20)
    ax[1, 1].text(0.05, 0.925, 'd)', transform=ax[1, 1].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    c = ax[2, 1].scatter(abs(binary_exploration['rad_vel']),
                    binary_exploration['total_recoveries'],
                    c=binary_exploration['lum ratio'],
                    cmap='plasma', s=20, edgecolor='none',
                    alpha=0.5)    # plt.colorbar(plot)
    ax[2, 1].set_xlabel(r'$ | \Delta v _{rad} | \, \, [km/s] $', fontsize=20)
    ax[2, 1].set_ylabel(r'Total Recoveries', fontsize=20)
    cbar_3 = plt.colorbar(c, ax=ax[2, 1])
    cbar_3.ax.set_ylabel(r'$L_B / L_A$',  labelpad=20,
                        fontsize=20)
    ax[2, 1].text(0.9, 0.925, 'f)', transform=ax[2, 1].transAxes,
                         fontsize=20, verticalalignment='top', bbox=props)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    legend = ax[1, 0].legend(handles, labels,
                            edgecolor='inherit', fontsize=16)


    legend.legendHandles[0]._sizes = [65]
    legend.legendHandles[1]._sizes = [65]
    legend.legendHandles[2]._sizes = [65]
    legend.legendHandles[3]._sizes = [65]
    legend.legendHandles[4]._sizes = [65]

    plt.subplots_adjust(bottom=0.175)

    niceplot(fig, tight_layout=False)

    plt.savefig('binaries_examination.png', dpi=200)

################################################################################
# Prepare table of binaries that get recovered most of the times ###############

if apendix_table_bonafidebinaries == 1:

    binary_exploration = pd.read_csv('binary_exploration.csv')
    recoveries = binary_exploration['total_recoveries']
    percentile90 = np.percentile(recoveries, 90)
    mask_percentile90 = (recoveries >= percentile90)

    bonafide_binaries_total = ((binary_exploration[mask_percentile90]).sort_values('teff_A')).reset_index()
    indexes = np.linspace(0, len(bonafide_binaries_total)-1, 40, endpoint=True, dtype=int)

    # Define the range and cut it in equally sized steps based on temperature
    subsample_table = bonafide_binaries_total.iloc[indexes]
    # Round the columns of interest
    subsample_table = subsample_table.round({'teff_A' : 2,
                           'teff_B' : 2,
                           'feh_A' : 2,
                           'logg_A': 2,
                           'logg_B': 2,
                           'mass_A' : 2,
                           'mass_B' : 2,
                           'rad_vel' : 2,
                           'lum ratio' : 2,
                           'mass ratio' : 2})

    subsample_table = subsample_table.drop(columns=['comp_teff_A', 'comp_teff_B',
                                                    'binarity', 'feh_B'])
    subsample_table.to_csv('subsample_bonafidebinaries_appendix.csv', index=False)
