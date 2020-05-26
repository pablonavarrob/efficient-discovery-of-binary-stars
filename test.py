import numpy as np
import matplotlib.pyplot as plt
import scaling_relations
import pandas as pd
#%matplotlib qt
import pandas as pd
import galah
import matplotlib
from dbscan_ratio import niceplot

matplotlib.rcParams.update({'font.size': 20})
pop = pd.read_csv(('data/BinaryPopulation_5000_classicrelations' +
'_allStarsDifferent_DucheneKraus13_q_IMF_exponential0,3-----FINALPOPULATION.csv'))

fig, ax = plt.subplots(1, 4, figsize=[24, 10], tight_layout=False)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# plt.suptitle(r'Binary Population from Eggleton', y=0.94)

ax[0].hist(pop['teff_A'], bins=20, color='grey', alpha=0.5,
 label=r'$T_{\mathrm{eff}, A}$', histtype='stepfilled')
ax[0].hist(pop['teff_B'], bins=20, color='k', alpha=0.75,
 label=r'$T_{\mathrm{eff}, B}$', histtype='step')
ax[0].set_xlabel(r'T$_{\mathrm{eff}} \, [K]$')
ax[0].set_xticks([5000, 6000, 7000])
ax[0].legend()

ax[1].hist(pop['logg_A'], bins=20, color='grey',alpha=0.5,
 label=r'$log \, g_{A}$', histtype='stepfilled')
ax[1].hist(pop['logg_B'], bins=20, color='k', alpha=0.75,
 label=r'$log \, g_{B}$', histtype='step')
ax[1].set_xlabel(r'$log \, g$')
ax[1].legend(loc='upper left')

ax[2].hist(pop['mass_A'], bins=20, color='grey', alpha=0.5,
 label=r'$M_{A}$', histtype='stepfilled')
ax[2].hist(pop['mass_B'], bins=20, color='k', alpha=0.75,
 label=r'$M_{A}$', histtype='step')
ax[2].set_xlabel(r'$M/M_{\odot}$')
ax[2].legend()

ax[3].hist(pop['mass ratio'], bins=20, color='grey', alpha=0.5,
 label=r'$q$', histtype='stepfilled')
ax[3].hist(pop['lum ratio'], bins=20, color='k', alpha=0.75,
 label=r'$L_{B} / L_{A}$', histtype='step')
ax[3].set_xlabel(r'$L_{B}/L_{A}$ and $q$')
ax[3].legend(loc='upper left')
plt.savefig('distributions_binarypop.png', dpi=150)

# fig = plt.figure(figsize=[10, 8], tight_layout=True)
# ax_hist = fig.add_subplot(111, label='hist')
# ax_line = fig.add_subplot(111, label='line', frame_on=False)
#
# ax_hist.hist(pop['feh_A'], histtype='stepfilled',
#  bins=20, color='grey', alpha=0.5, label=r'[Fe/H]$_{A}$')
# ax_hist.hist(pop['feh_B'], histtype='step',
#  bins=20, color='k', label=r'[Fe/H]$_{B}$')
# ax_hist.set_xlabel('[Fe/H]')
# ax_hist.set_yticklabels([])
# ax_hist.legend(loc='upper left')
#
# ax_line.scatter(pop['feh_A'], pop['feh_B'], s=2, c='orange', alpha=0.35)
# ax_line.set_xlabel(r'[Fe/H]$_{A}$')
# ax_line.set_ylabel(r'[Fe/H]$_{B}$')
# ax_line.xaxis.tick_top()
# ax_line.yaxis.tick_right()
# ax_line.xaxis.set_label_position('top')
# ax_line.yaxis.set_label_position('right')
# plt.savefig('metallicity_binarypopulation.png', dpi=150)


# galah_data = galah.GALAH_survey()
# galah_data.get_stars_run()
# stars_run = galah_data.stars_run
#
# plt.hist(stars_run['teff'], bins=100, alpha=0.5)
# plt.hist(pop['teff_B'], bins=100, alpha=0.8)
#
# plt.figure(figsize=[15,15])
# plt.hist(v_rad_difference, bins=100, color='orange', alpha=0.75)
# plt.xlabel(r"$\Delta v_{rad} \, [km/s]$")
# plt.savefig(dirs.data + 'histogram_radvel.png', dpi=350)


from sklearn.cluster import DBSCAN
import directories
import dbscan_ratio

dirs = directories.directories()

PERPLEXITY = 30
SNR = 100
SELECTED_RANGES =[(550.0, 575.0)]

tsne_data = np.load(('/Users/pablonavarrobarrachina/Desktop/Results FFTW/' +
    'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy').format(SELECTED_RANGES,
    PERPLEXITY, SNR))

tsne_x = tsne_data[:,0]
tsne_y = tsne_data[:,1]

stellar_parameters = pd.read_csv(dirs.data +
        'stellar_parameters_duchenekrauspopulation.csv')

normalized_tsne_x = (2 * (tsne_x - np.min(tsne_x)) /
    (np.max(tsne_x) - np.min(tsne_x)) - 1) * 10
normalized_tsne_y = (2 * (tsne_y - np.min(tsne_y)) /
    (np.max(tsne_y) - np.min(tsne_y)) - 1) * 10
tSNEData_ = pd.DataFrame({'x': normalized_tsne_x, 'y': normalized_tsne_y})

naive_dbscan_labels = DBSCAN(eps=0.244444, min_samples=102.777778).fit(tSNEData_).labels_

dbscan = dbscan_ratio.dbscan_method(tsne_data[:, 0], tsne_data[:, 1],
                                stellar_parameters['binarity'],
                                stellar_parameters, SELECTED_RANGES,
                                SNR, PERPLEXITY)

ratiolabels = dbscan.find_binaries_ratio_dbscan(naive_dbscan_labels)

plt.figure(figsize = [20,20])
plt.scatter(normalized_tsne_x, normalized_tsne_y, s=1, alpha=0.45, c=ratiolabels)

plt.figure(figsize = [20,20])
plt.scatter(normalized_tsne_x, normalized_tsne_y, s=1, alpha=0.45, c=stellar_parameters['binarity'])
