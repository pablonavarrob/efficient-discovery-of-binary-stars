import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
from dbscan_ratio import niceplot

matplotlib.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size': 16})

# Plotting variables
plotKiel_raw = 0
plotKiel_filtered = 0
plotKiel_DwarfsGiants = 1
plot_parameterDistribution = 0


#__________________________________ KIEL DIAGRAM GALAH DR2 ________________________________________#

# Open fits file
GALAH = fits.open(('/Users/pablonavarrobarrachina/Desktop/Master/'+
'Master-Project/Code/GALAH_data/GALAH_DR2.1_catalog.fits.txt'))

from analyseSpectralData_functions import logg_zwitter

# Import GALAH data
data = GALAH[1].data

#____________________________________________________________________________________________________

# Mask all of the data
# Quality cuts: only stars flagged with 0 are good for using in this synthesis
mask_flag = data['flag_cannon'] == 0 # Creates the mask
data_filter_flag = data[mask_flag]   # Masks all the raw data

# Extract information from the filtered GALAH data
teff = data_filter_flag['teff'] # Effective temperature in Kelvin
logg = data_filter_flag['logg'] # Surface gravity in dex
fe_h = data_filter_flag['fe_h'] # Iron abundance

print('There are {} in the filtered set'.format(len(data_filter_flag)))

# Create set of temperatures to plot the line that divides the dwarfs from the giants using the
# Zwitter prescription
tempArray = np.linspace(np.min(data_filter_flag['teff']), np.max(data_filter_flag['teff']), 5000)

# Separate the filtered array into dawrfs and giants
dwarfs = data_filter_flag[logg_zwitter(data_filter_flag['teff']) <= data_filter_flag['logg']]
giants = data_filter_flag[logg_zwitter(data_filter_flag['teff']) >= data_filter_flag['logg']]

#____________________________________________________________________________________________________

if plotKiel_DwarfsGiants == 1:
    params = {'legend.fontsize': 30}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(figsize=[14,16], tight_layout=True)
    ax.plot(tempArray, logg_zwitter(tempArray), c = 'k', lw=1.25, alpha=0.75)
#    p = plt.scatter(teff, logg, c=fe_h, s=0.1, marker='x', cmap='ocean')
#    plt.colorbar(p, label='[Fe/H]')
    ax.scatter(dwarfs['teff'], dwarfs['logg'], c='black', s=2, alpha=0.2)
    ax.scatter(giants['teff'], giants['logg'], c='grey', s=2, alpha=0.2)
#   plt.title('Filtered GALAH stars - {} dwarfs, {} giants'.format(len(dwarfs), len(giants)))
    # Set fake plots for the labels and the legend
    ax.scatter(dwarfs['teff'][0], dwarfs['logg'][0], c='black',  s=0.01, alpha=0.75, label='Dwarfs')
    ax.scatter(giants['teff'], giants['logg'], c='grey', s=0.1, alpha=0.75, label='Giants')

    ax.set_ylabel(r'$logg \, \, [dex]$', fontsize=40)
    ax.set_xlabel(r'$T_{\mathrm{eff}} \, \, [K]$', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.tick_params(axis='both', which='minor', labelsize=36)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    legend1 = plt.legend(loc='upper left', fontsize=34)
    legend1.legendHandles[0]._sizes = [80]
    legend1.legendHandles[1]._sizes = [80]
    plt.rcParams.update({'font.size': 40})

    # niceplot(fig, remove_firstandlast_tick=True, tight_layout=False)

    plt.savefig('kiel_filtereddwarfsgiants.png', dpi=150)

if plotKiel_filtered == 1:
    plt.figure(figsize=[10,10])
    p = plt.scatter(teff, logg, c=fe_h, s=0.1, marker='x', cmap='ocean', alpha=0.75)
    plt.colorbar(p, label='[Fe/H]')
    plt.title('Filtered GALAH stars - {} stars'.format(len(data_filter_flag)))
    plt.ylabel('$logg \, \, [dex]$')
    plt.xlabel('$T_{eff} \, \, [K]$')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
#    plt.savefig('GALAHFilteredStars.png', dpi=150)

# Plot the rwa Kiel-Diagram
if plotKiel_raw == 1:
    plt.figure(figsize=[10,10])
    p = plt.scatter(data['teff'], data['logg'], c=data['fe_h'], s=0.1, marker='x', cmap='ocean', alpha=0.75)
    plt.colorbar(p, label='[Fe/H]')
    plt.title('Unfiltered GALAH stars - {} stars'.format(len(data)))
    plt.ylabel('$logg \, \, [dex]$')
    plt.xlabel('$T_{eff} \, \, [K]$')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig('GALAHUnfilteredStars.png', dpi=150)

#____________________________________________________________________________________________________

if plot_parameterDistribution == 1:
    # Plot the parameters on top of each other to compare: there will be 2 rows and 5
    fig, ax = plt.subplots(1, 3, figsize=(15,7), constrained_layout=True)
    fig.suptitle('Distribution of paramteters of the GALAH filtered stars')
    ax[0].hist(data_filter_flag['teff'], bins=50, color='k', alpha=0.85)
    ax[0].set_xlabel('$t_{eff}$', fontsize='x-large')
    ax[1].hist(data_filter_flag['logg'], bins=50, color='k', alpha=0.85)
    ax[1].set_xlabel('$log \, g$', fontsize='large')
    ax[2].hist(data_filter_flag['fe_h'], bins=50, color='k', alpha=0.85)
    ax[2].set_xlabel('[Fe/H]')
    plt.subplots_adjust(wspace=0.35, hspace=0.15)
    plt.savefig('GALAHFiltered_distributionParameters.png', dpi=150)
