# Plot synthetic spectra to compare with a real one
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import sys
import os
import matplotlib
import directories
from analyseSpectralData_functions import add_noise
from dbscan_ratio import niceplot
#____________________________________________________________________________________________________
# plt.rcParams.update({'font.size': 18})

dirs = directories.directories()

plotSpectra = 0
plotGregorSpectra = 0
plotSynthVSGALAH = 0
plot_GALAHbinaries = 0
plot_binaries_presentation = 1
#____________________________________________________________________________________________________

spectraFilename = 'Galah_wholerange_dwarfs_170911004701386_5751.79_4.25_0.02.fits'
spectraWholeRange = fits.open(dirs.data + spectraFilename, memmap=False)[0].data

spectraCrossMatched = 'Galah_wholerange_dwarfs_150604002901348_5560.70_3.85_0.25.fits'
spectraCMFlux = (fits.open(dirs.data  + spectraCrossMatched, memmap=False)[0].data)

#____________________________________________________________________________________________________
resolution              = 2.8e4
sampling                = 0.004 #((wave_top - wave_base)/resolution)/5
wave_b = 450
wave_t = 900
waverange = (np.arange(wave_b, wave_t, sampling)*10) # In Armstrongs
runname                 = 'GregorPlotStars'
extraCharacterLength     = len(runname + '_')
synth_spectra_dir = dirs.data + 'pablo_spectra'

# Import the GALAH dataset
# Open fits file
GALAH_fit = fits.open(('/Users/pablonavarrobarrachina/Desktop/Master/'+
'Master-Project/Code/GALAH_data/GALAH_DR2.1_catalog.fits.txt'))
GALAH = GALAH_fit[1].data

#____________________________________________________________________________________________________
# Load Gregor's spectra from GALAH
GregorSpectraFlux = [] # Flux values
GregorSpectraID = []    # From which the parameters will be later extracted
GregorSpectraSpectralRange = [] # One of the four GALAH spectral bands
GregorSpectraMin = [] # Starting wavelength
GregorSpectraSampling = [] # Sampling

for filename in glob.glob(synth_spectra_dir + '/1*.fits'):
    # Extracts the information from the fits files
    open_fits = fits.open(filename, memmap=False)

    # Extract data for the spectra. The number 4 is for the corrected spectr
    GregorSpectraFlux.append(open_fits[4].data)
    GregorSpectraMin.append(open_fits[4].header['CRVAL1']) # In Armstrongs
    GregorSpectraSampling.append(open_fits[4].header['CDELT1'])

    # Extracts the name of the file: for the ID we remove the last digit of the number, as it
    # corresponds to the spectral region, and we assign this digit to a new array
    GregorSpectraID.append(filename.replace(synth_spectra_dir + '/', '').replace('.fits', '')[:-1])
    GregorSpectraSpectralRange.append(filename.replace(synth_spectra_dir + '/', '').replace('.fits', '')[-1:])

# Gather together all the information from above in order to facilitate the plotting
GregorSpectra = (pd.DataFrame({'Flux' : GregorSpectraFlux,
          'sobject_id' : GregorSpectraID,
          'Region' : GregorSpectraSpectralRange,
          'minWave' : GregorSpectraMin,
          'Sampling' : GregorSpectraSampling})).sort_values(by=['sobject_id']).reset_index(drop=True)


#____________________________________________________________________________________________________
GregorSunlikeSpectraFlux = []
GregorSunlikeSpectraMin = []
GregorSunlikeSpectraSampling = []
for filename in glob.glob(synth_spectra_dir + '/solar*.fits'):
    # Extracts the information from the fits files
    open_fits_solar = fits.open(filename, memmap=False)
    # Extract data for the spectra. The number 4 is for the corrected spectrum
    GregorSunlikeSpectraFlux.append(open_fits_solar[0].data)
    GregorSunlikeSpectraMin.append(open_fits_solar[0].header['CRVAL1']) # IFn Armstrongs
    GregorSunlikeSpectraSampling.append(open_fits_solar[0].header['CDELT1'])

# Maybe it is not necessary to store this in a data frame as it is a small array.
#GregorSun =  (pd.DataFrame({'Flux' : GregorSunlikeSpectraFlux,
#                            'minWave' : GregorSunlikeSpectraMin,
#                            'Sampling' : GregorSpectraSampling}))

#____________________________________________________________________________________________________
def getWaverangeGALAHSpectra(minWave, sampling, amountOfPoints):
    """Construct the waverange for the given parameters of minimum wavelength and sampling and also
       the amount of flux values.

       minWave: beginning of the wavelength range
       sampling: self explanatory
       amount of points: refers to the amount of flux values in the given range. Basically given
       by the lenght of the flux array. """

    waverange = minWave + np.arange(0, amountOfPoints)*sampling
    return waverange

#____________________________________________________________________________________________________
# Need to cross-match the GALAH dataset to find the stars that were sent by Gregor

def crossmatch_GALAH(GALAH_data, sobject_IDs):
    """ The GALAH dataset needs to be imported and given to the function as a parameter. """

    starData = []
    # Loop over all the stars of the input and find the parameters from GALAH, which will be
    # later needed for plotting/synthesis
    for sobject_id_toMatch in sobject_IDs:
        matchedStar_data = GALAH_data[GALAH_data['sobject_id'] == int(sobject_id_toMatch)]
        starData.append(matchedStar_data)

    return starData

# Need to synthesize the stars obtained with the function above
toSynthesize_Gregor = crossmatch_GALAH(GALAH, np.unique(GregorSpectra['sobject_id']))

#____________________________________________________________________________________________________
if plotSpectra == 1:
    plt.figure(figsize=[15,5])
    # Avoid the first and last point as they are 0 and can mess up the later calculations
    plt.plot(waverange[1:-1], spectraWholeRange[1:-1], lw=0.5, c='red', alpha=1)
    for i in range(4):
        wavenrange = getWaverangeGALAHSpectra(GregorSunlikeSpectraMin[i],
                                             GregorSunlikeSpectraSampling[i],
                                             len(GregorSunlikeSpectraFlux[i]))

        plt.plot(wavenrange, GregorSunlikeSpectraFlux[i], lw=1, c='blue', alpha=0.75)


#    plt.title('Spectra of star with $T = {}$, $log \, g = {}$ and [Fe/H] = {}, compared to a sun-like star \
#              from GALAH'.format((5751.68), (4.25), (0.02)))
    plt.xlabel('Wavelength [\AA]')
    plt.ylabel('Normalized Flux')

#____________________________________________________________________________________________________

# Try to plot Gregor's spectra
if plotGregorSpectra == 1:
    fig, ax = plt.subplots(figsize=[15,5], tight_layout=True)

    # Plot the four waveranges together
    for i in range(4, 8):
        # First get the waverange of the given spectra
        wavenrange = getWaverangeGALAHSpectra(GregorSpectra['minWave'][i],
                                             GregorSpectra['Sampling'][i],
                                             len(GregorSpectra['Flux'][i]))

        ax.plot(wavenrange, GregorSpectra['Flux'][i], lw=1.15, c='k', alpha=0.85)

    ax.plot(waverange, spectraCMFlux, lw=1.15, c='red', alpha=0.85, label='Synthetic')
    ax.plot(waverange, np.zeros(len(spectraCMFlux)), lw=1.15, c='k', alpha=0.85, label='GALAH')
#    plt.title('Real GALAH spectra of star') # with $T = {}$, $log \,
#                                        g = {}$ and [Fe/H] = {}'.format((5751.68), (4.25), (0.02)))
    ax.set_xlabel(r'Wavelength $[\AA]$', fontsize=20)
    ax.set_ylabel('Normalized Flux', fontsize=20)
    ax.set_ylim(0.45, 1.15)
    ax.set_xlim(6635, 6695)
    legend = ax.legend(frameon=True, loc='lower right')
    niceplot(fig)
    params = {'legend.fontsize': 18}
    plt.rcParams.update(params)

    plt.savefig('spectracomparison', dpi=150)

#____________________________________________________________________________________________________

if plotSynthVSGALAH == 1:
    plt.figure(figsize=[15,5])
    # 32 to 36 is the real binary star, 4 to 8 is the only cross-matched star
    # Plot the four waveranges together
    for i in range(32, 36):
        # First get the waverange of the given spectra
        wavenrange = getWaverangeGALAHSpectra(GregorSpectra['minWave'][i],
                                             GregorSpectra['Sampling'][i],
                                             len(GregorSpectra['Flux'][i]))

        plt.plot(wavenrange, GregorSpectra['Flux'][i], lw=1, c='red', alpha=1)#, label='Real Spectra')
    # plt.plot(waverange, spectraCMFlux, lw=1, c='blue', alpha=0.75, label='Synthetic Spectrum')
    plt.plot(waverange, np.zeros(len(spectraCMFlux)), lw=1, c='red', alpha=1, label='Binary Spectrum')

#    plt.title('Real GALAH binary spectra of star')# vs synthesized: $T = {}$, $log \, g = \
             # {}$ and [Fe/H] = {}'.format((5560.70), (3.85), (0.25)))
    # with $T = {}$, $log \, g = {}$ and [Fe/H] = {}'.format((5751.68), (4.25), (0.02)))
    plt.xlabel('Wavelength [\AA]')
    plt.ylabel('Normalized Flux')
    plt.legend(frameon=True)
    plt.axvline(x=6562.8, ls=':', c='k', label='$H-\alpha$')
    plt.xlim(6511, 6580)
    plt.ylim(0.1, 1.3)

#____________________________________________________________________________________________________

if plot_GALAHbinaries == 1:
    # Import the data from the GALAH binaries
    galah_binaries = []
    for filename in glob.glob(dirs.data + 'GALAH_binaries' + '/*.txt'):
        # Extracts the information from the fits files
        spectrum = { 'flux' : np.loadtxt(filename)[:,1],
                     'sobject_id' : filename.split('/')[7].replace('.txt', '')}

        galah_binaries.append(spectrum)
    galah_binary_waverange = (np.loadtxt(filename)[:,0])/10
    galah_binaries = pd.DataFrame(galah_binaries)
    print('...Plotting...')
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)

    # Plot the GALAH binaries in adjacent plots
    fig, ax = plt.subplots(6, 2, figsize=[22, 26], tight_layout=True, sharex='col')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.35)
    letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    for i in range(6):
        for j in range(2):
            if j == 0:
                ax[i, 0].plot(galah_binary_waverange, galah_binaries['flux'][i],
                 lw=1.5, c='k', alpha=0.8)
                ax[i, 0].set_xlim(653, 658)
                ax[i, 0].set_ylabel('Normalized Flux')
                ax[i, 0].set_yticks([0.2, 0.4, 0.6, 0.8, 1])

            if j == 1:
                ax[i, 1].plot(galah_binary_waverange, galah_binaries['flux'][i],
                 lw=1.5, c='k', alpha=0.8)
                ax[i, 1].set_xlim(665, 670)
                ax[i, 1].set_ylim(0.725, 1.075)
                ax[i, 1].set_yticks([0.8, 0.9, 1])
                ax[i, 1].text(0.85, 0.175, letters[i],
                 transform=ax[i, 1].transAxes, fontsize=20, verticalalignment='top',
                 bbox=props)

    ax[5, 0].set_xlabel(r'Wavelength [nm]')
    ax[5, 1].set_xlabel(r'Wavelength [nm]')

    plt.rcParams.update({'font.size': 24})
    plt.savefig('binaryspectra.png', dpi=200)

#____________________________________________________________________________________________________

if plot_binaries_presentation == 1:
    # Import the data from the GALAH binaries
    galah_binaries = []
    for filename in glob.glob(dirs.data + 'GALAH_binaries' + '/*.txt'):
        # Extracts the information from the fits files
        spectrum = { 'flux' : np.loadtxt(filename)[:,1],
                     'sobject_id' : filename.split('/')[7].replace('.txt', '')}

        galah_binaries.append(spectrum)
    galah_binary_waverange = (np.loadtxt(filename)[:,0])/10
    galah_binaries = pd.DataFrame(galah_binaries)
    print('...Plotting...')
    props = dict(boxstyle='round', facecolor='white', alpha=0.85)

    # Plot the GALAH binaries in adjacent plots
    # Plot the GALAH binaries in adjacent plots
    fig, ax = plt.subplots(3, 1, figsize=[20, 16], tight_layout=True, sharex='col')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.35)
    selected_indexes = [1, 0, 3]
    for j, i in zip(selected_indexes, range(3)):
        ax[i].plot(galah_binary_waverange, galah_binaries['flux'][j],
         lw=2, c='k', alpha=0.8)
        ax[i].set_xlim(653, 658)
        ax[i].set_ylabel('Normalized Flux', fontsize=30)
        ax[i].set_yticks([0.2, 0.4, 0.6, 0.8, 1])
        ax[i].tick_params(axis='both', which='major', labelsize=28)
        ax[i].tick_params(axis='both', which='minor', labelsize=28)

    ax[2].set_xlabel(r'Wavelength [nm]', fontsize=30)

    plt.rcParams.update({'font.size': 28})
    plt.savefig('binaryspectra_presentation.png', dpi=200)
