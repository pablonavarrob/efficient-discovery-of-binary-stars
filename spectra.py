import numpy as np
import pandas as pd
import socket
from astropy.io import fits
import glob as glob
import matplotlib.pyplot as plt
# %matplotlib qt
# Import other modules
from directories import directories
from galah import GALAH_survey
from plotting import plotting_main

# Necessary to maintain the synthesis order.
np.random.seed(1)

def load_synthetic_single_spectra(spectra_file_dir, file_format = '*.npy'):
    """ Loads the synthetic single spectral data from the -several - files
    containing that information. Default file format is .npy but this
    can be changed at will. """

    synthetic_single_fluxes = []
    # Extract spectra from files
    for file in glob.glob(spectra_file_dir + file_format):
        temp = np.load(file)
        synthetic_single_fluxes.append(temp)

    return np.vstack(synthetic_single_fluxes)

def load_synthetic_binary_population(binary_population_dir):
    """ Simple function to return the read dataframe corresponding to the
    binary population. """
    return pd.read_csv(binary_population_dir)

def load_synthetic_binary_spectra(binary_spectra_dir, file_format = '.npy'):
    """ Simple function to return the spectral data
     of the synthesized binaries. """
    return np.load(binary_spectra_dir + file_format)

def load_synthetic_stellar_parameters():
    return pd.read_csv('computed_data/stellar_parameters_dataframe.csv')


class spectra():
    """ Main class for synthetic stellar spectra. Contains methods
    used by both single and binary classes.

    The displacement_binary_range corresponds to the correction factor
    applied during the binary interpolation, it is fixed at 0.5 but can be
    changed at will.  """

    sampling = 0.004
    resolution = 2.8e4
    displacement_binary_range = 0.5 # nanometers
    factor = displacement_binary_range/sampling


    def __init__(self, spectra, spectral_ranges):
        self.spectra = spectra
        self.spectral_ranges = spectral_ranges

    def get_waverange(self):
        waveranges = []
        for range in self.spectral_ranges:
            waverange = np.arange(range[0], range[1], self.sampling)
            waveranges.append(waverange)
        return waveranges

    def get_corrected_spectra(self):
        """ Corrects the spectra of the single stars to match the length of the
        binary stars (which has been shortened due to the interpolation). """

        # Initialize the index count
        indexes_list = 0 # counts the indexes in the whole wavelength array
        corrected_spectra = []
        waveranges = self.get_waverange()

        for range in waveranges:
            spectra_chunk = self.spectra[:,
             int(indexes_list + self.factor): int(indexes_list + len(range) - self.factor)]
            indexes_list += len(range)
            corrected_spectra.append(spectra_chunk)

        return np.hstack(corrected_spectra)

    def get_corrected_waverange(self):
        """ Corrects the waveranges to match that of the binary stars. """

        # List comprehension
        corrected_waveranges = [np.arange(range[0] + self.displacement_binary_range,
         range[1] - self.displacement_binary_range, self.sampling) for
                                        range in self.spectral_ranges]

        return corrected_waveranges


class single_spectra(spectra, plotting_main):
    """ Sub-class for single synthetic spectra. Load the same parameter file
    as for the binaries, but select only the stars that really are singles.

    Load parameter file with the following naming:
        - teff_A for the effective temperature
        - logg_A for the surface gravity
        - feh_A for the metallicity
        - id_A for the sobject_id
    """

    def __init__(self, spectra, spectral_ranges, stellar_parameters, figure_size=[15,5]):
        super().__init__(spectra, spectral_ranges)
        self.figure_size = figure_size
        self.amount_singles = len(self.spectra)
        self.teff = stellar_parameters['teff_A'][0:self.amount_singles]
        self.logg = stellar_parameters['logg_A'][0:self.amount_singles]
        self.fe_h = stellar_parameters['feh_A'][0:self.amount_singles]
        self.sobject_id = stellar_parameters['id_A'][0:self.amount_singles]
        self.waveranges = self.get_waverange()
        # Gets the spectra arrays and the waverange arrays corrected to match
        # the length and wavelength space of that of the binaries
        self.corrected_spectra = self.get_corrected_spectra()
        self.corrected_waveranges = self.get_corrected_waverange()

    def plot(self, index_spectra):
        """ Plots the selected synthetic star fluxes. If multiple,
         need to be given as a tuple. """
        amount_ranges = len(self.spectral_ranges)
        # Define several colors,
        colors = ['black', 'red', 'cyan', 'green', 'yellow', 'orange', 'grey']
        fig, ax = plt.subplots(figsize=self.figure_size, constrained_layout=True)
        # This will plot as many spectra overlapped as given by index spectra
        if type(index_spectra) is int:
            index_count = 0
            for range in self.waveranges:
                range_length = len(range)
                ax.plot(range, self.spectra[index_spectra][index_count: index_count + range_length],
                lw=1.75, alpha=0.75, c=colors[0])
        else:
            color_count = 0
            for index in list(index_spectra):
                index_count = 0 # To account for the different waveranges
                for range in self.waveranges:
                    range_length = len(range)
                    ax.plot(range, self.spectra[index][index_count: index_count + range_length],
                    lw=1.75, alpha=0.75, c=colors[color_count])
                    index_count += range_length
                color_count += 1

        ax.set_xlabel('Wavelength $[\AA]$')
        ax.set_ylabel('Normalized Flux')

    def plot_corrected(self, index_spectra):
        """ Plots the selected synthetic star fluxes. If multiple,
         need to be given as a tuple. """
        amount_ranges = len(self.spectral_ranges)
        # Define several colors,
        colors = ['black', 'red', 'cyan', 'green', 'yellow', 'orange', 'grey']
        fig, ax = plt.subplots(figsize=self.figure_size, constrained_layout=True)
        # This will plot as many spectra overlapped as given by index spectra
        if type(index_spectra) is int:
            index_count = 0
            for range in self.corrected_waveranges:
                range_length = len(range)
                ax.plot(range, self.corrected_spectra[index_spectra][index_count: index_count + range_length],
                lw=1.75, alpha=0.75, c=colors[0])
        else:
            color_count = 0
            for index in list(index_spectra):
                index_count = 0 # To account for the different waveranges
                for range in self.corrected_waveranges:
                    range_length = len(range)
                    ax.plot(range, self.corrected_spectra[index][index_count: index_count + range_length],
                    lw=1.75, alpha=0.75, c=colors[color_count])
                    index_count += range_length
                color_count += 1

        ax.set_xlabel('Wavelength $[\AA]$')
        ax.set_ylabel('Normalized Flux')


class binary_spectra(spectra, plotting_main):
    """ Holds information about the binaries from the data-set..?
    Subscript A is for primary star and B for secondary. """

    def __init__(self, spectra, spectral_ranges, stellar_parameters, figure_size=[15,5]):
        super().__init__(spectra, spectral_ranges)
        self.spectra = spectra
        self.spectral_ranges = spectral_ranges
        self.figure_size = figure_size
        # The corrected waveranges are the ones corresponding to that of
        # the binary grid, therefore it's called here simply waverange
        self.waveranges = self.get_corrected_waverange()

        # Define stellar parameters
        self.amount_singles = len(stellar_parameters[stellar_parameters['binarity'] == 0])
        self.amount_binaries = len(stellar_parameters[stellar_parameters['binarity'] == 1])
        self.teff_A = (stellar_parameters['teff_A'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.teff_B = (stellar_parameters['teff_B'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.logg_A = (stellar_parameters['logg_A'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.logg_B = (stellar_parameters['logg_B'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.feh_A = (stellar_parameters['feh_A'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.feh_B = (stellar_parameters['feh_B'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.rad_vel = (stellar_parameters['rad_vel'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.sobject_id_A = (stellar_parameters['id_A'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.sobject_id_B = (stellar_parameters['id_B'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.lum_ratio = (stellar_parameters['lum ratio'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.mass_A = (stellar_parameters['mass_A'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)
        self.mass_B = (stellar_parameters['mass_B'][self.amount_singles: self.amount_singles+
                                                self.amount_binaries]).reset_index(drop=True)

    def plot(self, index_spectra):
        """ Plots the selected binary star fluxes. If multiple,
         need to be given as a tuple. """
        amount_ranges = len(self.spectral_ranges)
        # Define several colors,
        colors = ['black', 'red', 'cyan', 'green', 'yellow', 'orange', 'grey']
        fig, ax = plt.subplots(figsize=self.figure_size, constrained_layout=True)
        # This will plot as many spectra overlapped as given by index spectra
        color_count = 0
        for index in list(index_spectra):
            index_count = 0 # To account for the different waveranges
            for range in self.waveranges:
                range_length = len(range)
                ax.plot(range, self.spectra[index][index_count: index_count + range_length],
                                                vlw=1.75, alpha=0.75, c=colors[color_count],
                label="$\Delta v_{R}$ = % .2f, $L_{B}/L_{A}$ = % .2f" % (self.rad_vel[index],
                                         self.lum_ratio[index]) if index_count == 0 else "")
                index_count += range_length
            color_count += 1

        ax.set_xlabel('Wavelength $[\AA]$')
        ax.set_ylabel('Normalized Flux')
        legend = ax.legend(loc='lower right')

if __name__ == 'main':
    dirs = directories()
    spectra_dir = dirs.single_synth_spectra
    binary_spectra_dir = dirs.binary_synth_spectra
    ranges_to_cut = [(460, 470), (650, 680), (715, 720)]
    stellar_parameters = pd.read_csv("/Users/pablonavarrobarrachina/Desktop/Scripts/data/stellar_parameters_duchenekrauspopulation.csv")
    single_synth_spectra = load_synthetic_single_spectra(spectra_dir)
    binary_synth_spectra = load_synthetic_binary_spectra(binary_spectra_dir)
    singles = single_spectra(single_synth_spectra, ranges_to_cut, stellar_parameters)
    binaries = binary_spectra(binary_synth_spectra, ranges_to_cut, stellar_parameters)
