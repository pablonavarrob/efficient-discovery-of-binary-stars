# Functions for the spectral analysis file
from astropy.io import fits
import glob
import numpy as np
import pandas as pd
import itertools
import socket
from tqdm import tqdm


def add_noise(SNR, spectra):
    """ Adds noise to the spectra according to the input SNR value, which
    accounts for the standard deviation of the Gaussian distribution.

    From wikipedia, it SNR can be described as SNR = mu/sigma, where mu is
    the expected value of the target value and sigma the standard distribution
    of the noise. """

    # # Create array of noise and add it to the input spectra
    # noisy_spectra = []
    # for spectrum in spectra:
    #     noisy_spectrum = spectrum + np.random.normal(0, 1/SNR, len(spectrum))
    #     noisy_spectra.append(noisy_spectrum)
    #
    # return np.round(np.vstack(noisy_spectra), 3)

    return spectra + np.random.normal(0, 1/float(SNR), spectra.shape)


def logg_zwitter(teff):
    """ Computes the logg to be used later to differentiate between dwarf and giant. Filter stars
    wether they are giants or dwarfs with equation 1 from Zwitter, T. et al. 2018."""

    logg = 3.2 + (4.7 - 3.2) * (6500 - teff) / (6500 - 4100)

    return logg


def get_data_run(numberSynthStars, data_dir, use_id_list=False, remove_corrupt=True):
    """ iSpec generation into a function. Loops over all of the stars. The names of the outputs
    are the name of the run followed by the effective temperature, gravity and metallicty of the
    star. Selectable are the different GALAH spectral bands.

    A necessary input parameter is the directory of the real spectral data, called 'data_dir'.

    At this point the function generates spectra for dwarfs. Can be easily changed. """

    survey_spectra = fits.open(data_dir + '/GALAH_DR2.1_catalog.fits.txt')

    try:
        id_list = np.load('computed_data/wholeRange_ids.npy')
        if remove_corrupt == True:
            corrupt_files = np.load('computed_data/ids_corrupt_files.npy')
            # Remove the bad id's
            mask_corrupt = pd.DataFrame(id_list)[0].isin(corrupt_files)
            id_list = id_list[~mask_corrupt]
            print('Got file for corrupted fits')
    except:
        pass
    # Open GALAH data
    data = survey_spectra[1].data

    # Quality cuts: only stars flagged with 0 are good for using in this synthesis
    # Creates the mask - in this case written for the
    mask_flag = data['flag_cannon'] == 0
    # GALAH survey.
    data_filter_flag = data[mask_flag]   # Masks all the raw data
    # __________________________________ Select dwarf stars ______________________________________ #
    # Select the dwarfs from the whole dataset.
    dwarfs = data_filter_flag[logg_zwitter(
        data_filter_flag['teff']) <= data_filter_flag['logg']]

    if use_id_list == True:
        # Transform into a dataframe to obtain the mask and then mask the regular
        # array. Saves a lot of time!
        stars_ = pd.DataFrame(dwarfs)
        mask = stars_['sobject_id'].isin(id_list)

        # Mask the original arrays
        stars_run = dwarfs[mask]

        print("=" * 40)
        print("The amount of stars selected for this run is {}, out of a total of {}. A list of ids was used"
              .format(len(stars_run), len(dwarfs)))

    else:
        # Defines the set of stars from which the parameters will be drawn in the current synthesis run
        stars_run = dwarfs[np.random.permutation(
            len(dwarfs))][0:numberSynthStars]

        print("=" * 40)
        print("The amount of stars selected for this run is {}, out of a total of {}".format(
            len(stars_run), len(dwarfs)))

    # Wavelength ranges in nanometers, where wave_b is the beginning wavelength and the wave_t is the
    # end wavelength. Wavelength is in angstroms

    return stars_run


def get_wavelengths(GALAH_band):
    """ Returns the wavelengths, base and top, for a given GALAH region.
    The first ouput is the lower limpit of the wavelength range, wave_b,
    and the second ouput is the higher limit, wave_t. """

    if GALAH_band == 'infrared':
        wave_b = 759.0
        wave_t = 789.0

    if GALAH_band == 'red':
        wave_b = 648.1
        wave_t = 673.9

    if GALAH_band == 'green':
        wave_b = 564.9
        wave_t = 587.3

    if GALAH_band == 'blue':
        wave_b = 471.8
        wave_t = 490.3

    if GALAH_band == 'whole':
        wave_b = 450.0
        # 900 because Calcium Triplet and contained in GAIA data.
        wave_t = 900.0

    return (wave_b, wave_t)


def get_waverange(wavelength_limits, sampling=0.004):
    return np.arange(wavelength_limits[0], wavelength_limits[1], sampling)


def select_waverange(spectra_to_cut, wavelength_ranges_input, wavelength_ranges_cut, sampling=0.004):
    """ Cuts the input spectra in the given ranges. Ranges are given in
    tuples. If ranges > 1, introduce as a list.

    The function is hoped to be flexible, that is: for any given range of
    input wavelengths within the waverange on which the to-cut spectra is
    defined can be selected. """

    waverange_input = get_waverange(wavelength_ranges_input)

    # For each one of the tuples, we can define a new mask and mask the
    # input waverange array with it.
    masks = [(waverange_input >= ranges[0]) &
             (waverange_input <= ranges[1]) for ranges in wavelength_ranges_cut]

    # Cut the spectra and the waveranges and save it
    cut_spectra = [spectra_to_cut[mask] for mask in masks]
    cut_waveranges = [waverange_input[mask] for mask in masks]

    return cut_spectra, cut_waveranges


def read_spectra_fits(synth_spectra_directory, ranges_selection, sampling=0.004,
                      initial_waverange='whole', run_name=[]):
    """ Returns arrays containing the cut spectra. Each array contains subarrays
    depending on the number of ranges given. """

    print("=" * 40)
    print('Importing data and extracting filenames. Selecting wavelengths in the {} spectral range'.format(
        ranges_selection))

    singleSynthSpectra = []  # Synthesized spectra for single stars
    selectedWaveranges = []
    corrupt_files = []
    i = 0

    for filename in tqdm(glob.glob(synth_spectra_directory + '/*.fits')):
        # Add an i no matter if there's an error or not.
        i += 1
        open_fits = fits.open(filename, memmap=False)
        # Select a given wavelength range in order to avoid memory issues
        synthSpectrum, selected_waveranges = select_waverange(
            open_fits[0].data, get_wavelengths(
                '{}'.format(initial_waverange)), ranges_selection)
        # # Save the spectrum as a single row of flux values and round
        # to three decimal ciphers
        singleSynthSpectra.append(np.round(np.hstack(synthSpectrum), 3))
        if len(selectedWaveranges) == 0:
            selectedWaveranges.append(selected_waveranges)
            print('Wave range saved.')

        open_fits.close()
        del open_fits

    print("=" * 40)
    print('--> Import, filename extraction and spectral region selection completed')

    return singleSynthSpectra, selectedWaveranges, corrupt_files


def get_corrupt_fits_files(synth_spectra_directory, run_name):
    """ Extract the non-readable fits files. """

    corrupt_files = []
    for filename in tqdm(glob.glob(synth_spectra_directory + '/*.fits')):
        try:
            open_fits = fits.open(filename, memmap=False)
            open_fits.close()
            del open_fits
        except:
            id_corrupt = filename.replace(synth_spectra_directory, '').replace(
                run_name + '_', '').replace('.fits', '').replace('/', '').split('_')
            print('Corrupt fits file with sobject_id {}'.format(id_corrupt[0]))
            corrupt_files.append(int(id_corrupt[0]))

    return corrupt_files


def extract_parameters_filename(filename_list, run_name, synth_spectra_directory, get_filenames=False):

    print("=" * 40)
    print('--> Extracting parameters from the filenames')

    extraCharacterLength = len(synth_spectra_directory + run_name + '_')

    if get_filenames == True:
        filename_list = []
        for file in glob.glob(synth_spectra_directory + '/*.fits'):
            filename_list.append(file)

    # Define variable for the lenght of the spectral being handled to speed the writing
    n_spectra = len(filename_list)
    # Obtain the parameters from the filenames for the single stars
    parameters = []

    for filename in filename_list:
        # The numbers below were obtained from the files and should not be changed unless the
        # output format is changed form the file ispec_control_all_variables.
        try:
            params = filename[extraCharacterLength:]
            id_ = int(params[0:15])
            teff = params[16:23]
            logg = params[24:28]
            if len(params) >= 39:
                fe_h = params[29:34]
            else:
                fe_h = params[29:33]

            star_params = {'teff_A': teff, 'logg_A': logg,
                           'feh_A': fe_h, 'id_A': id_}
            parameters.append(star_params)
        except TypeError:
            print('Star with name {} produced an error'.format(
                filename))

    print("=" * 40)
    print('--> Parameter extraction finished')
    return pd.DataFrame(parameters)


def get_tsne_sample(single_spectra, binary_spectra, selected_waveranges):
    """Cuts single spectra properly to match that of the
    singles. """

    # Sum the max of each range for the cutting
    indexes_list = 0
    cut_single_spectra = []

    for range in selected_waveranges:
        waverange_single = get_waverange(range)
        spectra_chunk = single_spectra[:,
                                       indexes_list + 125: indexes_list + len(waverange_single) - 125]
        indexes_list += len(waverange_single)
        cut_single_spectra.append(spectra_chunk)

    tSNE_sample = np.concatenate(
        (np.hstack(cut_single_spectra), binary_spectra))

    return np.round(tSNE_sample, 3)


def check_astrolabe(password, directory_to_check):
    """ Connects to astrolabe to check the stars in the given directory. """
    import pysftp

    myHostname = 'astrolabe.astro.lu.se'
    myUsername = 'pablo'
    myPassword = password

    list_of_wholeRange_stars = []

    with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword) as sftp:
        print("Connection succesfully stablished ... ")

        # Switch to a remote directory
        sftp.cwd(directory_to_check)

        # Obtain structure of the remote directory '/var/www/vhosts'
        directory_structure = sftp.listdir_attr()

        # Print data
        i = 0
        for attr in directory_structure:
            list_of_wholeRange_stars.append(attr.filename)
            i += 1
            if i % 100 == 0:
                print(i)

    # All the id's of the whole range synthetic stars
    wholeRange_ids = [int((name.split('_'))[3])
                      for name in list_of_wholeRange_stars[1:-7]]

    return wholeRange_ids
