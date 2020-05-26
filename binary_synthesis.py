import numpy as np
from analyseSpectralData_functions import get_waverange


def doppler_shift(wavelength_rest, v_rad):
    """ Calculates the corresponding Doppler shift for a given radial velocity value.
    The input wavelength rest is the synthetic spectra imported from the fits file and the
    radial velocity value is randomly drawn from the Gaussian distribution. """
    c = 299792.458  # km/s
    ratio = v_rad / c + 1
    wavelength_shift = wavelength_rest * ratio
    return wavelength_shift


def generate_spectra_binary(spectra_primary, spectra_secondary,
                            v_distribution, lum_ratio, selected_wavelengths,
                            normalize_main_star, input_v_rad, sampling=0.004):
    """ Combines input spectra into the spectra of a binary system.

    The wavelengths input is referred to the wave_b and wave_t of each one
    of the waveranges for the input spectra. These wavelengths are tuples and
    if used from the same analyseSpectralData script, they are the
    ranges to cut list.

    For the radial velocity distributions:
        - 'gaussian'
        - 'exponential'
        - 'beta'

    The GALAH spectral band is selected similar as in the single spectra synthesis.
    Infrared, red, green and blue are the single bands and whole encompasses
    all of them.

    Sampling is that of the input synthetic spectra.
    Normalize main star sets the radial velocity of the main star to 0.    """

    # ______________________________________ Distributions ____________________ #
    # Sample parameters from random distributions
    # Radial velocity difference distribution in km/s
    if len([input_v_rad]) == 0:
        if v_distribution == 'gaussian':
            v_rads = np.random.normal(0, 30, 2)

        if v_distribution == 'exponential':
            v_rads = ((-1)**np.random.randint(0, 9, 2)) * \
                np.random.exponential(30, 2)

        if v_distribution == 'beta':
            v_rads = ((-1)**np.random.randint(0, 9, 2)) * \
                300 * np.random.beta(1, 3, 2)

        else:
            print('Not defined')

    if len([input_v_rad]) == 1:
        v_rads = [input_v_rad]

    # ___________________________________ Combine the spectra _________________ #
    # Create the binary spectra - main steps of the process:
        # 1. Calculate the flux coefficients with the formulas given
        # 2. Shift and then interpolate with numpy, both spectra to a smaller
        # grid containing both
        # 3. Sum the fluxes and renormalize to 1

    # Step 1. Calculate the flux coefficients from the flux ratio
    a = 1.0 / (1.0 + lum_ratio)
    b = lum_ratio / (1.0 + lum_ratio)

    # Quantities are sampled and calculated before the main loop

    # _________________________________________________________________________

    # Step 2. Doppler shift the spectra according to the radial velocity, done
    # for each one of the spectra

    # Input wave ranges are separated and we need to join them in a single
    # array
    wave_range = []
    for range in selected_wavelengths:
        partial_waverange = get_waverange(range)
        wave_range.append(partial_waverange)
    wave_range = np.hstack(wave_range)

    # Define the shift of the primary star, depending on wether we normalize
    # to the main star or not
    if normalize_main_star == True:
        shift_waverange_1 = doppler_shift(wave_range, 0)
        # Shift of the secondary star
        shift_waverange_2 = doppler_shift(
            wave_range, v_rads[0])  # Secondary spectra

    else:
        shift_waverange_1 = doppler_shift(
            wave_range, v_rads[0])  # Primary spectra
        # Shift of the secondary star
        shift_waverange_2 = doppler_shift(
            wave_range, v_rads[1])  # Secondary spectra

    # Define a shortened wave range for the interpolation
    interpolation_waverage = []
    for range in selected_wavelengths:
        wave_b_interpolation = range[0] + 0.5
        wave_t_interpolation = range[1] - 0.5
        partial_interpolation_waverange = get_waverange(
            (wave_b_interpolation, wave_t_interpolation))
        interpolation_waverage.append(partial_interpolation_waverange)
    # hstack in order to get the same shape as before
    interpolation_waverange = np.hstack(interpolation_waverage)

    # Interpolate to new grid
    interpolated_spectra_1 = np.interp(interpolation_waverange, shift_waverange_1,
                                       spectra_primary.flatten())
    interpolated_spectra_2 = np.interp(interpolation_waverange, shift_waverange_2,
                                       spectra_secondary.flatten())

    # Step 3. Sum the fluxes to obtain the combined spectra, save as a
    # separate array
    binary_spectrum = a * interpolated_spectra_1 + b * interpolated_spectra_2

    return binary_spectrum, v_rads[0], interpolation_waverange


def generate_binary_grid(wave_b, wave_t, sampling):
    waverangeBinaries = np.arange(wave_b + 0.5, wave_t - 0.5, sampling)
    return waverangeBinaries
