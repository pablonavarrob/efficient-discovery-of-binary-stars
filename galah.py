import numpy as np
from directories import directories
from astropy.io import fits
import pandas as pd


class GALAH_survey(directories):
    """ Returns relevant data from the GALAH survey. Either ranges of wavelength,
    or also the raw and the filtered data. Filtering is done to get
    the dwarf stars present in the data set. """

    number_synth_stars = 100000

    def __init__(self):
        directories.__init__(self)
        # super(directories, self).__init__()
        self.infrared_band = (759.0, 789.0)
        self.red_band = (648.1, 673.9)
        self.green_band = (564.9, 587.3)
        self.blue_band = (471.8, 490.3)
        self.whole_band = (450.0, 900.0)

    def get_raw_data(self):
        self.raw_data = fits.open(self.GALAH_data +
                                  'GALAH_DR2.1_catalog.fits.txt')[1].data
        print("GALAH raw data succesfully imported.")

    def get_dwarfs(self):
        """ The prescription for the selection of the dwarf stars is
        from Zwitter, T. et al. 2018. All dwarf stars from the data set
        are selected with this function. """

        self.get_raw_data()
        mask_flag = self.raw_data['flag_cannon'] == 0
        logg = 3.2 + (4.7 - 3.2) * \
            (6500 - self.raw_data['teff']) / (6500 - 4100)
        self.dwarfs = self.raw_data[mask_flag][logg[mask_flag] <=
                                               self.raw_data[mask_flag]['logg']]

        print("Dwarf stars from the GALAH survey loaded.")

    def get_stars_run(self):
        """Selects number_synth_stars for a given run. Not necessary unless a file cannot
        be separately loaded. """

        # Filter the corrupt files + those ids that are synthesized
        id_list = np.load(self.data + 'wholeRange_ids.npy')
        corrupt_files = np.load(
            self.data + 'ids_corrupt_files.npy')
        # Remove the id's
        mask_corrupt = pd.DataFrame(id_list)[0].isin(corrupt_files)
        # Masked id list
        id_list = id_list[~mask_corrupt]
        print('Got file for corrupted fits')

        # Call the previous function which calls the previous one as well
        self.get_dwarfs()
        stars_ = pd.DataFrame(self.dwarfs)
        mask = stars_['sobject_id'].isin(id_list)
        self.stars_run = self.dwarfs[mask]

        # Print statements for control
        print("=" * 40)
        print("The amount of stars selected for this run is {}, out of a total of {}".format(
            len(self.stars_run), len(self.dwarfs)))
