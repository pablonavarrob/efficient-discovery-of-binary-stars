import socket


class directories():
    """ Import directories and things that are used all the time. """

    name_analysis_run = 'run_intervals_25nm'

    def __init__(self):
        computer_name = socket.gethostname()
        if computer_name == 'astrolabe':
            self.GALAH_data = '/home/pablo/Data/'
            self.iSpec = '/home/pablo/iSpec/'
            self.single_synth_spectra = ('/home/pablo/'
                                         'wholeRangeSynthesis_100000/'
                                         'stellarSynthesis/')
            # In the mounted disk
            self.data = '/mnt/data/Pablo/data/'
            self.figures = '/mnt/data/Pablo/figures/'
            self.tsne_results = self.data + self.name_analysis_run + '/tsne_results/'
            self.dbscan_results = self.data + self.name_analysis_run + '/dbscan_results/'
            self.single_spectra = self.data + self.name_analysis_run + '/single_spectra/'
            self.samples_to_tsne = self.data + self.name_analysis_run + '/samples_to_tsne/'


        else:
            self.GALAH_data = ('/Users/pablonavarrobarrachina/Desktop/Master/'
                               'Master-Project/Code/GALAH_data/')
            self.iSpec = ('/Users/pablonavarrobarrachina/iSpec/')
            self.single_synth_spectra = ('/Users/pablonavarrobarrachina/Desktop/'
                                         'Scripts/data/spectra_read_chunk*')
            self.data = ('/Users/pablonavarrobarrachina/Desktop/Scripts/data/')
            self.figures = (
                '/Users/pablonavarrobarrachina/Desktop/Scripts/figures/')
            self.tsne_results = self.data + self.name_analysis_run + '/tsne_results/'
            self.dbscan_results = self.data + self.name_analysis_run + '/dbscan_results/'
            self.single_spectra = self.data + self.name_analysis_run + '/single_spectra/'
            self.samples_to_tsne = self.data + self.name_analysis_run + '/samples_to_tsne/'
