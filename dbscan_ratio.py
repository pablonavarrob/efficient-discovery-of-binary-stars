import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import directories as directories
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt

################################################################################
# Define the niceplot function from Gregor #####################################

def niceplot(fig, labelsize=20, tight_layout=True, remove_firstandlast_tick = True):
    import matplotlib.ticker as ticker
    ax_list = fig.axes
    for ax in ax_list:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.tick_params(direction="in", which='major', right=True, top=True, length=10, width=2)
        ax.tick_params(direction="in", which='minor', right=True, top=True, length=5, width=1)
        try:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator((ax.get_yticks()[1]-ax.get_yticks()[0])/5.))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator((ax.get_xticks()[1]-ax.get_xticks()[0])/5.))
        except:
            pass
        if remove_firstandlast_tick == True:
            ax.set_xticks(ax.get_xticks()[1:-1])
            ax.set_yticks(ax.get_yticks()[1:-1])

        else:
            ax.set_xticks(ax.get_xticks())
            ax.set_yticks(ax.get_yticks())

    if tight_layout == True:
        fig.set_tight_layout(True)
    else:
        pass

################################################################################

class dbscan_method():

    ratio = 0.9
    iterations = 10
    normalization_constant = 10
    use_normalized_data = True
    colormap = 'plasma'
    # matplotlib.rcParams['text.usetex'] = True
    save_plots = True
    save_plots_dir = directories.directories().figures

    # DBSCAN parameters - can be overwritten from within the class
    min_eps = 0.1
    max_eps = 0.75
    min_minpts = 25
    max_minpts = 125

    """ Apply DBSCAN to the results from the t-SNE analysis.

    The iterations parameter corresponds to how many values of epsilon
    and min pts will be used for the exploration of the parameter space

    The following parameters will be added to the instance after the main
    algorithm is run once (names are self-explanatory):

    self.parameter_space
    self.optimized_eps
    self.optimized_minpts
    self.dbscan_labels
    self.recovery_ratio_optimized

    """

    def __init__(self, tsne_x, tsne_y, real_labels, stellar_parameters,
                 spectral_range, SNR, perplexity):

        self.tsne_x = tsne_x
        self.tsne_y = tsne_y
        self.real_labels = real_labels
        # self.min_eps = min_eps
        # self.max_eps = max_eps
        # self.min_minpts = min_minpts
        # self.max_minpts = max_minpts
        self.stellar_parameters = stellar_parameters
        self.spectral_ranges = spectral_range
        self.SNR = SNR
        self.perplexity = perplexity

        # These parameters will only be available after running the corresponding
        # functions, which are defined underneath
        self.normalized_tsne_x = []
        self.normalized_tsne_y = []

    def normalize_data_tSNE(self):
        """ Returns the t-SNE coordinates normalized between -1 and 1 times a
         predefined constant k. """
        self.normalized_tsne_x = (2 * (self.tsne_x - np.min(self.tsne_x)) /
                                  (np.max(self.tsne_x) - np.min(self.tsne_x))
                                  - 1) * self.normalization_constant
        self.normalized_tsne_y = (2 * (self.tsne_y - np.min(self.tsne_y)) /
                                  (np.max(self.tsne_y) - np.min(self.tsne_y))
                                   - 1) * self.normalization_constant

    def find_binaries_ratio_dbscan(self, labels_DBSCAN):
        """ Find binaries from within the t-SNE maps using the ratio
        prescription. Input has to be a value of epsilon and minPts, possibly
        optimized using the function below.  """

        # Create the DBSCAN labels to search binaries for
        # Find all the unique labels found by DBSCAN
        unique_labelsDBSCAN = np.unique(labels_DBSCAN)

        # Initialize array to save the corrected labels
        ratioLabels = np.zeros(len(self.real_labels))

        # Loop over the unique labels and compare them to the ones manually defined
        for DBSCANLabel in unique_labelsDBSCAN:
            # To avoid the points DBSCAN marked as noise
            if DBSCANLabel != -1:
                # Get the real labels corresponding to the elements of DBSCANLabel
                filtered_realLabels = self.real_labels[labels_DBSCAN == DBSCANLabel]

                # Get the indexes to save the corrected label in the corrected spot
                indexes_filtered_realLabels = np.argwhere(
                    [labels_DBSCAN == DBSCANLabel])[:, 1]

                # Get the amount of real singles and real binaries in the filtered array
                filtered_singles = len(
                    filtered_realLabels[filtered_realLabels == 0])
                filtered_binaries = len(
                    filtered_realLabels[filtered_realLabels == 1])

                # Get the ratio
                try:
                    ratio_ = float(filtered_binaries) / \
                        float((filtered_singles + filtered_binaries))
                except ZeroDivisionError:
                    pass

                if ratio_ >= self.ratio:
                    # 1 as the binary label
                    np.add.at(ratioLabels, indexes_filtered_realLabels, 1)
                elif filtered_singles == 0:
                    # 1 as the binary label
                    np.add.at(ratioLabels, indexes_filtered_realLabels, 1)
                elif filtered_binaries == 0:
                    # 0 as the binary label
                    np.add.at(ratioLabels, indexes_filtered_realLabels, 0)
                else:
                    # 0 as the binary label
                    np.add.at(ratioLabels, indexes_filtered_realLabels, 0)

            else:
                # Mark noise as singles
                np.add.at(ratioLabels, np.argwhere(labels_DBSCAN == -1), 0)

        return ratioLabels

    def explore_parameter_space(self):
        """ Find binary stars from within the t-SNE maps using DBSCAN and a
        predefined ratio of binaries per cluster. """

        # Define the arrays to loop
        epsilon_explore_parameter_space = np.linspace(
            self.min_eps, self.max_eps, self.iterations)
        minpts_explore_parameter_space = np.linspace(
            self.min_minpts, self.max_minpts, self.iterations)

        # The sum of the normalized arrays must be zero if nothing has been
        # assigned to them, as they are initialized as empty lists.
        if (self.use_normalized_data == True and
                len(self.normalized_tsne_x) + len(self.normalized_tsne_y) == 0):
            # Call the function as it hadn't been called before
            self.normalize_data_tSNE()
            tSNEData_ = pd.DataFrame(
                {'x': self.normalized_tsne_x, 'y': self.normalized_tsne_y})
            print('Exploring DBSCAN using the normalized data...')

        # If the data has been normalized before, then the sum won't be zero
        # and the normalized data will be used directly without calling the function
        elif (self.use_normalized_data == True and
              len(self.normalized_tsne_x) + len(self.normalized_tsne_y) != 0):
            tSNEData_ = pd.DataFrame(
                {'x': self.normalized_tsne_x, 'y': self.normalized_tsne_y})
            print('Exploring DBSCAN using the normalized data...')

        elif self.use_normalized_data == False:
            tSNEData_ = pd.DataFrame({'x': self.tsne_x, 'y': self.tsne_y})
            print('Exploring DBSCAN...')

        # Initialize the counting
        exploration_results = []
        i = 0

        # Loop over all of the possible combinations in order to obtain better statistics
        for epsilon in epsilon_explore_parameter_space:
            for min_pts in minpts_explore_parameter_space:
                # Calculate DBSCAN
                labelsDBSCAN = DBSCAN(
                    eps=epsilon, min_samples=min_pts).fit(tSNEData_).labels_

                # Get the labels from the above function
                ratioLabels = self.find_binaries_ratio_dbscan(labelsDBSCAN)

                # Need to count the amount of singles and binaries after the DBSCAN ratio method
                amountSingles = len(ratioLabels[ratioLabels == 0])
                amountBinary = len(ratioLabels[ratioLabels == 1])
                exploration_results.append(
                    (amountSingles, amountBinary, epsilon, min_pts))

                recovery = amountBinary / \
                    len(self.real_labels[self.real_labels == 1])
                # Counter
                i += 1
                if i%100 == 0:
                    print('Step', i,
                     ':: current mode: epsilon %.2f and min pts %.2f' % (epsilon, min_pts))


        # Convert the results into a dataframe for easier handling
        self.parameter_space = pd.DataFrame(exploration_results)
        self.parameter_space.columns = ['Singles', 'binaries', 'eps', 'minpts']
        # Add the column for the recovery ratio
        self.parameter_space['recovery_ratio'] = self.parameter_space['binaries'] / \
            len(self.real_labels[self.real_labels == 1])

        # Even if this throws an error, the file saving is alrerady done
        try:
            # Prepare variables for the return
            self.optimized_recovery_ratio = max(
                self.parameter_space['recovery_ratio'])

            mask_optimize = self.parameter_space['recovery_ratio'] == self.optimized_recovery_ratio
            self.optimized_eps = float(
                self.parameter_space['eps'][mask_optimize].values)
            self.optimized_minpts = float(
                self.parameter_space['minpts'][mask_optimize].values)

            print("Saving the labels for the optimized method...")
            self.labels_dbscan = DBSCAN(eps=self.optimized_eps,
                min_samples=self.optimized_minpts).fit(tSNEData_).labels_
            self.ratio_labels = self.find_binaries_ratio_dbscan(self.labels_dbscan)

        except:
            print("There was an error calcualting the optimized parameter: to be done manually")

    def get_variables_from_imported_data(self):
        best_dbscan_mode = self.parameter_space[self.parameter_space['recovery_ratio']
                                                == np.max(self.parameter_space['recovery_ratio'])]

        self.labels_dbscan = DBSCAN(eps=best_dbscan_mode['eps'].values[0],
         min_samples=best_dbscan_mode['minpts'].values[0]).fit(np.array((self.normalized_tsne_x,
         self.normalized_tsne_y)).T).labels_
        self.ratio_labels = self.find_binaries_ratio_dbscan(
            self.labels_dbscan)

        self.optimized_eps = best_dbscan_mode['eps'].values[0]
        self.optimized_minpts = best_dbscan_mode['minpts'].values[0]
        self.optimized_recovery_ratio = best_dbscan_mode['recovery_ratio'].values[0]

    def plot_parameter_space(self):
        """ Plots a 2-D map of the explore DBSCAN's parameter space. """
        self.get_variables_from_imported_data()
        # x = eps, y = min samples
        x, y = np.meshgrid(np.linspace(self.min_eps, self.max_eps, self.iterations),
                           np.linspace(self.min_minpts,
                                       self.max_minpts,
                                       self.iterations))

        z = (self.parameter_space['recovery_ratio'].values).reshape(x.shape)

        mask_recovery = self.parameter_space['recovery_ratio'] == np.max(
            self.parameter_space['recovery_ratio'])

        fig, ax = plt.subplots(2, 2, figsize=[20, 10])
        plt.suptitle(('DBSCAN parameter space: {} iterations in {} range, perplexity' +
                      ' {} and SNR of {}. Mode eps = {}, min pts = {} and ratio {}').format(
                      self.iterations, self.spectral_ranges, self.perplexity, self.SNR,
                      np.round(self.optimized_eps,2), np.round(self.optimized_minpts,3),
                      self.ratio))

        max_recovery_per_epsilon = [np.max(z[i]) for i in range(self.iterations)]
        ax[0, 0].plot(y[:, 0], max_recovery_per_epsilon, lw=1.25, alpha=0.85, c='k')
        ax[0, 0].set_xlabel('min pts')
        ax[0, 0].set_ylabel('Recovery')
        ax[0, 0].grid()

        max_recovery_per_minpts = [np.max(z[:, i]) for i in range(self.iterations)]
        ax[1, 1].plot(x[0], max_recovery_per_minpts, lw=1.25, alpha=0.85, c='k')
        ax[1, 1].set_xlabel('epsilon')
        ax[1, 1].set_ylabel('Recovery')
        ax[1, 1].grid()

        contour = ax[1, 0].contourf(y, x, z.T, 75, cmap='plasma')
        ax[1, 0].plot(self.optimized_minpts, self.optimized_eps,
                         marker='x', c='k', alpha=0.85)


        # cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        # cb = plt.colorbar(ax1, cax = cbaxes)
        plt.colorbar(contour, ax=ax[1,0])

        ax[1, 0].set_ylabel('epsilon')
        ax[1, 0].set_xlabel('min pts')
        fig.delaxes(ax[0, 1])  # Removes the extra plot

        niceplot(fig)

        if self.save_plots == True:
            plt.savefig(self.save_plots_dir + ('DBSCAN_parameter_space_range_{}_perplexity_{}'
            + '_SNRof{}_iterations_{}_ratio_{}.png').format(self.spectral_ranges,
            self.perplexity, self.SNR, self.iterations,self.ratio), dpi=150)

    def plot_tsne_maps(self):
        """ Plots a selection of the t-SNE maps, parameter file needs
        to be input as well. """
        self.get_variables_from_imported_data()

        mask_noise = self.labels_dbscan == -1
        mask_cluster = self.labels_dbscan != -1
        mask_singles = self.real_labels == 0
        mask_binaries = self.real_labels == 1
        mask_recovered_binaries = self.ratio_labels == 1
        mask_non_recovered_binaries = self.ratio_labels == 0

        matplotlib.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots(2, 3, figsize=[30, 20], tight_layout=True)
        # plt.suptitle((r't-SNE analysis of {} single and {} binary stars' +
        #     ', with perplexity {} in the {} spectral' +
        #     'range, SNR = {} and ratio {}').format(len(self.tsne_x[mask_singles]),
        #     len(self.tsne_x[mask_binaries]), self.perplexity,
        #     self.spectral_ranges, self.SNR, self.ratio))
        ax[0, 0].set_title(r't-SNE map')
        ax[0, 0].scatter(self.tsne_x[mask_singles],
                         self.tsne_y[mask_singles], c='grey', s=2, label='Single')
        ax[0, 0].scatter(self.tsne_x[mask_binaries],
                         self.tsne_y[mask_binaries], c='red', s=2, label='Binary')

        legend1 = ax[0, 0].legend(loc='lower right')
        legend1.legendHandles[0]._sizes = [30]
        legend1.legendHandles[1]._sizes = [30]

        ax[0, 1].set_title((r'Recovered binary'+
        ' systems: {} of {}').format(len(self.tsne_x[mask_recovered_binaries]),
        len(self.real_labels[self.real_labels == 1])))
        ax[0, 1].scatter(self.tsne_x[mask_non_recovered_binaries],
                         self.tsne_y[mask_non_recovered_binaries],
                         s=2, marker='.', c='grey', label='Noise')
        ax[0, 1].scatter(self.tsne_x[mask_recovered_binaries],
                         self.tsne_y[mask_recovered_binaries], s=2, marker='.',
                         c=self.ratio_labels[mask_recovered_binaries],
                         label='DBSCAN Clusters')

        legend2 = ax[0, 1].legend(loc='lower right')
        legend2.legendHandles[0]._sizes = [30]

        # ax[0, 2].set_title('$L_{B} \, / \, L_{A}$')
        ax[0, 2].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                         cmap=self.colormap, c='grey', s=2, label='Single')
        a = ax[0, 2].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                             c=self.stellar_parameters['lum ratio'][mask_binaries], s=2,
                             cmap=self.colormap, label='Single')
        cbar_1 = plt.colorbar(a, ax=ax[0, 2])
        cbar_1.ax.set_ylabel(r'$L_{B} \, / \, L_{A}$',  labelpad=20)

        # ax[1, 0].set_title('T$_{eff} \, \, [K]$ primary')
        b = ax[1, 0].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                             c=self.stellar_parameters['teff_A'][mask_singles],
                             cmap=self.colormap,
                             s=2, label='Single')
        ax[1, 0].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                         c=self.stellar_parameters['teff_A'][mask_binaries],
                         cmap=self.colormap,
                         s=2, label='Single')
        cbar_2 = plt.colorbar(b, ax=ax[1, 0])
        cbar_2.ax.set_ylabel(r'T$_{eff} \, \, [K]$ primary',  labelpad=20)

        # ax[1, 1].set_title('$log \, g \, \, [dex]$ primary')
        c = ax[1, 1].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                             c=self.stellar_parameters['logg_A'][mask_singles],
                             cmap=self.colormap,
                             s=2, label='Single')
        ax[1, 1].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                         c=self.stellar_parameters['logg_A'][mask_binaries],
                         cmap=self.colormap,
                         s=2, label='Single')
        cbar_3 = plt.colorbar(c, ax=ax[1, 1])
        cbar_3.ax.set_ylabel(r'$log \, g \, \, [dex]$ primary',  labelpad=20)

        # ax[1, 2].set_title('$|\Delta v _{rad}| \, \, [m/s] $')
        ax[1, 2].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                         c='grey', s=2, label='Single')
        d = ax[1, 2].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                             c=self.stellar_parameters['rad_vel'][mask_binaries],
                             cmap=self.colormap,
                             s=2, label='Single')
        cbar_4 = plt.colorbar(d, ax=ax[1, 2])
        cbar_4.ax.set_ylabel(r'$|\Delta v _{rad}| \, \, [m/s] $',  labelpad=20)

        niceplot(fig)

        if self.save_plots == True:
            plt.savefig((self.save_plots_dir + ('tSNEmaps+DBSCAN_range_{}'+
            '_perplexity_{}_SNRof{}_iterations_{}_ratio_{}.png')).format(self.spectral_ranges,
            self.perplexity, self.SNR, self.iterations, self.ratio), dpi=200)

    def plot_tsne_maps_raw(self):
        """ Plots a selection of the t-SNE maps, parameter file needs
        to be input as well. """
        self.get_variables_from_imported_data()

        mask_noise = self.labels_dbscan == -1
        mask_cluster = self.labels_dbscan != -1
        mask_singles = self.real_labels == 0
        mask_binaries = self.real_labels == 1
        mask_recovered_binaries = self.ratio_labels == 1
        mask_non_recovered_binaries = self.ratio_labels == 0

        print(len(self.tsne_x[mask_singles]), len(self.tsne_x[mask_binaries]))
        matplotlib.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots(1, 2, figsize=[30, 15], tight_layout=True)

        ax[0].scatter(self.tsne_x[mask_singles],
                         self.tsne_y[mask_singles], c='grey', s=2, label='Single')
        ax[0].scatter(self.tsne_x[mask_binaries],
                         self.tsne_y[mask_binaries], c='red', s=2, label='Binary')

        legend1 = ax[0].legend(loc='lower right')
        legend1.legendHandles[0]._sizes = [30]
        legend1.legendHandles[1]._sizes = [30]

        ax[1].set_title((r'Recovered binary'+
        ' systems: {} of {}').format(len(self.tsne_x[mask_recovered_binaries]),
        len(self.real_labels[self.real_labels == 1])))
        ax[1].scatter(self.tsne_x[mask_non_recovered_binaries],
                         self.tsne_y[mask_non_recovered_binaries],
                         s=2, c='grey', label='Noise')
        ax[1].scatter(self.tsne_x[mask_recovered_binaries],
                         self.tsne_y[mask_recovered_binaries], s=2,
                         c=self.ratio_labels[mask_recovered_binaries],
                         label='DBSCAN Clusters')

        legend2 = ax[1].legend(loc='lower right')
        legend2.legendHandles[0]._sizes = [30]

        niceplot(fig, 18)

        if self.save_plots == True:
            plt.savefig((self.save_plots_dir + ('tSNEmaps_raw_range_{}'+
            '_perplexity_{}_SNRof{}_iterations_{}_ratio_{}.png')).format(self.spectral_ranges,
            self.perplexity, self.SNR, self.iterations, self.ratio), dpi=150)

    def plot_tsne_maps_stellar_parameters(self):
        """ Plots a selection of the t-SNE maps, parameter file needs
        to be input as well. """
        self.get_variables_from_imported_data()

        mask_noise = self.labels_dbscan == -1
        mask_cluster = self.labels_dbscan != -1
        mask_singles = self.real_labels == 0
        mask_binaries = self.real_labels == 1
        mask_recovered_binaries = self.ratio_labels == 1
        mask_non_recovered_binaries = self.ratio_labels == 0

        matplotlib.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots(2, 2, figsize=[25, 20], tight_layout=True)
        ax[0, 0].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                         cmap=self.colormap, c='grey', s=2, label='Single')
        a = ax[0, 0].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                             c=self.stellar_parameters['lum ratio'][mask_binaries], s=2,
                             cmap=self.colormap, label='Single')
        cbar_1 = plt.colorbar(a, ax=ax[0, 0])
        cbar_1.ax.set_ylabel(r'$L_{B} \, / \, L_{A}$',  labelpad=20)

        # ax[1, 0].set_title('T$_{eff} \, \, [K]$ primary')
        b = ax[1, 0].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                             c=self.stellar_parameters['teff_A'][mask_singles],
                             cmap=self.colormap,
                             s=2, label='Single')
        ax[1, 0].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                         c=self.stellar_parameters['teff_A'][mask_binaries],
                         cmap=self.colormap,
                         s=2, label='Single')
        cbar_2 = plt.colorbar(b, ax=ax[1, 0])
        cbar_2.ax.set_ylabel(r'T$_{eff} \, \, [K]$ primary',  labelpad=20)

        # ax[1, 1].set_title('$log \, g \, \, [dex]$ primary')
        c = ax[1, 1].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                             c=self.stellar_parameters['logg_A'][mask_singles],
                             cmap=self.colormap,
                             s=2, label='Single')
        ax[1, 1].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                         c=self.stellar_parameters['logg_A'][mask_binaries],
                         cmap=self.colormap,
                         s=2, label='Single')
        cbar_3 = plt.colorbar(c, ax=ax[1, 1])
        cbar_3.ax.set_ylabel(r'$log \, g \, \, [dex]$ primary',  labelpad=20)

        # ax[1, 2].set_title('$|\Delta v _{rad}| \, \, [m/s] $')
        ax[0, 1].scatter(self.tsne_x[mask_singles], self.tsne_y[mask_singles],
                         c='grey', s=2, label='Single')
        d = ax[0, 1].scatter(self.tsne_x[mask_binaries], self.tsne_y[mask_binaries],
                             c=self.stellar_parameters['rad_vel'][mask_binaries],
                             cmap=self.colormap,
                             s=2, label='Single')
        cbar_4 = plt.colorbar(d, ax=ax[0, 1])
        cbar_4.ax.set_ylabel(r'$|\Delta v _{rad}| \, \, [m/s] $',  labelpad=20)

        niceplot(fig)

        if self.save_plots == True:
            plt.savefig((self.save_plots_dir + ('tSNEmaps_stellarparamters_range_{}'+
            '_perplexity_{}_SNRof{}_iterations_{}_ratio_{}.png')).format(self.spectral_ranges,
            self.perplexity, self.SNR, self.iterations, self.ratio), dpi=150)

    def plot_histograms(self):
        self.normalize_data_tSNE()
        self.get_variables_from_imported_data()

        retrieved_binaries = self.stellar_parameters[self.ratio_labels == 1]
        # Get the binaries that were not properly retrieved
        non_retrievedBinaries = self.stellar_parameters[(
            self.stellar_parameters['binarity'] == 1) & (self.ratio_labels == 0)]
        all_Binaries = self.stellar_parameters[self.stellar_parameters['binarity'] == 1]

        fig, ax = plt.subplots(2, 4, figsize=(30, 18))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fontsize = 26

        ax[0, 0].set_xlabel(r'T$_{{\mathrm{{eff}}}} \, \, [K] $ (A)',
         fontsize=fontsize)
        ax[0, 0].hist(retrieved_binaries['teff_A'],  density=False,
                           color='grey', alpha=0.5, bins=25, label='Recovered binaries')
        ax[0, 0].hist(non_retrievedBinaries['teff_A'],
                           density=False, histtype='step',
                           color='black', bins=25, label='Non-recovered binaries')
        ax[0, 0].hist(all_Binaries['teff_A'],  density=False,
                           histtype='step', color='blue',
                           bins=25, label='Whole binary sample')

        ax[1, 0].set_xlabel(r'T$_{{\mathrm{{eff}}}} \, \, [K] $ (B)',
        fontsize=fontsize)
        ax[1, 0].hist(retrieved_binaries['teff_B'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[1, 0].hist(non_retrievedBinaries['teff_B'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[1, 0].hist(all_Binaries['teff_B'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[0, 1].set_xlabel(r'[Fe / H] [dex] (A)', fontsize=fontsize)
        ax[0, 1].hist(retrieved_binaries['feh_A'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[0, 1].hist(non_retrievedBinaries['feh_A'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[0, 1].hist(all_Binaries['feh_A'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[1, 1].set_xlabel('[Fe / H] [dex](B)', fontsize=fontsize)
        ax[1, 1].hist(retrieved_binaries['feh_B'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[1, 1].hist(non_retrievedBinaries['feh_B'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[1, 1].hist(all_Binaries['feh_B'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[0, 2].set_xlabel('$log \, g \, \, [dex]$ (A)', fontsize=fontsize)
        ax[0, 2].hist(retrieved_binaries['logg_A'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[0, 2].hist(non_retrievedBinaries['logg_A'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[0, 2].hist(all_Binaries['logg_A'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[1, 2].set_xlabel(r'$log \, g  \, \, [dex]$ (B)', fontsize=fontsize)
        ax[1, 2].hist(retrieved_binaries['logg_B'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[1, 2].hist(non_retrievedBinaries['logg_B'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[1, 2].hist(all_Binaries['logg_B'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[0, 3].set_xlabel(r'$L_{B} \, / \, L_{A}$', fontsize=fontsize)
        ax[0, 3].hist(retrieved_binaries['lum ratio'],  density=False,
                           color='grey', alpha=0.5, bins=25)
        ax[0, 3].hist(non_retrievedBinaries['lum ratio'],  density=False,
                      histtype='step', color='black', bins=25)
        ax[0, 3].hist(all_Binaries['lum ratio'],  density=False,
                      histtype='step', color='blue', bins=25)

        ax[1, 3].set_xlabel(r'$|\Delta v _{rad}| \, \, [km/s] $', fontsize=fontsize)
        ax[1, 3].hist(np.abs(retrieved_binaries['rad_vel']),
                           color='grey', alpha=0.5, bins=25)
        ax[1, 3].hist(np.abs(non_retrievedBinaries['rad_vel']),
                      density=False, histtype='step', color='black', bins=25)
        ax[1, 3].hist(np.abs(all_Binaries['rad_vel']),  density=False,
                      histtype='step', color='blue', bins=25)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        ax[1, 2].legend(handles, labels, edgecolor='inherit',
        fontsize=20, loc='upper left')
        # plt.subplots_adjust(bottom=0.1)

        matplotlib.rcParams.update({'font.size': fontsize})
        niceplot(fig, labelsize = fontsize, tight_layout=True)
        matplotlib.rcParams.update({'font.size': 26 })

        if self.save_plots == True:
            # plt.savefig(self.save_plots_dir + ('DBSCAN_histograms'+
            # '_range_{}_perplexity_{}_SNRof{}_iterations_{}_ratio_{}.png').format(self.spectral_ranges,
            # self.perplexity, self.SNR, self.iterations, self.ratio), dpi=300)
            plt.savefig(('DBSCAN_histograms'+
            '_range_{}_perplexity_{}_SNRof{}_iterations_{}_ratio_{}.png').format(self.spectral_ranges,
            self.perplexity, self.SNR, self.iterations, self.ratio), dpi=300)
