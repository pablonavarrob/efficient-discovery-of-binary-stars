# import matplotlib.pyplot as plt
import scaling_relations
import galah
import directories
import os
import sys
import socket
import numpy as np
# import matplotlib
import pandas as pd
computer_name = socket.gethostname()

np.random.seed(1)

# ---- Control Panel ------------------------------------------------------------------------------- Control Panel
numberSynthStars = 100000
# Initialize directories
dirs = directories.directories()

# ---------------------------------------------------------------------------------------------------
# def get_mass_ratio(low_limit=0.35):
#     q = []
#     while len(q) == 0:
#         # 3.25 from the paper Hogeven 91.
#         dist = np.random.power(3.25, 1)
#         mask = dist > low_limit
#         q = dist[mask]
#     return float(q)


def generate_binarySystems(numberOfBinaries, stars_run):

    binaryStars = []

    while len(binaryStars) < numberOfBinaries:
        # --- To be done in every iteration --------------------------------------------------------
        # Sample the mass ratio AND the mass of the primary star - with that calculate mass of the secondary
        massRatio = np.random.uniform(0.35, 1, 1)

        # Should the mass of the primary star be sampled from the IMF? It might be more realistic
        # Randomly sample the mass of the primary star.

        # Rnages for primar mass: between 3785 and 7529 K
        stellarMass_primary = float(np.random.uniform(0.5, 2.5))
        stellarMass_secondary = massRatio * stellarMass_primary

        # Use the new scaling relations:
        temperature_PrimaryStar = scaling_relations.mass_teff_relation(
            stellarMass_primary)
        logg_PrimaryStar = scaling_relations.get_logg_star(stellarMass_primary)
        temperature_SecondaryStar = scaling_relations.mass_teff_relation(
            stellarMass_secondary)
        logg_SecondaryStar = scaling_relations.get_logg_star(
            stellarMass_secondary)
        luminosityRatio = scaling_relations.mass_luminosity_relation(stellarMass_secondary) / \
            scaling_relations.mass_luminosity_relation(stellarMass_primary)

        try:

            # --- Define masks and ranges for the parameters -------------------------------------------
            # Get stars similar to the sampled one: define ranges for each quantity
            temperatureRange = 0.005  # Plus / minus value for the crossmatching
            loggRange = 0.1  # logg has large innacuracies in the measurements

            # Define the mask to obtain the GALAH stars within certain ranges of the sampled
            mask_candidatesPrimary = ((temperature_PrimaryStar - temperature_PrimaryStar * temperatureRange
                                       < stars_run['teff']) &
                                      (temperature_PrimaryStar + temperature_PrimaryStar * temperatureRange
                                       > stars_run['teff']) &
                                      (logg_PrimaryStar - logg_PrimaryStar * loggRange < stars_run['logg']) &
                                      (logg_PrimaryStar + logg_PrimaryStar * loggRange > stars_run['logg']))

            # We do the above to find a star that could serve as primary - aka, we try to find within the GALAH
            # dataset a star that matches the sampled/calculated parameters.
            primaryStar_Candidate = np.random.permutation(
                stars_run[mask_candidatesPrimary])[0]

        except IndexError:
            # Index error happens when no primary stars has been found.
            pass

        try:
            # ---Find candidates for the secondary star ------------------------------------------------
            # As said: a secondary star should be of almost the same metallicity, have lower temperature than the
            # primary star but higher luminosityl
            fe_hRange = 0.01
            teffRange_secondary = 0.01

            # Mask for negative metallicities
            if primaryStar_Candidate['fe_h'] < 0:
                mask_candidatesSecondary = ((primaryStar_Candidate['teff'] > stars_run['teff']) &
                                            (primaryStar_Candidate['logg'] < stars_run['logg']) &
                                            (primaryStar_Candidate['fe_h'] + primaryStar_Candidate['fe_h'] * fe_hRange <
                                             stars_run['fe_h']) &
                                            (primaryStar_Candidate['fe_h'] - primaryStar_Candidate['fe_h'] * fe_hRange >
                                             stars_run['fe_h']) &
                                            (temperature_SecondaryStar + temperature_SecondaryStar * teffRange_secondary >
                                             stars_run['teff']) &
                                            (temperature_SecondaryStar - temperature_SecondaryStar * teffRange_secondary <
                                             stars_run['teff']))

            else:                                 # Mask for positive metallicities
                mask_candidatesSecondary = ((primaryStar_Candidate['teff'] > stars_run['teff']) &
                                            (primaryStar_Candidate['logg'] < stars_run['logg']) &
                                            (primaryStar_Candidate['fe_h'] + primaryStar_Candidate['fe_h'] * fe_hRange >
                                             stars_run['fe_h']) &
                                            (primaryStar_Candidate['fe_h'] - primaryStar_Candidate['fe_h'] * fe_hRange <
                                             stars_run['fe_h']) &
                                            (temperature_SecondaryStar + temperature_SecondaryStar * teffRange_secondary >
                                             stars_run['teff']) &
                                            (temperature_SecondaryStar - temperature_SecondaryStar * teffRange_secondary <
                                             stars_run['teff']))

            secondaryStar_Candidate = np.random.permutation(
                stars_run[mask_candidatesSecondary])[0]

            # Create a dictionary to append the data to the above dataframe
            # The appended stellar parameters is the one from the GALAH stars.
            binarySystem_data = {'mass_A': stellarMass_primary,
                                 'teff_A': primaryStar_Candidate['teff'],
                                 'logg_A': primaryStar_Candidate['logg'],
                                 'feh_A': primaryStar_Candidate['fe_h'],
                                 'mass_B': stellarMass_secondary,
                                 'teff_B': secondaryStar_Candidate['teff'],
                                 'logg_B': secondaryStar_Candidate['logg'],
                                 'feh_B': secondaryStar_Candidate['fe_h'],
                                 'mass ratio': massRatio,
                                 'lum ratio': luminosityRatio,
                                 'comp_teff_A': temperature_PrimaryStar,
                                 'comp_teff_B': temperature_SecondaryStar,
                                 'comp_logg_A': logg_PrimaryStar,
                                 'comp_logg_B': logg_SecondaryStar,
                                 'id_A': primaryStar_Candidate['sobject_id'],
                                 'id_B': secondaryStar_Candidate['sobject_id']}

            # Append the dictionary to the empty list above: convert later into dataframe for easier handling
            binaryStars.append(binarySystem_data)
            print(len(binaryStars))

        except:
            # When no secondary star with the given parameters could be found.
            IndexError

    # Convert to dataframe for easier handling
    binaryStars = pd.DataFrame(binaryStars)

    return binaryStars


if __name__ == "__main__":
    # ---- Import/Filter Data ---------------------------
    # Gets the data from GALAH for the stellar synthesis
    galah_data = galah.GALAH_survey()
    galah_data.get_stars_run()

    # ---- Run the population simulation ----------------
    numberOfBinaries = 1000
    binaryStars_generated = generate_binarySystems(
        numberOfBinaries, galah_data.stars_run)
    binaryStars_generated.to_csv(dirs.data +
                                 'BinaryPopulation_{}_NEWRUN.csv'.format(numberOfBinaries), index=False)

# ___________________________________________________________________________________________________ Plots

#binaryStars_generated = pd.read_csv('/stellarSynthesis/Binaries_GALAH_v2.csv')
#
# plt.figure(figsize=[10,10])
#plt.title('$T_{eff}$ of primary and secondary star of the synthetic population')
#plt.hist(binaryStars_generated['teff_A'], histtype='step', bins=20, label='Primary')
#plt.hist(binaryStars_generated['teff_B'], histtype='step', bins=20, label='Secondary')
# plt.legend()
# plt.show()
#plt.savefig('BinaryPopulation_temperatureDist.png', dpi=150)
#
# plt.figure(figsize=[10,10])
#plt.title('$log \, g$ of primary and secondary star of the synthetic population')
#plt.hist(binaryStars_generated['logg_A'], histtype='step', bins=20, label='Primary')
#plt.hist(binaryStars_generated['logg_B'], histtype='step', bins=20, label='Secondary')
# plt.legend()
# plt.show()
#plt.savefig('BinaryPopulation_loggsDist.png', dpi=150)
#
# plt.figure(figsize=[10,10])
#plt.title('[Fe/H] of primary and secondary star of the synthetic population')
#plt.hist(binaryStars_generated['feh_A'], histtype='step', bins=20, label='Primary')
#plt.hist(binaryStars_generated['feh_B'], histtype='step', bins=20, label='Secondary')
# plt.legend()
# plt.show()
#plt.savefig('BinaryPopulation_fehsDist.png', dpi=150)
#
# plt.figure(figsize=[10,10])
#plt.title('Mass ratio between primary and secondary star of the synthetic population')
#plt.hist(binaryStars_generated['mass ratio'], histtype='step', bins=20)
# plt.show()
#plt.savefig('BinaryPopulation_massRatios.png', dpi=150)
#
# plt.figure(figsize=[10,10])
#plt.title('Luminosity ratio between primary and secondary star of the synthetic population')
#plt.hist(binaryStars_generated['lum ratio'], histtype='step', bins=20)
# plt.show()
#plt.savefig('BinaryPopulation_luminosityRatios.png', dpi=150)
#
# Plot metallicities: primary on X axis and secondary on Y axis + 1:1 line to see difference.
# plt.figure(figsize=[10,10])
#plt.title('[Fe/H] of the primary against [Fe/H] of the secondary')
#plt.scatter(binaryStars_generated['feh_A'], binaryStars_generated['feh_B'], s=2, c='orange', alpha=0.35)
#plt.plot(np.linspace(-1, 0.7, 5000), np.linspace(-1, 0.7, 5000), lw = 0.75, c='k', alpha=1, ls='-.')
#plt.xlabel('Metallicity Primary Star')
#plt.ylabel('Metallicity Secondary Star')
#plt.savefig('MetallicitiesBinaryPopulation.png', dpi=150)
# plt.show()


# plt.figure(figsize=[10,10])
#plt.scatter(binaryStars_generated['comp teff A'], binaryStars_generated['teff_A'], s=1, alpha=0.25)
#plt.scatter(binaryStars_generated['comp teff B'], binaryStars_generated['teff_B'], s=1, alpha=0.25)
#plt.xlabel('Calc Teff Primary Star')
#plt.ylabel('Sampled Teff Primary Star')
#
# plt.figure(figsize=[10,10])
#plt.scatter(binaryStars_generated['comp logg A'], binaryStars_generated['logg_A'], s=1, alpha=0.25)
#plt.scatter(binaryStars_generated['comp logg B'], binaryStars_generated['logg_B'], s=1, alpha=0.25)
#plt.xlabel('Calc logg Primary Star')
#plt.ylabel('Sampled logg Primary Star')
