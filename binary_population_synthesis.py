import scaling_relations
import galah
import directories
import os
import sys
import socket
import numpy as np
import pandas as pd
computer_name = socket.gethostname()

# ---- Control Panel ----------------------------------------------------------- Control Panel
# Initialize directories
dirs = directories.directories()

# ------------------------------------------------------------------------------

def get_mass_ratio(low_limit=0.3):
    # Exponent 3.25 from the paper Hogeven 91.
    # The beta distribution with b=1 acts like a power law
    # dist = np.random.beta(1.1, 0.95, 100)
    # mask = dist > low_limit
    # q = dist[mask][0]

    # Mass ratio distribution from Duchene and Kraus 13, power law
    # with 0.3 in the exponent
    return np.random.power(1.3)

def generate_binarySystems(numberOfBinaries, stars_run):

    binaryStars = []
    ids_used_stars = []
    teffRange_primary = 75  # Plus / minus Kelvin value for the crossmatching
    fe_hRange_secondary = 0.025 # dex
    teffRange_secondary = 75 # plus/minus Kelvin

    while len(binaryStars) < numberOfBinaries:
        # --- To be done in every iteration ------------------------------------

        massRatio = get_mass_ratio()
        # stellarMass_primary = 0
        # while stellarMass_primary == 0:
        # stellarMass_primary = np.random.exponential(0.3)

        stellarMass_primary = np.random.exponential(0.3)
        stellarMass_secondary = massRatio * stellarMass_primary

        # Use the new scaling relations:
        temperature_PrimaryStar = scaling_relations.classic_MTR(
            stellarMass_primary)
        temperature_SecondaryStar = scaling_relations.classic_MTR(
            stellarMass_secondary)

        luminosityRatio = scaling_relations.classic_MLR(stellarMass_secondary) / \
            scaling_relations.classic_MLR(stellarMass_primary)

        # Main loop
        try: # Tries to find primary

            # Define the mask to obtain the GALAH stars
            mask_candidatesPrimary = (abs(temperature_PrimaryStar - stars_run['teff']) <= teffRange_primary)

            # Find a candidate the matches the defined conditions
            primaryStar_Candidate = np.random.permutation(stars_run[mask_candidatesPrimary])[0]

            if primaryStar_Candidate['sobject_id'] in ids_used_stars:
                print('Primary star skipped, was used and on the list already.')
                pass

            try: # Tries to find secondary
                # Secondary star should be of almost the same metallicity, have lower temperature than the
                # primary star and higher logg
                mask_candidatesSecondary =  ((abs(temperature_SecondaryStar - stars_run['teff']) <= teffRange_secondary) &
                                            (primaryStar_Candidate['teff'] > stars_run['teff']) &
                                            (primaryStar_Candidate['logg'] < stars_run['logg']) &
                                            (abs(primaryStar_Candidate['fe_h'] - stars_run['fe_h']) <= fe_hRange_secondary))

                # The permutation is to pick the a random candidate each time
                secondaryStar_Candidate = np.random.permutation(stars_run[mask_candidatesSecondary])[0]

                # Check if the secondary has already been used
                if secondaryStar_Candidate['sobject_id'] in ids_used_stars:
                    print('Secondary star skipped, was used and on the list already. <--')
                    pass

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
                                     'id_A': primaryStar_Candidate['sobject_id'],
                                     'id_B': secondaryStar_Candidate['sobject_id']}

                ids_used_stars.append(primaryStar_Candidate['sobject_id'])
                ids_used_stars.append(secondaryStar_Candidate['sobject_id'])
                binaryStars.append(binarySystem_data)
                print("System:", len(binaryStars), ",Amount of used stars:", len(ids_used_stars))

            except IndexError:
                # When no secondary star with the given parameters could be found.
                pass

        except IndexError:
            # Index error happens when no primary stars has been found.
            pass

    binaryStars = pd.DataFrame(binaryStars)

    return binaryStars

if __name__ == "__main__":
    # ---- Import/Filter Data ---------------------------
    # Gets the data from GALAH for the stellar synthesis
    galah_data = galah.GALAH_survey()
    galah_data.get_stars_run()

    # ---- Run the population simulation ----------------
    numberOfBinaries = 5000
    binaryStars_generated = generate_binarySystems(
        numberOfBinaries, galah_data.stars_run)

    binaryStars_generated.to_csv(dirs.data +
     ('BinaryPopulation_{}_classicrelations_allStarsDifferent'
     '_DucheneKraus13_q_IMF_exponential0,3.csv').format(numberOfBinaries), index=False)
