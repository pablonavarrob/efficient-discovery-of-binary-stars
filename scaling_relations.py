# Scaling relations for main sequence stars according to Eker, Z + 18.
# Interrelated MRR and MTR. 6-step linear relation for MLR.
import numpy as np

SB_CONSTANT = 5.670374e-8  # W/m^2 K^4
SB_CONSTANT_CGS = 5.670374e-5  # erg
LUMINOSITY_SUN = 3.827e26  # W
LUMINOSITY_SUN_CGS = 3.839e33  # erg s-1
MASS_SUN = 1.989e30  # kg
MASS_SUN_CGS = 1.989e33  # in CGS units
TEFF_SUN = 5778  # K
RADIUS_SUN = 6.9634e8  # m
RADIUS_SUN_CGS = 6.9634e10  # cm

# Returns luminosity in solar units

def mass_luminosity_relation(mass_star):
    """Calculates the luminosity for a given mass, Eker, Z. et al. 2018.
    Different cuts depending on the mass. ULM, VLM, LM, IM, HM and VHM.
    Masses all given in solar units and return luminosity as well. """

    # Obtain the parameters
    if 0.179 < mass_star <= 0.45:
        alpha = 2.028
        beta = -0.976
    if 0.45 < mass_star <= 0.72:
        alpha = 4.572
        beta = -0.102
    if 0.72 < mass_star <= 1.05:
        alpha = 5.473
        beta = -0.007
    if 1.05 < mass_star <= 2.4:
        alpha = 4.329
        beta = 0.010
    if 2.4 < mass_star <= 7:
        alpha = 3.967
        beta = 0.093
    if 7 < mass_star <= 31:
        alpha = 2.865
        beta = 1.105

    # Calcualate the luminosity from the mass
    log_lum = alpha*np.log10(mass_star) + beta
    luminosity = 10**log_lum

    # Luminosity in solar units
    return luminosity

# Returns either radius or temperature in SI units, which
# need to be converted later

def stefan_boltzman_radius(luminosity, temperature):
    """ To return the radius - for the teff calculations. """
    # Convert to SI units from the solar, input
    luminosity_SI = luminosity*LUMINOSITY_SUN_CGS
    # teff_SI = temperature*teff_sun
    radius = (luminosity_SI/(4*np.pi*SB_CONSTANT_CGS*temperature**4))**(1/2)

    # Returns the radius in SI units
    return radius


def stefan_boltzman_teff(luminosity, radius):
    """ To return teff - for the radius calculations. """
    # Convert to SI units from the solar, input
    luminosity_SI = luminosity*LUMINOSITY_SUN_CGS
    teff = (luminosity_SI/(4*np.pi*SB_CONSTANT_CGS*radius**2))**(1/4)
    # Returns the temperature in SI units
    return teff


def mass_radius_relation(mass):
    # Check masses - and assuming luminosities have previously been computed

    if 0.179 < mass <= 1.5:
        # Return result in SI
        radius = (0.438 * mass**2 + 0.479 * mass + 0.075)*RADIUS_SUN_CGS

    else:
        teff = mass_teff_relation(mass)  # in SI units
        # in solar units, will be converted
        lum = mass_luminosity_relation(mass)
        # From Stefan Boltzman relation, radius in SI units
        radius = stefan_boltzman_radius(lum, teff)
    # Append within the loop

    return radius  # Always in cgs units, cm


def mass_teff_relation(mass):

    if 1.5 < mass <= 31:
        log_teff = (-0.170 * np.log10(mass)**2 +
                    0.888 * np.log10(mass) + 3.671)
        teff = (10**log_teff)

    else:
        radius = mass_radius_relation(mass)
        # in solar units, will be converted
        lum = mass_luminosity_relation(mass)
        # From Stefan Boltzman relation, temperature in SI units, convert to log
        teff = stefan_boltzman_teff(lum, radius)

    return teff  # Also in cgs units, kelvin


def get_logg_star(mass):
    """ Calculates the logarithmic surface gravity for a given stellar mass according to the
         scaling relations. """

    radius = mass_radius_relation(mass)/RADIUS_SUN_CGS
    logg = np.log10(mass) - 2 * np.log10(radius) + 4.437  # The last term is
    # to leave the results in cgs coming from solar units

    return logg  # in CGS

def classic_MLR(mass):
    """ With the usual exponent of 3.5 - to avoid jumps in the mass ranges..?
    Returns in solar units. """
    return mass**4.5

def classic_MTR(mass):
    """ Power law coefficient extracted from the results of Eker, Z. et al. 2015.
    The returned value is in Kelvin. """

    return (mass**0.38)*TEFF_SUN

if __name__ == "__main__":
    range_masses = np.linspace(0.2, 30, 450)
    luminosities = np.array([mass_luminosity_relation(mass)
                             for mass in range_masses])  # in solar lums
    teffs = np.array([mass_teff_relation(mass) for mass in range_masses])
    radii = np.array([mass_radius_relation(mass) for mass in range_masses])
    loggs = np.array([get_logg_star(mass) for mass in range_masses])

    fig, ax = plt.subplots(4, 1, figsize=[10, 15], sharex=True)
    plt.suptitle('Scaling relations Eker, Z. + 18')
    ax[0].plot(np.log10(range_masses), np.log10(luminosities),
               lw=0.75, c='red', alpha=0.85)
    ax[0].set_ylabel('log L/Lsun')
    ax[1].plot(np.log10(range_masses), np.log10(radii/RADIUS_SUN),
               lw=0.75, c='red', alpha=0.85)
    ax[1].set_ylabel('log R/Rsun')
    ax[2].plot(np.log10(range_masses), np.log10(teffs),
               lw=0.75, c='red', alpha=0.85)
    ax[2].set_ylabel('log teff')
    ax[3].plot(np.log10(range_masses), loggs,
               lw=0.75, c='red', alpha=0.85)
    ax[3].set_ylabel('logg')
