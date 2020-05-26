import sys
import directories
import numpy as np
import os
import pandas as pd
from fitsne import FItSNE

#  ___ Directory paths _________________________________________________________ Directory paths
dirs = directories.directories()

#  ___ Variables _______________________________________________________________ Variables
# These are neede for the naming scheme and for the run itself
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]
SNR = float(sys.argv[3])
PERPLEXITY = int(sys.argv[4])
NUMBER_RANDOMNESS = int(sys.argv[5])

FILENAME_TSNE_RESULT = (dirs.tsne_results +
 'tSNE_results_range_{}_perplexity_{}_SNRof{}.npy'.format(SELECTED_RANGES, PERPLEXITY, SNR))

if os.path.isfile(FILENAME_TSNE_RESULT) == True:
    print('File exists already, skipping this step..')
    exit()

else:
    pass

# _____ Load data ______________________________________________________________ Load data
# Load the data
sample_tsne_no_noise = np.load(
    dirs.samples_to_tsne + 'sample_to_tsne_range_{}.npy'.format(SELECTED_RANGES))

# Noise is added at each iteration then
sample_tsne = sample_tsne_no_noise + np.random.normal(0, 1/float(SNR),
 sample_tsne_no_noise.shape)

print("Added noise to the t-SNE sample with a SNR of {}".format(int(SNR)))

del sample_tsne_no_noise

# _____ t-SNE __________________________________________________________________ t-SNE
print('='*40)
print('--> Starting t-SNE analysis for perplexity {}'.format(PERPLEXITY))
# Run t-SNE for each one of the above perplexities
tSNEd_fluxes = FItSNE(X=sample_tsne, perplexity=PERPLEXITY)
# Save the data
np.save(dirs.tsne_results + 'tSNE_results_range_{}_perplexity_{}_SNRof{}_ramdomnesscheck_{}'.format(
    SELECTED_RANGES, PERPLEXITY, int(SNR), NUMBER_RANDOMNESS), tSNEd_fluxes)

print('='*40)
