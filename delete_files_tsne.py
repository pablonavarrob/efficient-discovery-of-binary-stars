import os
import sys
import directories

#  ___ Variables _______________________________________________________________ Variables
SELECTED_RANGES = [(float(sys.argv[1]), float(sys.argv[2]))]
SNR = int(sys.argv[3])
FILE_TO_REMOVE = sys.argv[4]

#  ___ Directories _____________________________________________________________ Directories
dirs = directories.directories()
filename_tsne_sample = (dirs.samples_to_tsne +
        'sample_to_tsne_range_{}_SNRof{}.npy'.format(SELECTED_RANGES, SNR))


#  ___ Delete files if exist ___________________________________________________ File delete

if FILE_TO_REMOVE == 'tsne':
    if os.path.isfile(filename_tsne_sample) == True:
        print('Deleting {}'.format(filename_tsne_sample))
        os.remove(filename_tsne_sample)
    print('Deleting t-SNE file..')

print('Removed the {} file from the disk'.format(FILE_TO_REMOVE))
print('=' * 40)
