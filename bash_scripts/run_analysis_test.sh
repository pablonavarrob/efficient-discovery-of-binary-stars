#!/bin/bash

# Runs the whole analysis for the project
# --> Declare the wavelength ranges for the analysis
declare -a lowerBound=($(seq 450 25 600))
declare -a upperBound=($(seq 475 25 625))

# --> Declare the signal-to-noise values to investigate
declare -a SNR_values=(10 100)

# --> Declare the perplexity values
declare -a perplexities=(5 30)
for ((i=0; i<${#lowerBound[@]}; ++i)); do
  # Read and cut the spectra for the whole analysis
  printf "BASH SCRIPT: Reading and cutting fits files within the range %s - %s nm \n" "${lowerBound[i]}" "${upperBound[i]}"
  nohup python -u read_cut_fits.py ${lowerBound[i]} ${upperBound[i]} &
  wait
  printf "BASH SCRIPT: Creating sample to t-SNE within the range %s - %s nm \n" "${lowerBound[i]}" "${upperBound[i]}"
  nohup python -u prepare_sample_tSNE.py ${lowerBound[i]} ${upperBound[i]} &
  wait

  for ((j=0; j<${#SNR_values[@]}; ++j)); do
        for ((k=0; k<${#perplexities[@]}; ++k)); do
      # Perform the analysis on the cut files
      printf "BASH SCRIPT: Running t-SNE with perplexity %f on sample SNR = %f within the range %s - %s nm \n" "${perplexities[k]}" "${SNR_values[j]}" "${lowerBound[i]}" "${upperBound[i]}"
      nohup python -u tsne.py  ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]} ${perplexities[k]} &
      wait

    done
  done
done
