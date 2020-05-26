#!/bin/bash

# Runs the whole analysis for the project
# --> Declare the wavelength ranges for the analysis
declare -a lowerBound=($(seq 450 25 875))
declare -a upperBound=($(seq 475 25 900))

# --> Declare the signal-to-noise values to investigate
declare -a SNR_values=(10 25 50 100 500)

# --> Declare the perplexity values
declare -a perplexities=(5 15 30 100)

# --> This creates the files to use in the analysis
for ((i=0; i<${#lowerBound[@]}; ++i)); do
  # Read and cut the spectra for the whole analysis
  printf "BASH SCRIPT: Reading and cutting fits files within the range %s - %s nm \n" "${lowerBound[i]}" "${upperBound[i]}"
  nohup python -u read_cut_fits.py ${lowerBound[i]} ${upperBound[i]} &
  wait
  for ((j=0; j<${#SNR_values[@]}; ++j)); do
    # Create the noisy sample
    printf "BASH SCRIPT: Creating samples to t-SNE with SNR = %f  within the range %s - %s nm \n" "${SNR_values[j]}" "${lowerBound[i]}" "${upperBound[i]}"
    # Preparing the t-SNE sample already takes care of the file produced above as it is only needed once to create the sample
    nohup python -u prepare_sample_tSNE.py ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]}
    wait
    for ((k=0; k<${#perplexities[@]}; ++k)); do
      # Perform the analysis on the cut files
      printf "BASH SCRIPT: Running t-SNE with perplexity %f on sample SNR = %f within the range %s - %s nm \n" "${perplexities[k]}" "${SNR_values[j]}" "${lowerBound[i]}" "${upperBound[i]}"
      nohup python -u tsne.py  ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]} ${perplexities[k]} &
      wait
      printf "BASH SCRIPT: Performing DBSCAN analysis to the results of t-SNE with a perplexity %f on sample SNR = %f within the range %s - %s nm \n" "${perplexities[k]}" "${SNR_values[j]}" "${lowerBound[i]}" "${upperBound[i]}"
      nohup python -u dbscan_analysis.py  ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]} ${perplexities[k]} &
      wait
    done
  done
  # Remove the t-SNE sample
  printf "BASH SCRIPT: Deleting the sample file and the cut spectra from the previous analysis"
  nohup python -u delete_files_tsne.py ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]} $"tsne" &
  wait

  # Remove the synthetic single spectra
  printf "BASH SCRIPT: Deleting the sample file and the cut spectra from the previous analysis \n \n \n"
  nohup python -u delete_files_spectra.py ${lowerBound[i]} ${upperBound[i]} $"spectra" &
  wait
done
