#!/bin/bash

# Runs the whole analysis for the project
# --> Declare the wavelength ranges for the analysis
declare -a lowerBound=($(seq 450 25 875))
declare -a upperBound=($(seq 475 25 900))

# --> Declare the signal-to-noise values to investigate
declare -a SNR_values=(10 25 50 100)

# --> Declare the perplexity values
declare -a perplexities=(5 15 30 100)

# --> This creates the files to use in the analysis
for ((i=0; i<${#lowerBound[@]}; ++i)); do
  for ((j=0; j<${#SNR_values[@]}; ++j)); do
    for ((k=0; k<${#perplexities[@]}; ++k)); do
      printf "BASH SCRIPT: Performing DBSCAN analysis to the results of t-SNE with a perplexity %f on sample SNR = %f within the range %s - %s nm \n" "${perplexities[k]}" "${SNR_values[j]}" "${lowerBound[i]}" "${upperBound[i]}"
      nohup python -u dbscan_analysis.py  ${lowerBound[i]} ${upperBound[i]} ${SNR_values[j]} ${perplexities[k]} &
      wait
    done
  done
done
