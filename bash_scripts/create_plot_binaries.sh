#!/bin/bash

printf "Generating spectra for singles and binaries in the range 500 to 525 nm"

nohup python -u read_cut_fits.py 500 525 &
wait
nohup python -u prepare_sample_tSNE.py 500 525 25 &
wait
nohup python -u prepare_sample_tSNE.py 500 525 100 &
wait

printf "Generating spectra for singles and binaries in the range 800 to 825 nm"

nohup python -u read_cut_fits.py 800 825 &
wait
nohup python -u prepare_sample_tSNE.py 800 825 25 &
wait
nohup python -u prepare_sample_tSNE.py 800 825 100 &
wait

printf "Done!"
