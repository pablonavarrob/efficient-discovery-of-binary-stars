#!/bin/bash

printf "Checking for randomness in the best case scenario"

nohup python -u tsne_randomness_check.py 525 550 100 15 4 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 15 5 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 15 6 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 4 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 5 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 6 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 4 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 5 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 6 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 4 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 5 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 6 &
wait

nohup python -u tsne_randomness_check.py 525 550 100 15 7 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 15 8 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 15 9 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 7 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 8 &
wait
nohup python -u tsne_randomness_check.py 525 550 100 30 9 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 7 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 8 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 15 9 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 7 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 8 &
wait
nohup python -u tsne_randomness_check.py 525 550 500 30 9 &
wait


printf "Checking for randomness in the worst case scenario"


nohup python -u tsne_randomness_check.py 675 700 100 15 4 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 15 5 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 15 6 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 4 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 5 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 6 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 4 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 5 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 6 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 4 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 5 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 6 &
wait

nohup python -u tsne_randomness_check.py 675 700 100 15 7 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 15 8 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 15 9 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 7 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 8 &
wait
nohup python -u tsne_randomness_check.py 675 700 100 30 9 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 7 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 8 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 15 9 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 7 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 8 &
wait
nohup python -u tsne_randomness_check.py 675 700 500 30 9 &
wait

printf "Done!"
