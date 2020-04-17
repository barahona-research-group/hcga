#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel
hcga extract_features $1 -m fast -n 10 -sl advanced # --runtimes #--no-norm #  
