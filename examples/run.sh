#!/bin/bash

export OMP_NUM_THREADS=1

hcga extract_features $1 -m fast -n 80 -sl basic --timeout 10  #--connected #--runtimes
