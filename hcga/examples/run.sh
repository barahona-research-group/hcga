#!/bin/bash

export OMP_NUM_THREADS=1

hcga extract_features $1 -m fast -n 4 -sl advanced --timeout 10  #--connected #--runtimes