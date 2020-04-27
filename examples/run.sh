#!/bin/bash

export OMP_NUM_THREADS=1

hcga extract_features $1 -m medium -n 12 -sl advanced --timeout 10 #--runtimes
