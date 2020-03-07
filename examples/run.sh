#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel
hcga get_data $1
hcga extract_features $1 -m fast -n 4 -of $1 --norm
