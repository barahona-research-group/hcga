#!/bin/bash

export OMP_NUM_THREADS=1

hcga extract_features $1 -m fast -n 2 -sl advanced --node-feat # --runtimes
