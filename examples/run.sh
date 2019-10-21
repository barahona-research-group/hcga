#!/bin/bash

set -e

module purge all
. ~/hcga/bin/activate

export OMP_NUM_THREADS=1

python test.py
