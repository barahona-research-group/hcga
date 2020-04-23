#!/bin/bash

export OMP_NUM_THREADS=1  # set to one to prevent numpy to run in parallel
hcga get_data $1

