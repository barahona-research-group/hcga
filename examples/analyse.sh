#!/bin/bash

hcga feature_analysis -ff $1 -m sklearn -c 'RF' --no-kfold
