# *hcga*: Highly comparative graph analysis

This is the official repository of *hcga*, a highly comparative graph analysis toolbox. It performs a massive feature extraction from a set of graphs, and apply supervised classification methods. 

## Installation

Navigate to the main folder simply type:

```pip install -e .```

## Documentation

See https://imperialcollegelondon.github.io/hcga/ for the documentation. 

## Cite

Please cite our paper if you use this code in your own work:

```
R. Peach, H. Palasciano, A. Arnaudon, M. Barahona, “hcga: Highly Comparative Graph Analysis for graph phenotyping”, In preparation, 2019

```

## Run test file

In the example folder, the script ``run.py`` can be used to run the examples of the paper, simply as 

```./run.sh DATASET```

where ``DATASET`` can be one of 
* ``ENZYMES``
* ``DD``
* ``COLLAB``
* ``PROTEINS``
* ``REDDIT-MULTI-12K``
* ``NEURONS``

More comments are in the scripts for some parameters choices. 

