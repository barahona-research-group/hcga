# Highly comparative graph analysis


## Installation

Navigate to the folder with setup.py and type:

```pip install -e .```


## Usage

See https://imperialcollegelondon.github.io/hcga/ for the documentation. 

## Cite

Please cite our paper if you use this code in your own work:

```
R. Peach, H. Palasciano, A. Arnaudon, M. Brahona, “hcga: Highly Comparative Graph Analysis for graph phenotyping”, In preparation, 2019

```

###

Download your data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets 

Add the data into TestData folder e.g. /TestData/ENZYMES/

### Run test file

In the command line type

```
python test.py

```
### Run an example
```
from hcga.graphs import Graphs

g = Graphs(directory='/home/robert/Documents/PythonCode/hcga/hcga/TestData',dataset='ENZYMES')

g.calculate_features(calc_speed='slow')
```


