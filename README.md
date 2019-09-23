# Highly comparative graph analysis


## Installation

Navigate to the folder with setup.py and type:

```pip install .```

and then type:

```python setup.py install```

## Dependencies


## Usage

See https://imperialcollegelondon.github.io/hcga/ for the documentation. 

## Cite

Please cite our paper if you use this code in your own work:

```

```

Functions to add/complete:

Sparsification function: implements various sparsification techniques and identifies important edges etc
Smallworldness function: implement the omega function - need to construct a lattice. networkx implementation is too slow.
Graph similarity measures: Build a similarity space between all graphs according to some kernel and then use the coordinates in that space
as a new feature.
Scalefree function: This needs massive expansion...


## Documentation compilation
We followed this webpage to create the doc with sphinx: https://daler.github.io/sphinxdoc-test/includeme.html

### Create the doc structure
This is to be done the first time, then the next to upate it. 
1) Create a folder called hcga-docs near the hcga git folder (same level):
``` mkdir hcga-docs```
2) Clone the hcga repo in subfolder html: 
```git clone https://github.com/ImperialCollegeLondon/hcga.git html```
3) move into it: 
```cd html```
4) Switch branches to gh-pages:
```
git branch gh-pages
git symbolic-ref HEAD refs/heads/gh-pages  # auto-switches branches to gh-pages
rm .git/index
git clean -fdx
git branch
```

### Update the doc
1) in the main repo (hcg), do
```
cd docs
make html
```
this will update the files in hcga-docs. 

2) To update the git at the same time as compiling, just do 
```
make full
```

Alternatively, you can update the git by hand using:
```
cd ../../hcga-docs/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages
```

###

Download your data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets 

Add the data into TestData folder e.g. /TestData/ENZYMES/


### Run an example
```
from hcga.graphs import Graphs

g = Graphs(directory='/home/robert/Documents/PythonCode/hcga/hcga/TestData',dataset='ENZYMES')

g.calculate_features(calc_speed='slow')
```


