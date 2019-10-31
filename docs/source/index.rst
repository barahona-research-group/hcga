..  -*- coding: utf-8 -*-

.. _contents:

*hcga*: Highly Comparative Graph Analysis
============================================

*hcga* is inspired by *hctsa*, the time serie massive feature extraction framework of [Hctsa]_ and instead extracts features from graphs to classify or regress them.  
This code comes with the companion paper [Hcga]_  containing more details and examples of applications. 

Installation
-------------------

To install the dev version from `GitHub <https://github.com/imperialcollegelondon/hcga/>`_ with the commands::

       $ git clone https://github.com/ImperialCollegeLondon/hcga.git
       $ cd hcga
       $ pip install -e .

There is no release version yet, but stay tuned for more!

Usage
-----

In the example folder, the script ``run.py`` can be used to run the examples of the paper, simply as 

```./run.sh DATASET```

where ``DATASET`` can be one of 
    * ENZYMES
    * DD
    * COLLAB
    * PROTEINS
    * REDDIT-MULTI-12K
    * NEURONS
    * HELICENES

More comments are in the scripts for some parameters choices. 

Citing
------

To cite *hcga*, please use [Hcga]_. 

Credits
-------

The code is still a preliminary version, and written by us.

Original authors:
^^^^^^^^^^^^^^^^^

- Robert Peach, GitHub: `peach-lucien <https://github.com/peach-lucien>`_
- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`_
- Henry Palasciano, GitHub: `henrypalasciano <https://github.com/henrypalasciano>`_

Contributors:
^^^^^^^^^^^^^

Any contributors are welcome, please contact us if interested. 

Bibliography
------------

.. [Hcga]  R. Peach, H. Palasciano, A. Arnaudon, M. Barahona, 
                “hcga: Highly Comparative Graph Analysis for graph phenotyping”, In preparation, 2019
.. [Hctsa]  B. D. Fulcher and N. S. Jones, 
                “hctsa:  A computational framework for automated time-series phenotyping using massive feature extraction,” Cell systems, vol. 5, no. 5, pp. 527–531, 2017

API documentation
=================

Documentation of the API of *hcga*. 

.. toctree::
   :maxdepth: 3

   graph
   operations
   Features


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
