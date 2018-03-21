[![Github release](https://img.shields.io/github/release/dih5/duat.svg)](https://github.com/dih5/duat/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/duat.svg)](https://pypi.python.org/pypi/duat)

[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/Dih5/duat/master/LICENSE.txt)

[![Documentation Status](https://readthedocs.org/projects/duat/badge/?version=latest)](http://duat.readthedocs.io/en/latest/?badge=latest)


# duat
A Python interface to the OSIRIS PIC code.

Note this package is not supported by the OSIRIS developers, use at your own risk.


* [Features](#features)
* [Installation](#installation)
* [Running](#running)
* [Versioning](#versioning)


## Features
The package provides an interface to the OSIRIS code, allowing:
* Definition of input files in Python syntax, as well as loading data from an existing OSIRIS input.
* Running OSIRIS from the Python interpreter, perhaps using a grid system and/or mpi.
* Plotting results, reducing multidimensional data. Animations in time can also be created.


## Installation
Assuming you have [Python](https://www.python.org/) with [pip](https://pip.pypa.io/en/stable/installing/):
* To install the last release from pypi: ```pip install --upgrade duat```.
* To install the last development version: ```pip install --upgrade https://github.com/dih5/duat/archive/master.zip```. 

If you are a [cool guy](https://wiki.python.org/moin/Python2orPython3) you'll prefer to use python3 and pip3 instead (unless you are a cool [Arch Linux user](https://www.archlinux.org/news/python-is-now-python-3/)).

## Running
The suggested environment to run duat is [Jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html), but
you can also run it from a python script or from the python interpreter.

The updated documentation can be found [here](http://duat.readthedocs.io/en/latest/index.html#), check it for a starting
guide.


## Versioning
Beta releases will be tagged as v0.Y.Z where Y>1. For this beta releases, Y will increase if and only if backward-incompatible
public API changes are introduced. 
