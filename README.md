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
* Definition of input files in Python syntax. With loops and stuff.
* Running OSIRIS from the Python interpreter.
* Plotting results, reducing multidimensional data.


## Installation
Assuming you have [Python](https://www.python.org/) with [pip](https://pip.pypa.io/en/stable/installing/):
* To install the last 'stable' version from pypi: ```pip install duat```. Do not expect much [stability](#versioning) in the API though...
* To install the last development version: ```pip install --upgrade https://github.com/dih5/duat/archive/master.zip```. 

If you are a [cool guy](https://wiki.python.org/moin/Python2orPython3) you'll prefer to use python3 and pip3 instead, unless you are a cool [Arch Linux user](https://www.archlinux.org/news/python-is-now-python-3/).

## Running
The suggested environment to run duat is [Jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html), but
you can also run it from a python script or from the python interpreter.

The updated documentation can be found [here](http://duat.readthedocs.io/en/latest/index.html#), check it for a starting
guide.


## Versioning
Alpha releases (v0.1.Z) will be intended for personal use and the API may freely change, so do not expect stability yet. 
 
However, the Python item access notation for editing the configuration file is unlikely to change, so you may use the package at your own risk expecting only to have to update the running and plotting API. 
