# Functional Neural Architectures (FNA)

![testsuite-CI](https://github.com/rcfduarte/func-neurarch/workflows/testsuite-CI/badge.svg)
[![License](http://img.shields.io/:license-GPLv2+-green.svg)](http://www.gnu.org/licenses/gpl-2.0.html)

Functional Neural Architectures (FNA) is a python library for neural network functional benchmarking, analysis and comparison. It provides high-level wrapper for PyNEST and TensorFlow (which is used as the core simulation engines). As such, the types of architectures and their properties are determined by the
models available in NEST / TF. The use of these simulators allows efficient and highly scalable simulations of very
large and complex circuits, constrained only by the computational resources available to the user.

The modular design allows the user to specify numerical experiments with varying degrees of
complexity depending on concrete research objectives. The generality of some of these experiments
allows the same types of measurements to be performed on a variety of different circuits, which can
be useful for benchmarking and comparison purposes. Additionally, the code was designed to allow
an effortless migration across computing systems, i.e. the same simulations can be executed in a
local machine, in a computer cluster or a supercomputer, with straightforward resource allocation. 

The code is licensed under GPLv2 and available on [GitHub](https://github.com/rcfduarte/func-neurarch/).

### Getting started

For a detailed description of the framework and to make the most out of it, please read the 
[documentation](https://rcfduarte.github.io/nmsat/). 


#### Dependencies

* **Python** 3.6 or higher
* [**NEST**](http://www.nest-simulator.org/) version 2.16.0 or higher
* [**TensorFlow**](http://www.nest-simulator.org/) version 1.15.0 or smaller
* **numpy** version 1.14.0 or higher 
* **scipy** version 0.12.0 or higher
* **scikit-learn** version 0.18.0 or higher
* **matplotlib** version 1.2.0 or higher

Optional (for additional functionality):
* [**PySpike**](https://github.com/mariomulansky/PySpike) version 0.5.1
* **h5py** version 2.2.1 or higher
* **mayavi** 
* **networkx**
* **pytest**
* **mnist**
* **nltk**
* **pandas**
* **tqdm**

To install all dependencies for Python 3, run

```bash
python3 -m pip install -r requirements.txt
```

and compile NEST with the correct Python bindings
```bash
-Dwith-python=3
```

#### Installation

The code is currently available only in this GitHub repository. To use it, simply download the source or fork and 
clone the repository, and installing it as a library (**fna**) by running

```
pip install .
```

or install directly from GitHub

```
pip install git+https://github.com/rcfduarte/func-neurarch.git
```

We recommend that all dependencies are satisfied before installation, in particular the TensorFlow and NEST libraries. 


### License 

Copyright (C) 2020  Renato Duarte, Barna Zajzon  

Uses parts from NeuroTools for which Copyright (C) 2008  Daniel Bruederle, Andrew Davison, Jens Kremkow
Laurent Perrinet, Michael Schmuker, Eilif Muller, Eric Mueller, Pierre Yger

Neural Mircocircuit Simulation and Analysis Toolkit is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

See the [LICENSE](LICENSE) file for details.


### Acknowledgments

This work was done in the **[Computation in Neural Circuits (CiNC)](http://www.fz-juelich.de/inm/inm-6/EN/Forschung/Morrison/artikel.html)** group, at the Institute for Neuroscience and Medicine (INM-6) and Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Germany. 
We would like to thank Professor Abigail Morrison for her continued patience, advice and support and the **[Neurobiology of Language](http://www.mpi.nl/departments/neurobiology-of-language)** group, at the Max-Planck for Psycholinguistics, for valuable discussions and contributions.

We acknowledge partial support by the German Ministry for Education and Research (Bundesministerium für Bildung und Forschung) BMBF Grant 01GQ0420 to BCCN Freiburg, the Helmholtz Alliance on Systems Biology (Germany), the Initiative and Networking Fund of the Helmholtz Association, the Helmholtz Portfolio theme ‘Supercomputing and Modeling for the Human Brain’.
We additionally acknowledge the computing time granted by the JARA-HPC Vergabegremium on the supercomputer **[JURECA](https://jlsrf.org/index.php/lsf/article/view/121/pdf)** at Forschungszentrum Jülich, used during development.
