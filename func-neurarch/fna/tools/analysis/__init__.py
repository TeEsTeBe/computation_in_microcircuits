"""
========================================================================================================================
Analysis Module
========================================================================================================================
Collection of analysis and utility functions that are used by other tools
(Note: this documentation is incomplete)

Functions:
------------
ccf 						- fast cross-correlation function, using fft
_dict_max 					- for a dict containing numerical values, return the key for the highest
crosscorrelate 				-
makekernel 					- creates kernel functions for convolution
simple_frequency_spectrum 	- simple calculation of frequency spectrum

========================================================================================================================
Copyright (C) 2018  Renato Duarte, Barna Zajzon

Neural Mircocircuit Simulation and Analysis Toolkit is free software;
you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

"""

__all__ = ['metrics']
