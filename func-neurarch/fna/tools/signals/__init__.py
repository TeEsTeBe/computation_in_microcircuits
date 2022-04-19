"""
========================================================================================================================
Signals Module
========================================================================================================================
Collection of utilities and functions to create, use and manipulate signals (spike data or analog data)
(incomplete documentation)

========================================================================================================================
"""

__all__ = ['analog', 'spikes', 'helper']

from .analog import AnalogSignalList, AnalogSignal
from .spikes import SpikeList, SpikeTrain
from .helper import (convert_array, convert_activity, pad_array, to_pyspike,
                     make_simple_kernel, shotnoise_fromspikes)
