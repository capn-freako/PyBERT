"""
Common interface to channel solvers.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   22 September 2019

This class is virtual and must be instantiated and over-ridden by each
custom channel solver.
The class itself just provides a consistent interface to solvers, for PyBERT.
Each solver must define how the standard interface is implemented.

Copyright (c) 2019 by David Banas; all rights reserved World wide.
"""
from abc    import ABC, abstractmethod
from typing import List, Tuple, Dict
from enum   import Enum

ChType = Enum('ChType', 'microstrip_se microstrip_diff stripline_se stripline_diff')

class Solver(ABC):
    """Abstract base class providing a consistent interface to channel solver.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def solve(self,
          ch_type    : ChType = "microstrip_se",  #: Channel cross-sectional configuration.
          diel_const : float  = 4.0,    #: Dielectric constant of substrate (rel.).
          thickness  : float  = 0.036,  #: Trace thickness (mm).
          width      : float  = 0.254,  #: Trace width (mm).
          height     : float  = 0.127,  #: Trace height above/below ground plane (mm).
          separation : float  = 0.508,  #: Trace separation (mm).
          roughness  : float  = 0.004,  #: Trace surface roughness (mm-rms).
          ):
        pass
