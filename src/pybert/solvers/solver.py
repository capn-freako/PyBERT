"""Common interface to channel solvers.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   22 September 2019

This class is virtual and must be instantiated and over-ridden by each
custom channel solver.
The class itself just provides a consistent interface to solvers, for PyBERT.
Each solver must define how the standard interface is implemented.

Copyright (c) 2019 by David Banas; all rights reserved World wide.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

ChType = Enum("ChType", "microstrip_se microstrip_diff stripline_se stripline_diff")


class Solver(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class providing a consistent interface to channel
    solver."""

    @abstractmethod
    def solve(  # pylint: disable=too-many-arguments,dangerous-default-value,too-many-positional-arguments
        self,
        ch_type: str = "microstrip_se",  #: Channel cross-sectional configuration.
        diel_const: float = 4.3,  #: Dielectric constant of substrate at ``des_freq`` (rel.).
        loss_tan: float = 0.02,  #: Loss tangent at ``des_freq``.
        des_freq: float = 1.0e9,  #: Frequency at which ``diel_const`` and ``loss_tan`` are quoted (Hz).
        thickness: float = 0.036,  #: Trace thickness (mm).
        width: float = 0.254,  #: Trace width (mm).
        height: float = 0.127,  #: Trace height above/below ground plane (mm).
        separation: float = 0.508,  #: Trace separation (mm).
        roughness: float = 0.004,  #: Trace surface roughness (mm-rms).
        fs: List[float] = [],  #: Angular frequency sample points (Hz).  # pylint: disable=dangerous-default-value
        lic_path: str = "",  #: Path to license file.
        lic_name: str = "",  #: Name of license type (if needed by solver).
        prj_name: str = "",  #: Name of project (if needed by solver).
    ) -> Tuple[List[complex], List[complex], List[float]]:
        """Solves a particular channel cross-section.

        Returns:
            gamma: Frequency dependent complex propagation constant.
            Zc: Frequency dependent complex impedance.
            freqs: List of frequencies at which ``gamma`` and ``Zc`` were sampled.
        """
