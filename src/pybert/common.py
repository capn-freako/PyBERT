"""
Definitions common to all PyBERT modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   May 13, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional, TypeAlias, TypeVar  # pylint: disable=unused-import  # noqa: F401
import numpy as np  # type: ignore  # pylint: disable=unused-import  # noqa: F401
import numpy.typing as npt  # type: ignore

# Type variables, used to support generics.
S = TypeVar('S', Any, Any)
T = TypeVar('T', Any, Any)

# Handy often used type aliases.
Real = TypeVar('Real', np.float64, np.float64)			#: Real scalar
Comp = TypeVar('Comp', np.complex64, np.complex128)		#: Complex scalar
# I suspect I should move to this:
# Rvec: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]],
# in order to explicitly state the 1-D nature of `Rvec`.
# However, doing so breaks a LOT, all having the form:
#   Incompatible return value type (got "ndarray[tuple[int, ...], dtype[float64]]", expected "ndarray[tuple[int], dtype[float64]]")
Rvec: TypeAlias = npt.NDArray[Real]						#: Complex valued vector
Cvec: TypeAlias = npt.NDArray[Comp]                     #: Complex valued vector
Rmat: TypeAlias = npt.NDArray[Real]						#: Real valued matrix
Cmat: TypeAlias = npt.NDArray[Comp]						#: Complex valued matrix

# Handy often used constants.
# PI    = np.pi  # Making autodoc barf.
PI    = 3.14159		#: Value of Pi
TWOPI = 2.0 * PI 	#: 2 * Pi
