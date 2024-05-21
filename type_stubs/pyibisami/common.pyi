from typing import TypeAlias, TypeVar
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
PI: float
TWOPI: float

def deconv_same(y: Rvec, x: Rvec) -> Rvec: ...
