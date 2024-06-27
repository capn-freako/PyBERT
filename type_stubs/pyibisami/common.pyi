from _typeshed import Incomplete
from typing import TypeVar

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: Incomplete
Cvec: Incomplete
PI: float
TWOPI: float

def deconv_same(y: Rvec, x: Rvec) -> Rvec: ...
