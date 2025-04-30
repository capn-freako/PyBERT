from _typeshed import Incomplete
from dataclasses import dataclass

@dataclass
class AmiReservedParameterName:
    pname: str
    def __post_init__(self) -> None: ...

RESERVED_PARAM_NAMES: Incomplete
