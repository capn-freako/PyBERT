import numpy as np
from .model import AMIModelInitializer
from .parameter                 import AMIParamError, AMIParameter
from .reserved_parameter_names  import AmiReservedParameterName, RESERVED_PARAM_NAMES
from _typeshed import Incomplete
from numpy.typing import NDArray
from traits.api import HasTraits
from typing         import Any, Callable, NewType, Optional, TypeAlias

__all__ = ['ParamName', 'ParamValue', 'Parameters', 'ParamValues', 'AmiName', 'AmiAtom', 'AmiExpr', 'AmiNode', 'AmiNodeParser', 'AmiParser', 'ami_parse', 'AMIParamConfigurator']

ParamName  = NewType("ParamName", str)
ParamValue:  TypeAlias = int | float | str | list["ParamValue"]
Parameters:  TypeAlias = dict[ParamName, "'AMIParameter' | 'Parameters'"]
ParamValues: TypeAlias = dict[ParamName, "'ParamValue'   | 'ParamValues'"]

AmiName = NewType("AmiName", str)
AmiAtom: TypeAlias = bool | int | float | str
AmiExpr: TypeAlias = "'AmiAtom' | 'AmiNode'"
AmiNode: TypeAlias = tuple[AmiName, list[AmiExpr]]
AmiNodeParser: TypeAlias = Callable[[str], AmiNode]
AmiParser:     TypeAlias = Callable[[str], tuple[AmiName, list[AmiNode]]]  # Atoms may not exist at the root level.

ParseErrMsg = NewType("ParseErrMsg", str)
AmiRootName = NewType("AmiRootName", str)
ReservedParamDict: TypeAlias = dict[AmiReservedParameterName, AMIParameter]
ModelSpecificDict: TypeAlias = dict[ParamName, "'AMIParameter' | 'ModelSpecificDict'"]

class AMIParamConfigurator(HasTraits):
    def __init__(self, ami_file_contents_str: str) -> None: ...
    def __call__(self) -> None: ...
    def open_gui(self) -> None: ...
    def default_traits_view(self): ...
    def fetch_param(self, branch_names): ...
    def fetch_param_val(self, branch_names): ...
    def set_param_val(self, branch_names, new_val) -> None: ...
    @property
    def ami_parsing_errors(self): ...
    @property
    def ami_param_defs(self) -> dict[str, ReservedParamDict | ModelSpecificDict]: ...
    @property
    def input_ami_params(self) -> ParamValues: ...
    def input_ami_param(self, params: Parameters, pname: ParamName, prefix: str = '') -> ParamValues: ...
    @property
    def info_ami_params(self): ...
    def get_init(self, bit_time: float, sample_interval: float, channel_response: NDArray[np.longdouble], ami_params: dict[str, Any] | None = None) -> AMIModelInitializer: ...
# atom = number | symbol | ami_string | true | false
# expr = atom | node
ami_parse: AmiParser
