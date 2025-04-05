from _typeshed import Incomplete
from pyibisami.ibis.parser import parse_ibis_file as parse_ibis_file
from traits.api import HasTraits

class IBISModel(HasTraits):
    pin_: Incomplete
    pin_rlcs: Incomplete
    model: Incomplete
    pins: Incomplete
    models: Incomplete
    def get_models(self, mname): ...
    def get_pins(self): ...
    debug: Incomplete
    GUI: Incomplete
    def __init__(self, ibis_file_name, is_tx, debug: bool = ..., gui: bool = ...) -> None: ...
    def info(self): ...
    def __call__(self) -> None: ...
    def log(self, msg, alert: bool = ...) -> None: ...
    def default_traits_view(self): ...
    @property
    def ibis_parsing_errors(self): ...
    @property
    def log_txt(self): ...
    @property
    def model_dict(self): ...
    @property
    def dll_file(self): ...
    @property
    def ami_file(self): ...
