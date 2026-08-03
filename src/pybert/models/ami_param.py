"IBIS-AMI Model_Specific parameter tuner, used by the optimizer."

from traits.api import Bool, Float, HasTraits, List, Str


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class AmiParamTuner(HasTraits):
    """Object used to populate the rows of the Tx/Rx AMI parameter tuning tables.

    Represents one *Model_Specific* 'In'/'InOut' AMI parameter, of 'Range'
    format, offered to the user for inclusion in EQ optimization.
    """

    name = Str("(noname)")  # Display name (fully hierarchical, e.g. "tx_preset_coeffs_main").
    branch_names = List(Str)  # Hierarchical path for `fetch_param_val()`/`set_param_val()`.
    enabled = Bool(False)  # Will participate in EQ optimization when *True*.
    min_val = Float(0.0)  # Minimum allowed value during optimization.
    max_val = Float(0.0)  # Maximum allowed value during optimization.
    step = Float(0.0)  # Increment used during optimization.
    value = Float(0.0)  # Current value.
    is_int = Bool(False)  # Round swept values to the nearest integer when *True*.

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self, name: str = "(noname)", branch_names: "list[str] | None" = None, enabled: bool = False,
        min_val: float = 0.0, max_val: float = 0.0, step: float = 0.0, value: float = 0.0, is_int: bool = False
    ):
        """
        Allows user to define properties, at instantiation.

        Keyword Args:
            name: Parameter name/label.
                Default: "(noname)"
            branch_names: Hierarchical path to the parameter, rooted at "Model_Specific".
                Default: []
            enabled: Will participate in EQ optimization when *True*.
                Default: *False*
            min_val: Minimum allowed value during optimization.
                Default: 0.0
            max_val: Maximum allowed value during optimization.
                Default: 0.0
            step: Increment used during optimization.
                Default: 0.0
            value: Current value.
                Default: 0.0
            is_int: Round swept values to the nearest integer when *True*.
                Default: *False*
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.name = name
        self.branch_names = branch_names or []
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = value
        self.is_int = is_int
