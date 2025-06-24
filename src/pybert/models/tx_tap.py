"Tx FFE tap weight tuner, used by the optimizer."

from traits.api import Bool, Float, HasTraits, Int, String


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name = String("(noname)")
    pos = Int(0)  # negative = pre-cursor; positive = post-cursor
    enabled = Bool(False)
    min_val = Float(-0.1)
    max_val = Float(0.1)
    step = Float(0.01)
    value = Float(0.0)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self, name: str = "(noname)", pos: int = 0, enabled: bool = False,
        min_val: float = -0.1, max_val: float = 0.1, step: float = 0.01, value: float = 0.0
    ):
        """
        Allows user to define properties, at instantiation.

        Keyword Args:
            name: Tap name/label.
                Default: "(noname)"
            pos: Tap position (0 = cursor).
                Default: 0
            enabled: Will participate in EQ optimization when *True*.
                Default: *False*
            min_val: Minimum allowed value during optimization.
                Default: -0.1
            max_val: Maximum allowed value during optimization.
                Default: 0.1
            step: Increment used during optimization.
                Default: 0.01
            value: Current value.
                Default: 0.0
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.name = name
        self.pos = pos
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = value
