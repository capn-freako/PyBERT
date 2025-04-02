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
    steps = Int(0)  # Non-zero means we want to sweep it.

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, name="(noname)", pos=0, enabled=False, min_val=-0.1, max_val=0.1, step=0.01, value=0.0, steps=0):
        """Allows user to define properties, at instantiation."""

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
        self.steps = steps
