"Tx FFE tap weight tuner, used by the optimizer."

from traits.api import Bool, Float, HasTraits, Int, String


class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name = String("(noname)")
    enabled = Bool(False)
    min_val = Float(0.0)
    max_val = Float(0.0)
    value = Float(0.0)
    steps = Int(0)  # Non-zero means we want to sweep it.

    # pylint: disable=too-many-arguments
    def __init__(self, name="(noname)", enabled=False, min_val=0.0, max_val=0.0, value=0.0, steps=0):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.name = name
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.steps = steps
