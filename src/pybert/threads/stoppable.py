"""All pybert threads that support stop or abort derive from this class."""
from threading import Event, Thread


class StoppableThread(Thread):
    """Thread class with a stop() method.

    The thread itself has to check regularly for the stopped() condition.

    All PyBERT thread classes are subclasses of this class.
    """

    def __init__(self):
        super().__init__()
        self._stop_event = Event()

    def stop(self):
        """Called by thread invoker, when thread should be stopped
        prematurely."""
        self._stop_event.set()

    def stopped(self):
        """Should be called by thread (i.e. - subclass) periodically and, if this function
        returns True, thread should clean itself up and quit ASAP.
        """
        return self._stop_event.is_set()
