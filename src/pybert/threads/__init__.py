"""
Various types of Python thread definitions used by *PyBERT*.

A separate thread is used for optimization, in order to preserve GUI responsiveness to user input while optimizing.
And all custom threads used in PyBERT are derived from `StoppableThread`, so that any optimization may be aborted by the user.
"""
