"""A group of channel solvers, used by the *PyBERT* application.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   20 September 2019

Each solver is given its own subdirectory
(containing a @__init__.py@ file, so it can be interpreted as a Python package).
This approach is taken, as opposed to creating a single Python file for each,
so as to support solvers, which may need extra data/support files.

Copyright (c) 2019 by David Banas; all rights reserved World wide.
"""
__all__ = [
    "simbeor",
]  # Should contain the name of each submodule.
# from . import *  # Makes each solver package available as: solvers.simbeor, etc.
# Actually, the above line causes a self-import.
# The `__all__` definition, above, should be all we need, such that
# `from pybert.solvers import *` does the right thing.
