"""A package of Python modules, used by the *PyBERT* application.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by:      Mark Marlett <mark.marlett@gmail.com>

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
__version__ = "4.1.0"
__date__ = "March 15, 2023"
__authors__ = "David Banas & David Patterson"
__copy__ = "Copyright (c) 2014 David Banas, 2019 David Patterson"

# You can play with the different alternatives below,
# to try and optimize the visual appearance of PyBERT plots.
# You should only have one of the "ETSConfig.toolkit = ..." lines uncommented.
# fmt: off
# isort: off
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'  # Yields unacceptably small font sizes in plot axis labels.
# ETSConfig.toolkit = 'qt4.celiagg'   # Yields unacceptably small font sizes in plot axis labels.
# ETSConfig.toolkit = 'qt.qpainter'  # Was causing crash on Mac.
# ETSConfig.toolkit = 'qt.image'     # Program runs, but very small fonts in plot titles and axis labels.
# ETSConfig.toolkit = 'wx'           # Crashes on launch.
# fmt: on
# isort: on
