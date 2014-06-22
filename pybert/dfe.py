#! /usr/bin/env python

"""
Behavioral model of a decision feedback equalizer (DFE).

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

This Python script provides a behavioral model of a decision feedback
equalizer (DFE). The class defined, here, is intended for integration
into the larger `PyBERT' framework, but is also capable of
running in stand-alone mode for preliminary debugging.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from enthought.traits.api \
    import HasTraits, Array, Range, Float, Enum, Property, String, List, cached_property
from enthought.traits.ui.api import View, Item, VSplit, Group, VGroup, HGroup, Label, Action, Handler, DefaultOverride
from enthought.chaco.chaco_plot_editor import ChacoPlotItem
from numpy import arange, real, concatenate, angle, sign, sin, pi, array, float, zeros
from numpy.fft import ifft
from numpy.random import random
from scipy.signal import lfilter, firwin, iirdesign, iirfilter, freqz
import re

# Default model parameters - Modify these to customize the default simulation.
gNpts  = 1024    # number of vector points
gNtaps = 6
gGain  = 1.0

# Model Proper - Don't change anything below this line.
class DFE(HasTraits):
    """A Traits based class providing behavioral modeling of a decision
    feedback equalizer (DFE)."""

    def __init__(self, n_taps, gain):
        super().__init__()
        self.tap_weights = Array([0.0] * n_taps)
        self.tap_values  = Array([0.0] * n_taps)
        self.gain        = gain

    def step(self, decision, error):
        """Step the DFE, according to the new decision and error inputs."""

        # Copy class object variables into local function namespace, for efficiency.
        tap_weights = self.tap_weights
        tap_values  = self.tap_values
        gain        = self.gain

        # Perform the adaptation step and generate the new output.
        tap_weights = map(lambda x: x + x * error * gain, tap_weights)
        tap_values  = Array([decision] + tap_values[:-1])
        filter_out  = sum(tap_weights * tap_values)

        # Copy local values back to their respective class object variables.


