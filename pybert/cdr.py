"""
Behavioral model of a "bang-bang" clock data recovery (CDR) unit.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

This Python script provides a behavioral model of a "bang-bang" clock
data recovery (CDR) unit. The class defined, here, is intended for
integration into the larger *PyBERT* framework.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from numpy import sign, array, arange, mean, where

class CDR(object):
    """
    A class providing behavioral modeling of a 'bang- bang' clock
    data recovery (CDR) unit.
    """

    def __init__(self, delta_t, alpha, ui, n_lock_ave=500, rel_lock_tol=0.01, lock_sustain=500):
        """
        Args:
            delta_t (float): The proportional branch correction, in seconds.
            alpha (float): The integral branch correction, normalized to
                proportional branch correction.
            ui (float): The nominal unit interval, in seconds.
            n_lock_ave (Optional, int): Number of unit intervals to use for
                determining lock. Defaults to 500.
            rel_lock_tol(Optional, float): Lock tolerance, relative to
                *delta_t*. Defaults to 0.01.
            lock_sustain(Optional, int): Length of lock sustain vector
                used to provide histerysis. Defaults to 500.

        Notes:
            The code does not care what units are actually used for
            'delta_t' and 'ui'; only that they are the same.
        """

        self.delta_t             = delta_t
        self.alpha               = alpha
        self.nom_ui              = ui
        self._ui                 = ui
        self.n_lock_ave          = n_lock_ave
        self.rel_lock_tol        = rel_lock_tol
        self._locked             = False
        self.lock_sustain        = lock_sustain
        self.integral_corrections     = [0.0]
        self.proportional_corrections = []
        self.lockeds                  = []

    @property
    def ui(self):
        """The current unit interval estimate."""

        return self._ui

    @property
    def locked(self):
        """The current locked state."""

        return self._locked

    def adapt(self, samples):
        """
        Adapt period/phase, according to 3 samples.

        Should be called, when the clock has just struck.

        Synopsis:
          (ui, locked) = adapt(samples)

        Args:
            samples ([float]): A list of 3 samples of the input waveform, as follows:

                - at the last clock time
                - at the last unit interval boundary time
                - at the current clock time

        Returns:
            tuple (float, bool):

                ui:
                    The new unit interval estimate, in seconds.
                locked:
                    Boolean flag indicating 'locked' status.

        """

        integral_corrections     = self.integral_corrections
        proportional_corrections = self.proportional_corrections
        delta_t                  = self.delta_t
        locked                   = self._locked
        lockeds                  = self.lockeds
        lock_sustain             = self.lock_sustain
        n_lock_ave               = self.n_lock_ave
        rel_lock_tol             = self.rel_lock_tol

        integral_correction      = integral_corrections[-1]

        samples = map(sign, samples)
        if(samples[0] == samples[2]):   # No transition; no correction.
            proportional_correction = 0.0
        elif(samples[0] == samples[1]): # Early clock; increase period.
            proportional_correction = delta_t
        else:                           # Late clock; decrease period.
            proportional_correction = -delta_t
        integral_correction += self.alpha * proportional_correction
        ui = self.nom_ui + integral_correction + proportional_correction

        integral_corrections.append(integral_correction)
        if(len(integral_corrections) > n_lock_ave):
            integral_corrections.pop(0)
        proportional_corrections.append(proportional_correction)
        if(len(proportional_corrections) > n_lock_ave):
            proportional_corrections.pop(0)
        if(len(proportional_corrections) == n_lock_ave):
            x      = array(integral_corrections) #- mean(integral_corrections)
            var    = sum(x ** 2) / n_lock_ave
            lock   = abs(mean(proportional_corrections) / delta_t) < rel_lock_tol and (var / delta_t) < rel_lock_tol
            lockeds.append(lock)
            if(len(lockeds) > lock_sustain):
                lockeds.pop(0)
            if(locked):
                if(len(where(array(lockeds) == True)[0]) < 0.2 * lock_sustain):
                    locked = False
            else:
                if(len(where(array(lockeds) == True)[0]) > 0.8 * lock_sustain):
                    locked = True
            self._locked = locked

        self.integral_corrections     = integral_corrections
        self.proportional_corrections = proportional_corrections
        self.lockeds                  = lockeds
        self._ui                      = ui

        return (ui, locked)

