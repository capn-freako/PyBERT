"""
Behavioral model of a decision feedback equalizer (DFE).

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

This Python script provides a behavioral model of a decision feedback
equalizer (DFE). The class defined, here, is intended for integration
into the larger *PyBERT* framework.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from typing import Any, Optional

import numpy        as np
import numpy.typing as npt

from numpy        import array, mean, sign, zeros
from scipy.signal import iirfilter

from pybert.common     import Rvec
from pybert.models.cdr import CDR

gNch_taps = 3  # Number of taps used in summing node filter.


class LfilterSS:  # pylint: disable=too-few-public-methods
    """A single steppable version of ``scipy.signal.lfilter()``."""

    # def __init__(self, b: list[float], a: list[float]):
    def __init__(self, b: npt.NDArray[np.float64], a: npt.NDArray[np.float64]):
        """
        Args:
            b: Coefficients of the numerator of the rational transfer function.
            a: Coefficients of the denominator of the rational transfer function.
        """

        if a[0] != 1.0:
            b = array(b) / a[0]
            a = array(a) / a[0]

        self.b = b
        self.a = a
        self.xs = [0.0] * (len(b) - 1)
        self.ys = [0.0] * (len(a) - 1)

    def step(self, x):
        """Step the filter.

        Args:
            x(float): Next input value.

        Returns:
            (float): Next output value.
        """

        b = self.b
        a = self.a
        xs = self.xs
        ys = self.ys

        y = sum(b * ([x] + xs)) - sum(a[1:] * ys)
        xs = [x] + xs[:-1]
        ys = [y] + ys[:-1]

        self.xs = xs
        self.ys = ys

        return y


class DFE:  # pylint: disable=too-many-instance-attributes
    """Behavioral model of a decision feedback equalizer (DFE)."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        self,
        n_taps: int,
        gain: float,
        delta_t: float,
        alpha: float,
        ui: float,
        n_spb: int,
        decision_scaler: float,
        mod_type: int = 0,
        bandwidth: float = 100.0e9,
        n_ave: int = 10,
        n_lock_ave: int = 500,
        rel_lock_tol: float = 0.01,
        lock_sustain: int = 500,
        ideal: bool = True,
        limits: Optional[list[tuple[float, float]]] = None,
        agc_n_ave: int = 100,
    ):
        """
        Args:
            n_taps: # of taps in adaptive filter
            gain: adaptive filter tap weight correction gain
            delta_t: CDR proportional branch constant (ps)
            alpha: CDR integral branch constant (normalized to delta_t)
            ui: nominal unit interval (ps)
            n_spb: # of samples per unit interval
            decision_scaler: multiplicative constant applied to the result of the sign function, when making a "1 vs. 0" decision.
                Sets the target magnitude for the DFE.

        Keyword Args:
            mod_type: The modulation type

                - 0: NRZ
                - 1: Duo-binary
                - 2: PAM-4

            bandwidth: The bandwidth, at the summing node (Hz).
            n_ave: The number of averages to take, before adapting.
                (Also, the number of CDR adjustments per DFE adaptation.)
            n_lock_ave: The number of unit interval estimates to consider, when determining locked status.
            rel_lock_tol: The relative tolerance for determining lock.
            lock_sustain: Length of the histerysis vector used for lock flagging.
            ideal: Boolean flag. When true, use an ideal summing node.
            limits: List of pairs containing min/max values per tap.
            agc_n_ave: Number of previous slicer sample to keep, for AGC operation.

        Raises:
            RuntimeError: If the requested modulation type is unknown.
        """

        # Design summing node filter.
        fs = n_spb / ui
        (b, a) = iirfilter(gNch_taps - 1, bandwidth / (fs / 2), btype="lowpass")
        self.summing_filter = LfilterSS(b, a)

        # Initialize class variables.
        self.tap_weights = [0.0] * n_taps
        self.tap_values = [0.0] * n_taps
        self.gain = gain
        self.ui = ui
        self.decision_scaler = decision_scaler
        self.mod_type = mod_type
        self.cdr = CDR(delta_t, alpha, ui, n_lock_ave, rel_lock_tol, lock_sustain)
        self.n_ave = n_ave
        self.corrections = zeros(n_taps)
        self.ideal = ideal
        self.limits = limits
        self.agc_n_ave = agc_n_ave

        # Misc. finalization
        self.update_thresholds()

    def update_thresholds(self):
        """Update decision thresholds."""
        decision_scaler = self.decision_scaler
        match self.mod_type:
            case 0:
                thresholds = [0.0]
            case 1:
                thresholds = [-decision_scaler / 2.0, decision_scaler / 2.0]
            case 2:
                thresholds = [-decision_scaler * 2.0 / 3.0, 0.0, decision_scaler * 2.0 / 3.0]
            case _:
                raise RuntimeError("Unrecognized modulation type!")
        self.thresholds = thresholds

    def step(self, decision: float, error: float, update: bool):
        """
        Step the DFE, according to the new decision and error inputs.

        Args:
            decision: Current slicer output.
            error: Difference between summing node and slicer outputs.
            update: If true, update tap weights.

        Returns:
            res: New backward filter output value.
        """

        # Copy class object variables into local function namespace, for efficiency.
        tap_weights = self.tap_weights
        tap_values = self.tap_values
        gain = self.gain
        n_ave = self.n_ave

        # Calculate this step's corrections and add to running total.
        corrections = array([old + new for (old, new) in zip(self.corrections, [val * error * gain for val in tap_values])])

        # Update the tap weights with the average corrections, if appropriate.
        if update:
            if self.limits:
                limits = self.limits
                tap_weights = [max(limits[k][0],
                                   min(limits[k][1],
                                       weight + correction / n_ave))
                               for (k, (weight, correction)) in enumerate(zip(tap_weights, corrections))]
            else:
                tap_weights = [weight + correction / n_ave for (weight, correction) in zip(tap_weights, corrections)]
            corrections = zeros(len(corrections))  # Start the averaging process over, again.

        # Step the filter delay chain and generate the new output.
        tap_values = [decision] + tap_values[:-1]
        filter_out = sum(array(tap_weights) * array(tap_values))

        # Copy local values back to their respective class object variables.
        self.tap_weights = tap_weights
        self.tap_values = tap_values
        self.corrections = corrections

        return filter_out

    def decide(self, x: float) -> tuple[float, list[int]]:
        """Make the bit decisions, according to modulation type.

        Args:
            x: The signal value, at the decision time.

        Returns:
            A pair containing

                - One of:

                    - {-1, 1}              (NRZ)
                    - {-1, 0, +1}          (Duo-binary)
                    - {-1, -1/3, +1/3, +1} (PAM-4)

                - The list of bits recovered.

        Raises:
            RuntimeError: If the requested modulation type is unknown.
        """

        mod_type = self.mod_type

        if mod_type == 0:  # NRZ
            decision = sign(x)
            if decision > 0:
                bits = [1]
            else:
                bits = [0]
        elif mod_type == 1:  # Duo-binary
            if (x > self.thresholds[0]) ^ (x > self.thresholds[1]):
                decision = 0
                bits = [1]
            else:
                decision = sign(x)
                bits = [0]
        elif mod_type == 2:  # PAM-4
            if x > self.thresholds[2]:
                decision = 1
                bits = [1, 1]
            elif x > self.thresholds[1]:
                decision = 1.0 / 3.0
                bits = [1, 0]
            elif x > self.thresholds[0]:
                decision = -1.0 / 3.0
                bits = [0, 1]
            else:
                decision = -1
                bits = [0, 0]
        else:
            raise RuntimeError("ERROR: DFE.decide(): Unrecognized modulation type requested!")

        return decision, bits

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def run(
        self,
        sample_times: Rvec,
        signal: Rvec,
        use_agc: bool = False,
        dbg_dict: Optional[dict[str, Any]] = None
    ) -> tuple[
        npt.NDArray[np.float64],
        list[list[float]],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        list[bool],
        list[float],
        npt.NDArray[np.integer[Any]]
    ]:

        """
        Run the DFE on the input signal.

        Args:
            sample_times: Vector of signal sampling times.
            signal: Vector of sampled signal values.

        Keyword Args:
            use_agc: Perform continuous adjustment of `decision_scaler` when True.
                Default: False
            dbg_dict: Optional dictionary, for stashing debugging information at runtime.
                Default: None

        Returns:
            A tuple containing

                - res: Samples of the summing node output, taken at the times given in *sample_times*.
                - tap_weights: List of list of tap weights showing how the DFE adapted over time.
                - ui_ests: List of unit interval estimates, showing how the CDR adapted.
                - clocks: List of mostly zeros with ones at the recovered clocking instants.
                    Useful for overlaying the clock times on signal waveforms, in plots.
                - lockeds: List of Booleans indicating state of CDR lock.
                - clock_times: List of clocking instants, as recovered by the CDR.
                - bits: List of recovered bits.

        Raises:
            RuntimeError: If the requested modulation type is unknown.
        """

        ui = self.ui
        decision_scaler = self.decision_scaler
        n_ave = self.n_ave
        summing_filter = self.summing_filter
        ideal = self.ideal
        mod_type = self.mod_type
        thresholds = self.thresholds
        agc_n_ave = self.agc_n_ave

        clk_cntr = 0
        smpl_cntr = 0
        filter_out = 0.0
        nxt_filter_out = 0.0
        last_clock_sample = 0.0
        next_boundary_time = 0.0
        next_clock_time = ui / 2.0
        locked = False
        n_slicer_samps = 0
        n_ave_samps = 0

        res: list[float] = []
        tap_weights: list[list[float]] = [self.tap_weights]
        ui_ests: list[float] = []
        lockeds: list[bool] = []
        clocks = zeros(len(sample_times))
        clock_times = [next_clock_time]
        bits = []
        boundary_sample = 0
        slicer_samps = zeros(agc_n_ave)
        ave_samps = zeros(agc_n_ave)
        scalar_values = [decision_scaler]
        for t, x in zip(sample_times, signal):
            if not ideal:
                sum_out = summing_filter.step(x - filter_out)
            else:
                sum_out = x - filter_out
            res.append(sum_out)
            if t >= next_boundary_time:
                boundary_sample = sum_out
                filter_out = nxt_filter_out
                next_boundary_time += ui  # Necessary, in order to prevent premature reentry.
            if t >= next_clock_time:
                clk_cntr += 1
                clocks[smpl_cntr] = 1
                current_clock_sample = sum_out
                samples = array([last_clock_sample, boundary_sample, current_clock_sample])
                if mod_type == 0:  # NRZ
                    pass
                elif mod_type == 1:  # Duo-binary
                    if samples.mean() < 0.0:
                        samples -= thresholds[0]
                    else:
                        samples -= thresholds[1]
                elif mod_type == 2:  # PAM-4
                    pass
                else:
                    raise RuntimeError("ERROR: DFE.run(): Unrecognized modulation type!")
                ui, locked = self.cdr.adapt(samples)
                decision, new_bits = self.decide(sum_out)
                bits.extend(new_bits)
                slicer_output = decision * decision_scaler
                error = sum_out - slicer_output
                update = locked and (clk_cntr % n_ave) == 0
                if locked:  # We only want error accumulation to happen, when we're locked.
                    nxt_filter_out = self.step(slicer_output, error, update)
                else:
                    nxt_filter_out = self.step(slicer_output, 0.0, update)
                tap_weights.append(self.tap_weights)
                last_clock_sample = sum_out
                next_boundary_time = next_clock_time + ui / 2.0
                next_clock_time += ui
                clock_times.append(next_clock_time)
                if use_agc:
                    # Shift in new sample.
                    slicer_samps[:-1] = slicer_samps[1:]
                    slicer_samps[-1] = sum_out
                    n_slicer_samps += 1
                    if n_slicer_samps >= agc_n_ave:
                        ave_slicer_samps = mean(abs(slicer_samps))
                        # Shift in new average.
                        ave_samps[:-1] = ave_samps[1:]
                        ave_samps[-1] = ave_slicer_samps
                        n_ave_samps += 1
                        if n_ave_samps >= agc_n_ave:
                            ave_ave_samps = mean(ave_samps)
                            match self.mod_type:
                                case 0:
                                    decision_scaler = ave_ave_samps         # type: ignore
                                case 1:
                                    decision_scaler = 1.5 * ave_ave_samps   # type: ignore
                                case 2:
                                    decision_scaler = 1.5 * ave_ave_samps   # type: ignore
                                case _:
                                    raise RuntimeError("Unrecognized modulation type!")
                            scalar_values.append(decision_scaler)
                            self.decision_scaler = decision_scaler
                            self.update_thresholds()
            ui_ests.append(ui)
            lockeds.append(locked)
            smpl_cntr += 1

        self.ui = ui
        if dbg_dict is not None:
            dbg_dict["scalar_values"] = scalar_values

        return (array(res), tap_weights, array(ui_ests), clocks, lockeds, clock_times, array(bits))
