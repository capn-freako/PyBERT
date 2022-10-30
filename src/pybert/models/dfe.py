"""Behavioral model of a decision feedback equalizer (DFE).

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

This Python script provides a behavioral model of a decision feedback
equalizer (DFE). The class defined, here, is intended for integration
into the larger *PyBERT* framework.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
from numpy import array, sign, zeros
from scipy.signal import iirfilter

from pybert.models.cdr import CDR

gNch_taps = 3  # Number of taps used in summing node filter.


class LfilterSS:
    """A single steppable version of scipy.signal.lfilter()."""

    def __init__(self, b, a):
        """
        Args:
            b([float]): Coefficients of the numerator of the rational transfer function.
            a([float]): Coefficients of the denominator of the rational transfer function.
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


class DFE:
    """Behavioral model of a decision feedback equalizer (DFE)."""

    def __init__(
        self,
        n_taps,
        gain,
        delta_t,
        alpha,
        ui,
        n_spb,
        decision_scaler,
        mod_type=0,
        bandwidth=100.0e9,
        n_ave=10,
        n_lock_ave=500,
        rel_lock_tol=0.01,
        lock_sustain=500,
        ideal=True,
    ):
        """
        Inputs:

          Required:

          - n_taps           # of taps in adaptive filter

          - gain             adaptive filter tap weight correction gain

          - delta_t          CDR proportional branch constant (ps)

          - alpha            CDR integral branch constant (normalized to delta_t)

          - ui               nominal unit interval (ps)

          - n_spb            # of samples per unit interval

          - decision_scaler  multiplicative constant applied to the result of
                             the sign function, when making a "1 vs. 0" decision.
                             Sets the target magnitude for the DFE.

          Optional:

          - mod_type         The modulation type:
                             - 0: NRZ
                             - 1: Duo-binary
                             - 2: PAM-4

          - bandwidth        The bandwidth, at the summing node (Hz).

          - n_ave            The number of averages to take, before adapting.
                             (Also, the number of CDR adjustments per DFE adaptation.)

          - n_lock_ave       The number of unit interval estimates to
                             consider, when determining locked status.

          - rel_lock_tol     The relative tolerance for determining lock.

          - lock_sustain     Length of the histerysis vector used for
                             lock flagging.

          - ideal            Boolean flag. When true, use an ideal summing node.

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

        thresholds = []
        if mod_type == 0:  # NRZ
            pass
        elif mod_type == 1:  # Duo-binary
            thresholds.append(-decision_scaler / 2.0)
            thresholds.append(decision_scaler / 2.0)
        elif mod_type == 2:  # PAM-4
            thresholds.append(-decision_scaler * 2.0 / 3.0)
            thresholds.append(0.0)
            thresholds.append(decision_scaler * 2.0 / 3.0)
        else:
            raise RuntimeError("ERROR: DFE.__init__(): Unrecognized modulation type requested!")
        self.thresholds = thresholds

    def step(self, decision, error, update):
        """Step the DFE, according to the new decision and error inputs.

        Args:
            decision(float): Current slicer output.
            error(float): Difference between summing node and slicer outputs.
            update(bool): If true, update tap weights.

        Returns:
            res(float): New backward filter output value.
        """

        # Copy class object variables into local function namespace, for efficiency.
        tap_weights = self.tap_weights
        tap_values = self.tap_values
        gain = self.gain
        n_ave = self.n_ave

        # Calculate this step's corrections and add to running total.
        corrections = [old + new for (old, new) in zip(self.corrections, [val * error * gain for val in tap_values])]

        # Update the tap weights with the average corrections, if appropriate.
        if update:
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

    def decide(self, x):
        """Make the bit decisions, according to modulation type.

        Args:
            x(float): The signal value, at the decision time.

        Returns:
            tuple(float, [int]): The members of the returned tuple are:

                decision:
                    One of:

                        - {-1, 1}              (NRZ)
                        - {-1, 0, +1}          (Duo-binary)
                        - {-1, -1/3, +1/3, +1} (PAM-4)

                        according to what the ideal signal level should have been.
                        ('decision_scaler' normalized)

                bits: The list of bits recovered.

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

    def run(self, sample_times, signal):
        """Run the DFE on the input signal.

        Args:
            sample_times([float]): Vector of time values at wich
                corresponding signal values were sampled.
            signal([float]): Vector of sampled signal values.

        Returns:
            tuple(([float], [[float]], [float], [int], [bool], [float], [int])):
                The members of the returned tuple, in order, are:

                    res([float]):
                        Samples of the summing node output, taken at the
                        times given in *sample_times*.
                    tap_weights([[float]]):
                        List of list of tap weights showing how the DFE
                        adapted over time.
                    ui_ests([float]):
                        List of unit interval estimates, showing how the
                        CDR adapted.
                    clocks([int]):
                        List of mostly zeros with ones at the recovered
                        clocking instants. Useful for overlaying the
                        clock times on signal waveforms, in plots.
                    lockeds([bool]):
                        List of Booleans indicating state of CDR lock.
                    clock_times([float]):
                        List of clocking instants, as recovered by the CDR.
                    bits([int]):
                        List of recovered bits.

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

        clk_cntr = 0
        smpl_cntr = 0
        filter_out = 0
        nxt_filter_out = 0
        last_clock_sample = 0
        next_boundary_time = 0
        next_clock_time = ui / 2.0
        locked = False

        res = []
        tap_weights = [self.tap_weights]
        ui_ests = []
        lockeds = []
        clocks = zeros(len(sample_times))
        clock_times = [next_clock_time]
        bits = []
        for (t, x) in zip(sample_times, signal):
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
                samples = [last_clock_sample, boundary_sample, current_clock_sample]
                if mod_type == 0:  # NRZ
                    pass
                elif mod_type == 1:  # Duo-binary
                    samples = array(samples)
                    if samples.mean() < 0.0:
                        samples -= thresholds[0]
                    else:
                        samples -= thresholds[1]
                    samples = list(samples)
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
                    nxt_filter_out = self.step(decision, 0.0, update)
                tap_weights.append(self.tap_weights)
                last_clock_sample = sum_out
                next_boundary_time = next_clock_time + ui / 2.0
                next_clock_time += ui
                clock_times.append(next_clock_time)
            ui_ests.append(ui)
            lockeds.append(locked)
            smpl_cntr += 1

        self.ui = ui

        return (res, tap_weights, ui_ests, clocks, lockeds, clock_times, bits)
