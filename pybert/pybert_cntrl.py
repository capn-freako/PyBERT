# Default controller definition for PyBERT class.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)
#
# Copyright (c) 2014 David Banas; all rights reserved World wide.

from numpy        import sign, sin, pi, array, linspace, float, zeros, repeat, where, diff
from numpy.random import normal
from scipy.signal import lfilter
from dfe          import DFE
from cdr          import CDR
import time

def get_chnl_in(self):
    """Generates the channel input, including any user specified jitter."""

    bits    = self.bits
    nspb    = self.nspb
    fs      = self.fs
    rj      = self.rj * 1.e-12
    sj_mag  = self.sj_mag * 1.e-12
    sj_freq = self.sj_freq * 1.e6
    t       = self.t
    ui      = self.ui * 1.e-12

    ts      = 1. / fs

    res                          = repeat(2 * array(bits) - 1, nspb)
    self.crossing_times_ideal    = find_crossing_times(t, res)
    self.crossing_times_ideal_ns = array(self.crossing_times_ideal) * 1.e9

    jitter = [sj_mag * sin(2 * pi * sj_freq * i * ui) + normal(0., rj) for i in range(len(bits) - 1)]
    i = 1
    for jit in jitter:
        if(jit < -ui):
            jit = -ui
        if(jit > ui):
            jit = ui
        if(jit < 0.):
            res[i * nspb + int(jit / ts - 0.5) : i * nspb] = res[i * nspb]
        else:
            res[i * nspb : i * nspb + int(jit / ts + 0.5)] = res[i * nspb - 1]
        i += 1
    self.crossing_times_chnl_in = find_crossing_times(t, res)

    ## Introduce a 1/2 UI phase shift, half way through the sequence, to test CDR adaptation.
    ##res = res[:len(res)/2 - nspb/2] + res[len(res)/2:] + res[len(res)/2 - nspb/2 : len(res)/2]

    return res

def find_crossing_times(t, x, anlg=False):
    """
    Finds the zero crossing times of the input signal.

    Inputs:

      - t     Vector of sample times. Intervals do NOT need to be uniform.

      - x     Sampled input vector.

      - anlg  Interpolation flag. When TRUE, use linear interpolation,
              in order to determine zero crossing times more precisely.
    """

    assert len(t) == len(x), "len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x))

    crossing_indeces     = where(diff(sign(x)))[0] + 1
    if(anlg):
        crossing_times   = array([t[i - 1] + (t[i] - t[i - 1]) * x[i - 1] / (x[i - 1] - x[i])
                                   for i in crossing_indeces])
    else:
        crossing_times   = [t[i] for i in crossing_indeces]
    return crossing_times

def my_run_dfe(self):
    start_time = time.clock()
    chnl_out        = self.chnl_out
    t               = self.t
    delta_t         = self.delta_t          # (ps)
    alpha           = self.alpha
    ui              = self.ui               # (ps)
    nbits           = self.nbits
    nspb            = self.nspb
    n_taps          = self.n_taps
    gain            = self.gain
    n_ave           = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave      = self.n_lock_ave
    rel_lock_tol    = self.rel_lock_tol
    lock_sustain    = self.lock_sustain
    dfe             = DFE(n_taps, gain, delta_t * 1.e-12, alpha, ui * 1.e-12, decision_scaler,
                          n_ave, n_lock_ave, rel_lock_tol, lock_sustain)

    (res, tap_weights, ui_ests, clocks, lockeds) = dfe.run(t, chnl_out)

    self.run_result = res
    self.adaptation = tap_weights
    self.ui_ests    = array(ui_ests) * 1.e12 # (ps)
    self.clocks     = clocks
    self.lockeds    = lockeds
    self.dfe_perf   = nbits * nspb / (time.clock() - start_time)

def my_run_cdr(self):
    start_time = time.clock()
    chnl_out      = self.chnl_out
    delta_t       = self.delta_t
    alpha         = self.alpha
    n_lock_ave    = self.n_lock_ave
    rel_lock_tol  = self.rel_lock_tol
    lock_sustain  = self.lock_sustain
    ui            = self.ui
    nbits         = self.nbits
    nspb          = self.nspb

    cdr                 = CDR(delta_t, alpha, ui, n_lock_ave, rel_lock_tol, lock_sustain)
    smpl_time           = ui / nspb
    t = next_bndry_time = 0.
    next_clk_time       = ui / 2.
    last_clk_smpl       = 1
    ui_est              = ui
    ui_ests             = []
    locked              = False
    lockeds             = []
    clocks              = zeros(len(chnl_out))
    clk_ind             = 0
    for smpl in chnl_out:
        if(t >= next_bndry_time):
            last_bndry_smpl  = sign(smpl)
            next_bndry_time += ui_est
        if(t >= next_clk_time):
            clocks[clk_ind] = 1
            (ui_est, locked) = cdr.adapt([last_clk_smpl, last_bndry_smpl, sign(smpl)])
            last_clk_smpl   = sign(smpl)
            next_bndry_time = next_clk_time + ui_est / 2.
            next_clk_time  += ui_est
        ui_ests.append(ui_est)
        lockeds.append(locked)
        t       += smpl_time
        clk_ind += 1

    self.clocks  = clocks
    self.ui_ests = ui_ests
    self.lockeds = lockeds
    self.cdr_perf = nbits * nspb / (time.clock() - start_time)

def my_run_channel(self):
    start_time = time.clock()
    chnl_in = get_chnl_in(self)
    a       = self.a
    b       = self.b
    nbits   = self.nbits
    nspb    = self.nspb

    res     = lfilter(b, a, chnl_in)[:len(chnl_in)]

    self.chnl_in  = chnl_in
    self.chnl_out = res
    self.channel_perf = nbits * nspb / (time.clock() - start_time)

# Plot updating
def update_results(self):
    # Copy globals into local namespace.
    ui            = self.ui * 1.e-12
    samps_per_bit = self.nspb
    eye_bits      = self.eye_bits
    num_bits      = self.nbits
    dfe_output    = self.run_result
    clocks        = self.clocks
    # Adjust the scaling.
    width    = 2 * samps_per_bit
    height   = 100
    y_max    = 1.1 * max(abs(dfe_output))
    y_scale  = height / (2 * y_max)          # (pixels/V)
    y_offset = height / 2                    # (pixels)
    x_scale  = width  / (2. * samps_per_bit) # (pixels/sample)
    # Do the plotting.
    # - composite eye "heat" diagram
    img_array    = zeros([height, width])
    tsamp        = ui / samps_per_bit
    for clock_index in where(clocks[-eye_bits * samps_per_bit:])[0] + len(clocks) - eye_bits * samps_per_bit:
        start = clock_index
        stop  = start + 2 * samps_per_bit
        i = 0.
        for samp in dfe_output[start : stop]:
            img_array[int(samp * y_scale) + y_offset, int(i * x_scale)] += 1
            i += 1
    self.plotdata.set_data("imagedata", img_array)
    xs = linspace(-ui * 1.e12, ui * 1.e12, width)
    ys = linspace(-y_max, y_max, height)
    self.plot_eye.components[0].components[0].index.set_data(xs, ys)
    self.plot_eye.components[0].x_axis.mapper.range.low = xs[0]
    self.plot_eye.components[0].x_axis.mapper.range.high = xs[-1]
    self.plot_eye.components[0].y_axis.mapper.range.low = ys[0]
    self.plot_eye.components[0].y_axis.mapper.range.high = ys[-1]
    self.plot_eye.components[0].invalidate_draw()
    # - zero crossing probability density function
    zero_xing_pdf = array(map(float, img_array[y_offset]))
    zero_xing_pdf *= 2. / zero_xing_pdf.sum()
    zero_xing_cdf = zero_xing_pdf.cumsum()
    bathtub_curve = abs(zero_xing_cdf - 1.)
    self.plotdata.set_data("zero_xing_pdf", zero_xing_pdf)
    self.plotdata.set_data("bathtub", bathtub_curve)
    self.plotdata.set_data("eye_index", xs)
    self.plot_eye.components[1].invalidate_draw()
    self.plot_eye.components[2].invalidate_draw()
    # - container redraw
    self.plot_eye.request_redraw()
        
