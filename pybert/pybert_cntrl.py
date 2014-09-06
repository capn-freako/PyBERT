# Default controller definition for PyBERT class.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)
#
# Copyright (c) 2014 David Banas; all rights reserved World wide.

from numpy        import sign, sin, pi, array, linspace, float, zeros, repeat, where, diff, log10
from numpy.random import normal
from numpy.fft    import fft
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

    res              = repeat(2 * array(bits) - 1, nspb)
    self.ideal_xings = find_crossing_times(t, res)

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

    return res

def find_crossing_times(t, x, anlg=False):
    """
    Finds the zero crossing times of the input signal.

    Inputs:

      - t     Vector of sample times. Intervals do NOT need to be uniform, but all values must be non-negative.

      - x     Sampled input vector.

      - anlg  Interpolation flag. When TRUE, use linear interpolation,
              in order to determine zero crossing times more precisely.

    Outputs:

      - xings  The crossing times, where the sign is used to indicate the crossing direction: + = upward, - = downward.

    """

    assert len(t) == len(x), "len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x))
    assert all(t >= 0), "All times must be non-negative, because we use the sign of the output to indicate crossing direction."

    sign_x      = sign(x)
    sign_x      = where(sign_x, sign_x, ones(sign_x)) # "0"s can produce duplicate xings.
    diff_sign_x = diff(sign_x)
    xing_ix     = where(diff_sign_x)[0]
    xing_sgn    = diff_sign_x[xing_ix] / 2
    if(anlg):
        xings = [t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) for i in xing_ix]
    else:
        xings = [t[i] for i in xing_ix]
    return array(xings) * xing_sgn

def calc_jitter_spectrum(t, jitter, ui, nbits):
    """
    Calculate the spectral magnitude estimate of the input jitter samples.

    The trick, here, is creating a uniformly sampled input vector for the FFT operation,
    since the jitter samples are almost certainly not uniformly sampled.
    We do this by simply zero padding the missing samples.

    The output is normalized, so as to produce a value of 'A / ui' at location 'fj',
    when the input jitter contains a component 'A * sin(2*pi*fj + phi)'.

    Inputs:

    - t      : The sample times for the 'jitter' vector.

    - jitter : The input jitter samples.

    - ui     : The nominal unit interval.

    - nbits  : The desired number of unit intervals, in the time domain.

    Output:

    - f      : The frequency ordinate for the spectral magnitude output vector.

    - y      : The spectral magnitude estimates vector.

    """

    assert len(t) == len(jitter)

    half_n         = nbits / 2
    run_lengths    = map(int, diff(t) / ui + 0.5)
    missing        = where(array(run_lengths) > 1)[0]
    for i in missing:
        for j in range(run_lengths[i] - 1):
            jitter.insert(i + 1, 0.)
    if(len(jitter) < nbits):
        jitter.extend([0.] * (nbits - len(jitter)))
    if(len(jitter) > nbits):
        jitter = jitter[:nbits]
    f0  = 1. / (ui * nbits)
    f   = [i * f0 for i in range(half_n)]
    y   = fft(jitter)
    y   = array(abs(y[:half_n])) / half_n / ui
    return (f, y)

def my_run_dfe(self):
    start_time      = time.clock()
    chnl_out        = self.chnl_out
    t               = self.t
    delta_t         = self.delta_t * 1.e-12
    alpha           = self.alpha
    ui              = self.ui * 1.e-12
    nbits           = self.nbits
    nspb            = self.nspb
    n_taps          = self.n_taps
    gain            = self.gain
    n_ave           = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave      = self.n_lock_ave
    rel_lock_tol    = self.rel_lock_tol
    lock_sustain    = self.lock_sustain
    dfe             = DFE(n_taps, gain, delta_t, alpha, ui, nspb, decision_scaler,
                          n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol, lock_sustain=lock_sustain)

    (res, tap_weights, ui_ests, clocks, lockeds) = dfe.run(t, chnl_out) # Run the DFE on the input signal.

    self.adaptation = tap_weights
    self.ui_ests    = array(ui_ests) * 1.e12 # (ps)
    self.clocks     = clocks
    self.lockeds    = lockeds
    self.run_result = res
    self.dfe_perf   = nbits * nspb / (time.clock() - start_time)

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
        prev_samp = dfe_output[start]
        i = 0
        for samp in dfe_output[start : stop]:
            img_array[int(samp * y_scale) + y_offset, int(i * x_scale)] += 1
            if(sign(samp) != sign(prev_samp)): # Trap zero crossings.
                img_array[y_offset, int(x_scale * (i - 1 + (samp - 0.) / (samp - prev_samp)))] += 1
            prev_samp = samp
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
        
