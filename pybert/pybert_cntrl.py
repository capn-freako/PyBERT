# Default controller definition for PyBERT class.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)
#
# Copyright (c) 2014 David Banas; all rights reserved World wide.

from numpy        import sign, sin, pi, array, linspace, float, zeros, ones, repeat, where, diff, log10
from numpy.random import normal
from numpy.fft    import fft
from scipy.signal import lfilter, iirfilter
from dfe          import DFE
from cdr          import CDR
import time
from pylab import *
from pybert_util import *

debug = False

def get_chnl_in(self):
    """Generates the channel input."""

    nbits   = self.nbits
    nspb    = self.nspb
    fs      = self.fs
    rn      = self.rn
    pn_mag  = self.pn_mag
    pn_freq = self.pn_freq * 1.e6
    t       = self.t
    ui      = self.ui * 1.e-12
    z0      = self.z0
    cout    = self.cout * 1.e-12
    pattern_len = self.pattern_len

    ts      = 1. / fs

    # Generate the ideal over-sampled signal.
    bits        = resize(array([0, 1] + [randint(2) for i in range(pattern_len - 2)]), nbits)
    res         = repeat(2 * bits - 1, nspb)
    ideal_xings = find_crossing_times(t, res, anlg=False)
    # Filter it.
    if(False): # Temporarily disabled, while I debug the excessive jitter coming out of the DFE.
        fc     = 1./(pi * z0 * cout)
        (b, a) = iirfilter(2, fc/(fs/2), btype='lowpass')
        res    = lfilter(b, a, res)[:len(res)]
        # Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
        # - Generate the ideal rectangular aggressor waveform.
        pn_period          = 1. / pn_freq
        pn_samps           = int(pn_period / ts + 0.5)
        pn                 = zeros(pn_samps)
        pn[pn_samps // 2:] = 1
        pn                 = resize(pn, len(res))
        # - High pass filter it. (Simulating capacitive coupling.)
        (b, a) = iirfilter(2, fc/(fs/2), btype='highpass')
        pn     = lfilter(b, a, pn)[:len(pn)]
        # Add the uncorrelated periodic and the random noise to the Tx output.
        res += pn + normal(scale=rn, size=(len(res),))
    # Calculate the jitter.
    tx_xings = find_crossing_times(t, res)
    if(tx_xings[0] < ui / 2.):
        tx_xings = tx_xings[1:]
    (jitter, t_jitter, isi, dcd, pj, rj, tie_ind) = calc_jitter(ui, nbits, pattern_len, ideal_xings, tx_xings)

    self.ideal_xings = ideal_xings
    self.tx_xings    = tx_xings
    self.isi_tx      = isi
    self.dcd_tx      = dcd
    self.pj_tx       = pj
    self.rj_tx       = rj
    self.tie_ind_tx  = tie_ind
    self.bits        = bits

    return res

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

    (res, tap_weights, ui_ests, clocks, lockeds, clock_times) = dfe.run(t, chnl_out) # Run the DFE on the input signal.
    assert len(filter(lambda x: x == None, res)) == 0, "len(t): %d, len(chnl_out): %d\nchnl_out:" % (len(t), len(chnl_out))

    self.adaptation = tap_weights
    self.ui_ests    = array(ui_ests) * 1.e12 # (ps)
    self.clocks     = clocks
    self.lockeds    = lockeds
    self.clock_times = clock_times
    self.run_result = res # Must be last, as it triggers calculations, which depend on some of the above.
    self.dfe_perf   = nbits * nspb / (time.clock() - start_time)

# Plot updating
def update_results(self):
    # Copy globals into local namespace.
    ui            = self.ui * 1.e-12
    samps_per_bit = self.nspb
    eye_bits      = self.eye_bits
    num_bits      = self.nbits
    dfe_output    = self.run_result
    clocks        = self.clocks
    clock_times   = self.clock_times
    # Adjust the scaling.
    width    = 2 * samps_per_bit
    height   = 100
    y_max    = 1.1 * max(abs(dfe_output))
    y_scale  = height / (2 * y_max)          # (pixels/V)
    y_offset = height / 2                    # (pixels)
    # Do the plotting.
    # - composite eye "heat" diagram
    img_array    = zeros([height, width])
    tsamp        = ui / samps_per_bit
    i = 0
    ignore_until = (num_bits - eye_bits) * ui
    while(clock_times[i] <= ignore_until):
        i += 1
        assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
    for clock_time in clock_times[i:]:
        start_time = clock_time - ui
        stop_time  = clock_time + ui
        start_ix   = int(start_time / tsamp)
        interp_fac = (start_time - start_ix * tsamp) / tsamp
        last_y     = dfe_output[start_ix]
        i = 0
        for (samp1, samp2) in zip(dfe_output[start_ix : start_ix + 2 * samps_per_bit],
                                  dfe_output[start_ix + 1 : start_ix + 1 + 2 * samps_per_bit]):
            y = samp1 + (samp2 - samp1) * interp_fac
            img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
            if(sign(y) != sign(last_y)): # Trap zero crossings.
                img_array[y_offset, int(i - 1 + y / (y - last_y) + 0.5)] += 1
            last_y = y
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
        
