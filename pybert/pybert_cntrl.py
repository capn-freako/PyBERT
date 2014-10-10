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

gFc = 1.e6 # corner frequency of high-pass filter used to model capacitive coupling of periodic noise.

def my_run_channel(self):
    start_time = time.clock()

    nbits   = self.nbits
    nspb    = self.nspb
    fs      = self.fs
    rn      = self.rn
    pn_mag  = self.pn_mag
    pn_freq = self.pn_freq * 1.e6
    t       = self.t
    Vod     = self.vod
    Rs      = self.rs
    Cs      = self.cout * 1.e-12
    RL      = self.rin
    CL      = self.cac * 1.e-6
    Cp      = self.cin * 1.e-12
    R0      = self.R0
    w0      = self.w0
    Rdc     = self.Rdc
    Z0      = self.Z0
    v0      = self.v0 * 3.e8
    Theta0  = self.Theta0
    w       = self.w
    l_ch    = self.l_ch
    pattern_len = self.pattern_len

    # Generate the impulse response of the channel.
    gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
    H         = exp(-l_ch * gamma)
    G         = calc_G(H, Rs, Cs, Zc, RL, Cp, CL, w)
    g         = 2 * Vod * real(ifft(G))

    # Trim impulse response, in order to shorten convolution processing time, by:
    #  - eliminating 90% of the overall delay from the beginning, and
    #  - clipping off the tail, after 99.9% of the total power has been captured.
    chnl_dly  = l_ch / v0
    Ts        = 1. / fs
    start_ix  = int(0.9 * chnl_dly / Ts)
    Pt        = 0.999 * sum(g ** 2)
    i         = 0
    P         = 0
    while(P < Pt):
        P += g[i] ** 2
        i += 1
    t_ns_chnl = self.t_ns[start_ix : i]
    g         = g[start_ix : i]
    s         = g.cumsum()
    if(False):
        plot(self.t_ns[start_ix : i], g)
        plot(self.t_ns[start_ix : i], s)
        show()

    # Generate the ideal over-sampled signal.
    bits        = resize(array([0, 1] + [randint(2) for i in range(pattern_len - 2)]), nbits)
    x           = repeat(2 * bits - 1, nspb)
    ideal_xings = find_crossing_times(t, x, anlg=False)

    # Filter it.
    y   = convolve(g, x)
    res = y[:len(x)]

    if(True):
        # Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
        # - Generate the ideal rectangular aggressor waveform.
        pn_period          = 1. / pn_freq
        pn_samps           = int(pn_period / Ts + 0.5)
        pn                 = zeros(pn_samps)
        pn[pn_samps // 2:] = pn_mag
        pn                 = resize(pn, len(res))
        # - High pass filter it. (Simulating capacitive coupling.)
        (b, a) = iirfilter(2, gFc/(fs/2), btype='highpass')
        pn     = lfilter(b, a, pn)[:len(pn)]
        # Add the uncorrelated periodic and the random noise to the Tx output.
        res += pn + normal(scale=rn, size=(len(res),))
    
    self.ideal_xings  = ideal_xings
    self.t_ns_chnl    = t_ns_chnl
    self.ch_freq_resp = 20. * log10(2. * abs(G[1 : len(G) // 2]))
    self.ch_f_GHz     = w[1 : len(G) // 2] / (2 * pi) / 1.e9
    self.ch_imp_resp  = g
    self.ch_step_resp = s
    self.chnl_out     = res         # Must come last, since what it triggers requires 'ideal_xings' be defined.

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
    bandwidth       = self.sum_bw * 1.e9
    ideal           = self.sum_ideal

    dfe             = DFE(n_taps, gain, delta_t, alpha, ui, nspb, decision_scaler,
                          n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol, lock_sustain=lock_sustain,
                          bandwidth=bandwidth, ideal=ideal)

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
    """Updates all plot data used by GUI."""

    # Direct transfers.
    self.plotdata.set_data("chnl_out", self.chnl_out)
    self.plotdata.set_data("t_ns", self.t_ns)
    self.plotdata.set_data("t_ns_chnl", self.t_ns_chnl)
    self.plotdata.set_data("ch_imp_resp", self.ch_imp_resp * 1.e-9 * self.fs ) # Need to scale from V/Ts to V/ns.
    self.plotdata.set_data("ch_step_resp", self.ch_step_resp)
    self.plotdata.set_data("ch_freq_resp", self.ch_freq_resp)
    self.plotdata.set_data("ch_f_GHz", self.ch_f_GHz)
    self.plotdata.set_data("clocks", self.clocks)
    self.plotdata.set_data("ui_ests", self.ui_ests)
    self.plotdata.set_data("lockeds", self.lockeds)
    self.plotdata.set_data("jitter", array(self.jitter) * 1.e12)
    self.plotdata.set_data("t_jitter", array(self.t_jitter) * 1.e9)
    self.plotdata.set_data("tie_hist_bins", array(self.tie_hist_bins) * 1.e12)
    self.plotdata.set_data("tie_hist_counts", self.tie_hist_counts)
    self.plotdata.set_data("tie_ind_hist_bins", array(self.tie_ind_hist_bins) * 1.e12)
    self.plotdata.set_data("tie_ind_hist_counts", self.tie_ind_hist_counts)
    self.plotdata.set_data("tie_hist_bins_rx", array(self.tie_hist_bins_rx) * 1.e12)
    self.plotdata.set_data("tie_hist_counts_rx", self.tie_hist_counts_rx)
    self.plotdata.set_data("tie_ind_hist_bins_rx", array(self.tie_ind_hist_bins_rx) * 1.e12)
    self.plotdata.set_data("tie_ind_hist_counts_rx", self.tie_ind_hist_counts_rx)
    self.plotdata.set_data("jitter_spectrum", self.jitter_spectrum[1:])
    self.plotdata.set_data("tie_ind_spectrum_chnl", self.tie_ind_spectrum_chnl[1:])
    self.plotdata.set_data("tie_ind_spectrum_rx", self.tie_ind_spectrum_rx[1:])
    self.plotdata.set_data("f_MHz", self.f_MHz[1:])
    self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])
    self.plotdata.set_data("jitter_rx", self.jitter_rx[1:] * 1.e12)
    self.plotdata.set_data("t_jitter_rx", array(self.t_jitter_rx) * 1.e9)
    self.plotdata.set_data("tie_ind_rx", self.tie_ind_rx[1:] * 1.e12)
    self.plotdata.set_data("jitter_spectrum_rx", self.jitter_spectrum_rx[1:])
    self.plotdata.set_data("tie_ind_spectrum_rx", self.tie_ind_spectrum_rx[1:])
    self.plotdata.set_data("dfe_out", self.run_result)
    # DFE tap weights.
    tap_weights = transpose(array(self.adaptation))
    i = 1
    for tap_weight in tap_weights:
        self.plotdata.set_data("tap%d_weights" % i, tap_weight)
        i += 1
    self.plotdata.set_data("tap_weight_index", range(len(tap_weight)))

    # Calculated results.
    #  - Copy globals into local namespace.
    ui            = self.ui * 1.e-12
    samps_per_bit = self.nspb
    eye_bits      = self.eye_bits
    num_bits      = self.nbits
    chnl_output   = self.chnl_out
    dfe_output    = self.run_result
    clocks        = self.clocks
    clock_times   = self.clock_times
    #  - Adjust the scaling.
    width    = 2 * samps_per_bit
    height   = 100
    xs       = linspace(-ui * 1.e12, ui * 1.e12, width)
    y_max    = 1.1 * max(abs(dfe_output))
    y_offset = height / 2                    # (pixels)
    # - Composite eye "heat" diagrams.
    eye_rx_in = calc_eye(ui, samps_per_bit, height, chnl_output[samps_per_bit // 2 :])
    self.plotdata.set_data("eye_rx_in", eye_rx_in)
    i = 0
    ignore_until = (num_bits - eye_bits) * ui
    while(clock_times[i] <= ignore_until):
        i += 1
        assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
    eye_rx_out = calc_eye(ui, samps_per_bit, height, dfe_output, clock_times[i:])
    self.plotdata.set_data("eye_rx_out", eye_rx_out)
    # - Zero crossing probability density functions.
    zero_xing_pdf_rx_in  = array(map(float, eye_rx_in[y_offset]))
    zero_xing_pdf_rx_in *= 2. / zero_xing_pdf_rx_in.sum()
    zero_xing_cdf_rx_in  = zero_xing_pdf_rx_in.cumsum()
    bathtub_curve_rx_in  = abs(zero_xing_cdf_rx_in - 1.)
    zero_xing_pdf_rx_out  = array(map(float, eye_rx_out[y_offset]))
    zero_xing_pdf_rx_out *= 2. / zero_xing_pdf_rx_out.sum()
    zero_xing_cdf_rx_out  = zero_xing_pdf_rx_out.cumsum()
    bathtub_curve_rx_out  = abs(zero_xing_cdf_rx_out - 1.)
    self.plotdata.set_data("zero_xing_pdf_rx_in", zero_xing_pdf_rx_in)
    self.plotdata.set_data("bathtub_rx_in", bathtub_curve_rx_in)
    self.plotdata.set_data("zero_xing_pdf_rx_out", zero_xing_pdf_rx_out)
    self.plotdata.set_data("bathtub_rx_out", bathtub_curve_rx_out)
    self.plotdata.set_data("eye_index", xs)
        
def update_eyes(self):
    ui            = self.ui * 1.e-12
    samps_per_bit = self.nspb
    dfe_output    = self.run_result

    width    = 2 * samps_per_bit
    height   = 100
    y_max    = 1.1 * max(abs(dfe_output))
    xs       = linspace(-ui * 1.e12, ui * 1.e12, width)
    ys       = linspace(-y_max, y_max, height)

    self.plot_eye.components[0].components[0].index.set_data(xs, ys)
    self.plot_eye.components[0].x_axis.mapper.range.low = xs[0]
    self.plot_eye.components[0].x_axis.mapper.range.high = xs[-1]
    self.plot_eye.components[0].y_axis.mapper.range.low = ys[0]
    self.plot_eye.components[0].y_axis.mapper.range.high = ys[-1]
    self.plot_eye.components[0].invalidate_draw()
    self.plot_eye.components[1].components[0].index.set_data(xs, ys)
    self.plot_eye.components[1].x_axis.mapper.range.low = xs[0]
    self.plot_eye.components[1].x_axis.mapper.range.high = xs[-1]
    self.plot_eye.components[1].y_axis.mapper.range.low = ys[0]
    self.plot_eye.components[1].y_axis.mapper.range.high = ys[-1]
    self.plot_eye.components[1].invalidate_draw()
    self.plot_eye.request_redraw()

def my_run_simulation(self):
    start_time = time.clock()

    nbits = self.nbits
    nspb  = self.nspb

    my_run_channel(self)
    if(self.use_dfe):
        self.status = 'Running DFE...'
        my_run_dfe(self)
    self.status = 'Calculating results...'
    update_results(self)
    update_eyes(self)

    self.total_perf   = nbits * nspb / (time.clock() - start_time)
    
