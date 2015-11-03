"""
Default controller definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>
Original date:   August 24, 2014 (Copied from `pybert.py', as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from time         import clock, sleep

from numpy        import sign, sin, pi, array, linspace, zeros, ones, repeat, where, sqrt
from numpy        import diff, log10, correlate, convolve, mean, resize, real, transpose, cumsum
from numpy.random import normal
from numpy.fft    import fft, ifft
from scipy.signal import lfilter, iirfilter, freqz, fftconvolve

from dfe          import DFE
from cdr          import CDR

from pybert_util  import find_crossings, trim_shift_scale, make_ctle, calc_jitter, moving_average, calc_eye

DEBUG           = False
MIN_BATHTUB_VAL = 1.e-18

gFc     = 1.e6     # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.

def my_run_sweeps(self):
    """
    Runs the simulation sweeps.

    """

    pretap        = self.pretap
    pretap_sweep  = self.pretap_sweep
    pretap_steps  = self.pretap_steps
    pretap_final  = self.pretap_final
    posttap       = self.posttap
    posttap_sweep = self.posttap_sweep  
    posttap_steps = self.posttap_steps
    posttap_final = self.posttap_final
    posttap2       = self.posttap2
    posttap2_sweep = self.posttap2_sweep  
    posttap2_steps = self.posttap2_steps
    posttap2_final = self.posttap2_final
    posttap3       = self.posttap3
    posttap3_sweep = self.posttap3_sweep  
    posttap3_steps = self.posttap3_steps
    posttap3_final = self.posttap3_final
    sweep_aves    = self.sweep_aves
    do_sweep      = self.do_sweep

    if(do_sweep):
        # Assemble the list of desired values for each sweepable parameter.
        pretap_vals  = [pretap]
        posttap_vals = [posttap]
        posttap2_vals = [posttap2]
        posttap3_vals = [posttap3]
        if(pretap_sweep):
            pretap_step = (pretap_final - pretap) / pretap_steps
            pretap_vals.extend([pretap + (i + 1) * pretap_step for i in range(pretap_steps)])
        if(posttap_sweep):
            posttap_step = (posttap_final - posttap) / posttap_steps
            posttap_vals.extend([posttap + (i + 1) * posttap_step for i in range(posttap_steps)])
        if(posttap2_sweep):
            posttap2_step = (posttap2_final - posttap2) / posttap2_steps
            posttap2_vals.extend([posttap2 + (i + 1) * posttap2_step for i in range(posttap2_steps)])
        if(posttap3_sweep):
            posttap3_step = (posttap3_final - posttap3) / posttap3_steps
            posttap3_vals.extend([posttap3 + (i + 1) * posttap3_step for i in range(posttap3_steps)])

        # Run the sweep, using the lists assembled, above.
        sweeps          = [(pretap_vals[l], posttap_vals[k], posttap2_vals[j], posttap3_vals[i])
                for i in range(len(posttap3_vals))
                for j in range(len(posttap2_vals))
                for k in range(len(posttap_vals))
                for l in range(len(pretap_vals))
                ]
        num_sweeps      = sweep_aves * len(sweeps)
        self.num_sweeps = num_sweeps
        sweep_results   = []
        sweep_num       = 1
        for (pretap_val, posttap_val, posttap2_val, posttap3_val) in sweeps:
            self.pretap    = pretap_val
            self.posttap   = posttap_val
            self.posttap2  = posttap2_val
            self.posttap3  = posttap3_val
            bit_errs       = []
            for i in range(sweep_aves):
                self.sweep_num = sweep_num
                my_run_simulation(self, update_plots=False)
                bit_errs.append(self.bit_errs)
                sweep_num += 1
            sweep_results.append((pretap_val, posttap_val, mean(bit_errs), std(bit_errs)))
        self.sweep_results = sweep_results
    else:
        my_run_simulation(self)

def my_run_simulation(self, initial_run=False, update_plots=True):
    """
    Runs the simulation.

    Inputs:

      - initial_run     If True, don't update the eye diagrams, since they haven't been created, yet.
                        (Optional; default = False.)

    """

    num_sweeps = self.num_sweeps
    sweep_num  = self.sweep_num

    start_time = clock()
    self.status = 'Running channel...(sweep %d of %d)' % (sweep_num, num_sweeps)

    self.run_count += 1  # Force regeneration of bit stream.

    # Pull class variables into local storage, performing unit conversion where necessary.
    t               = self.t
    t_ns            = self.t_ns
    f               = self.f
    w               = self.w
    bits            = self.bits
    symbols         = self.symbols
    ffe             = self.ffe
    nbits           = self.nbits
    nui             = self.nui
    bit_rate        = self.bit_rate * 1.e9
    eye_bits        = self.eye_bits
    eye_uis         = self.eye_uis
    nspb            = self.nspb
    nspui           = self.nspui
    rn              = self.rn
    pn_mag          = self.pn_mag
    pn_freq         = self.pn_freq * 1.e6
    Vod             = self.vod
    Rs              = self.rs
    Cs              = self.cout * 1.e-12
    RL              = self.rin
    CL              = self.cac * 1.e-6
    Cp              = self.cin * 1.e-12
    R0              = self.R0
    w0              = self.w0
    Rdc             = self.Rdc
    Z0              = self.Z0
    v0              = self.v0 * 3.e8
    Theta0          = self.Theta0
    l_ch            = self.l_ch
    pretap          = self.pretap
    posttap         = self.posttap
    posttap2        = self.posttap2
    posttap3        = self.posttap3
    pattern_len     = self.pattern_len
    rx_bw           = self.rx_bw * 1.e9
    peak_freq       = self.peak_freq * 1.e9
    peak_mag        = self.peak_mag
    delta_t         = self.delta_t * 1.e-12
    alpha           = self.alpha
    ui              = self.ui
    n_taps          = self.n_taps
    gain            = self.gain
    n_ave           = self.n_ave
    decision_scaler = self.decision_scaler
    n_lock_ave      = self.n_lock_ave
    rel_lock_tol    = self.rel_lock_tol
    lock_sustain    = self.lock_sustain
    bandwidth       = self.sum_bw * 1.e9
    rel_thresh      = self.thresh
    mod_type        = self.mod_type[0]
    ideal_h         = self.ideal_h

    # Correct pattern length, if using PAM-4.
    if(mod_type == 2):
        pattern_len = pattern_len * 2 // 2  # Catches the case of odd number of bits in pattern.

    # Calculate misc. values.
    fs         = bit_rate * nspb
    Ts         = t[1]

    # Generate the ideal over-sampled signal.
    #
    # Duo-binary is problematic, in that it requires convolution with the ideal duobinary
    # impulse response, in order to produce the proper ideal signal.
    #
    # Note that we don't use the ideal duobinary signal thusly constructed for determining
    # the ideal crossing locations, below, because it has "shelves" at zero volts, which
    # wreak havoc on our zero crossing detection algorithm.
    x = repeat(symbols, nspui)
    tmp_x = x
    if(mod_type == 1):
        duob_h = array(([0.5] + [0.] * (nspui - 1)) * 2)
        tmp_x  = convolve(tmp_x, duob_h)[:len(t)]
    self.ideal_signal = tmp_x

    # Find the ideal crossing times, for subsequent jitter analysis of transmitted signal.
    ideal_xings      = find_crossings(t, x, decision_scaler, min_delay = ui / 2., mod_type = mod_type)
    self.ideal_xings = ideal_xings

    # Generate the ideal impulse responses.
    chnl_h       = self.calc_chnl_h()
    self.chnl_g  = trim_shift_scale(ideal_h, chnl_h)

    # Calculate the channel output.
    #
    # Note: We're not using 'tmp_x', because we rely on the system response to
    #       create the duobinary waveform. We only create it explicitly, above,
    #       so that we'll have an ideal reference for comparison.
    chnl_out  = convolve(x, chnl_h)[:len(x)]

    self.channel_perf = nbits * nspb / (clock() - start_time)
    split_time        = clock()
    self.status       = 'Running Tx...(sweep %d of %d)' % (sweep_num, num_sweeps)

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the Tx.
    #
    # - Generate the ideal, post-preemphasis signal.
    # To consider: use 'scipy.interp()'. This is what Mark does, in order to induce jitter in the Tx output.
    ffe_out           = convolve(symbols, ffe)[:len(symbols)]
    self.rel_power    = mean(ffe_out ** 2)                    # Store the relative average power dissipated in the Tx.
    tx_out            = repeat(ffe_out, nspui)                # oversampled output

    # - Calculate the responses.
    # - (The Tx is unique in that the calculated responses aren't used to form the output.
    #    This is partly due to the out of order nature in which we combine the Tx and channel,
    #    and partly due to the fact that we're adding noise to the Tx output.)
    tx_h   = array(sum([[x] + list(zeros(nspui - 1)) for x in ffe], []))  # Using sum to concatenate.
    tx_h.resize(len(chnl_h))
    temp   = tx_h.copy()
    temp.resize(len(w))
    tx_H   = fft(temp)
    tx_H  *= sum(ffe) / abs(tx_H[0])  # Normalize for proper d.c. magnitude.

    # - Generate the uncorrelated periodic noise. (Assume capacitive coupling.)
    #   - Generate the ideal rectangular aggressor waveform.
    pn_period          = 1. / pn_freq
    pn_samps           = int(pn_period / Ts + 0.5)
    pn                 = zeros(pn_samps)
    pn[pn_samps // 2:] = pn_mag
    pn                 = resize(pn, len(tx_out))
    #   - High pass filter it. (Simulating capacitive coupling.)
    (b, a) = iirfilter(2, gFc/(fs/2), btype='highpass')
    pn     = lfilter(b, a, pn)[:len(pn)]

    # - Add the uncorrelated periodic noise to the Tx output.
    tx_out += pn

    # - Convolve w/ channel.
    tx_out_h   = convolve(tx_h, chnl_h)[:len(chnl_h)]
    tx_out_g   = trim_shift_scale(ideal_h, tx_out_h)
    temp       = tx_out_h.copy()
    temp.resize(len(w))
    tx_out_H   = fft(temp)
#    tx_out_H  *= sum(ffe) / abs(tx_out_H[0])  # Normalize for proper d.c. magnitude.
    rx_in      = convolve(tx_out, chnl_h)[:len(tx_out)]

    # - Add the random noise to the Rx input.
    rx_in     += normal(scale=rn, size=(len(tx_out),))

    self.tx_s      = tx_h.cumsum()
    self.tx_out    = tx_out
    self.rx_in     = rx_in
    self.tx_out_s  = tx_out_h.cumsum()
    self.tx_out_p  = self.tx_out_s[nspui:] - self.tx_out_s[:-nspui] 
    self.tx_H      = tx_H
    self.tx_h      = tx_h
    self.tx_out_H  = tx_out_H
    self.tx_out_h  = tx_out_h
    self.tx_out_g  = tx_out_g

    self.tx_perf   = nbits * nspb / (clock() - split_time)
    split_time     = clock()
    self.status    = 'Running CTLE...(sweep %d of %d)' % (sweep_num, num_sweeps)

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the CTLE.
    w_dummy, H      = make_ctle(rx_bw, peak_freq, peak_mag, w)
    ctle_H          = H
    ctle_h          = real(ifft(ctle_H))[:len(chnl_h)]
    ctle_h         *= abs(ctle_H[0]) / sum(ctle_h)
    ctle_out        = convolve(rx_in, ctle_h)[:len(tx_out)]
    ctle_out       -= mean(ctle_out)             # Force zero mean.
    if(self.use_agc):                            # Automatic gain control engaged?
        ctle_out   *= 2. * decision_scaler / ctle_out.ptp()
    self.ctle_s     = ctle_h.cumsum()
    ctle_out_h      = convolve(tx_out_h, ctle_h)[:len(tx_out_h)]
    ctle_out_g      = trim_shift_scale(ideal_h, ctle_out_h)
    conv_dly_ix     = where(ctle_out_h == max(ctle_out_h))[0][0]
    conv_dly        = t[conv_dly_ix]
    ctle_out_s      = ctle_out_h.cumsum()
    temp            = ctle_out_h.copy()
    temp.resize(len(w))
    ctle_out_H      = fft(temp)
#    ctle_out_H     *= sum(temp) / abs(ctle_out_H[0])
    # - Store local variables to class instance.
    self.ctle_out_s = ctle_out_s
    self.ctle_out_p = self.ctle_out_s[nspui:] - self.ctle_out_s[:-nspui] 
    self.ctle_H     = ctle_H
#    self.ctle_h     = ctle_h * 1.e-9 / Ts
    self.ctle_h     = ctle_h
    self.ctle_out_H = ctle_out_H
#    self.ctle_out_h = ctle_out_h * 1.e-9 / Ts
    self.ctle_out_h = ctle_out_h
    self.ctle_out_g = ctle_out_g
    self.ctle_out   = ctle_out
    self.conv_dly   = conv_dly
    self.conv_dly_ix = conv_dly_ix

    self.ctle_perf  = nbits * nspb / (clock() - split_time)
    split_time      = clock()
    self.status     = 'Running DFE/CDR...(sweep %d of %d)' % (sweep_num, num_sweeps)

    # Generate the output from, and the incremental/cumulative impulse/step/frequency responses of, the DFE.
    if(self.use_dfe):
        dfe = DFE(n_taps, gain, delta_t, alpha, ui, nspui, decision_scaler, mod_type,
                    n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol, lock_sustain=lock_sustain,
                    bandwidth=bandwidth, ideal=self.sum_ideal)
    else:
        dfe = DFE(n_taps,   0., delta_t, alpha, ui, nspui, decision_scaler, mod_type,
                    n_ave=n_ave, n_lock_ave=n_lock_ave, rel_lock_tol=rel_lock_tol, lock_sustain=lock_sustain,
                    bandwidth=bandwidth, ideal=True)
    (dfe_out, tap_weights, ui_ests, clocks, lockeds, clock_times, bits_out) = dfe.run(t, ctle_out)
    bits_out = array(bits_out)
    auto_corr       = 1. * correlate(bits_out[(nbits - eye_bits):], bits[(nbits - eye_bits):], mode='same') / sum(bits[(nbits - eye_bits):])
    auto_corr       = auto_corr[len(auto_corr) // 2 :]
    self.auto_corr  = auto_corr
    bit_dly         = where(auto_corr == max(auto_corr))[0][0]
    n_extra         = len(bits) - len(bits_out)
    bit_errs        = where(bits_out[(nbits - eye_bits + bit_dly):] ^ bits[(nbits - eye_bits) : len(bits_out) - bit_dly])[0]
    self.bit_errs   = len(bit_errs)

    dfe_h          = array([1.] + list(zeros(nspb - 1)) + sum([[-x] + list(zeros(nspb - 1)) for x in tap_weights[-1]], []))
    dfe_h.resize(len(ctle_out_h))
    temp           = dfe_h.copy()
    temp.resize(len(w))
    dfe_H          = fft(temp)
    self.dfe_s     = dfe_h.cumsum()
    dfe_out_H      = ctle_out_H * dfe_H
    dfe_out_h      = convolve(ctle_out_h, dfe_h)[:len(ctle_out_h)]
    dfe_out_g      = trim_shift_scale(ideal_h, dfe_out_h)
    self.dfe_out_s = dfe_out_h.cumsum()
    self.dfe_out_p = self.dfe_out_s[nspui:] - self.dfe_out_s[:-nspui] 
    self.dfe_H     = dfe_H
    self.dfe_h     = dfe_h
    self.dfe_out_H = dfe_out_H
#    self.dfe_out_h = dfe_out_h * 1.e-9 / Ts
    self.dfe_out_h = dfe_out_h
    self.dfe_out_g = dfe_out_g
    self.dfe_out   = dfe_out

    self.dfe_perf  = nbits * nspb / (clock() - split_time)
    split_time     = clock()
    self.status    = 'Analyzing jitter...(sweep %d of %d)' % (sweep_num, num_sweeps)

    # Analyze the jitter.
    # - channel output
    try:
        actual_xings = find_crossings(t, chnl_out, decision_scaler, mod_type = mod_type)
        (jitter, t_jitter, isi, dcd, pj, rj, jitter_ext, \
            thresh, jitter_spectrum, jitter_ind_spectrum, spectrum_freqs, \
            hist, hist_synth, bin_centers) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.t_jitter                 = t_jitter
        self.isi_chnl                 = isi
        self.dcd_chnl                 = dcd
        self.pj_chnl                  = pj
        self.rj_chnl                  = rj
        self.thresh_chnl              = thresh
        self.jitter_chnl              = hist
        self.jitter_ext_chnl          = hist_synth
        self.jitter_bins              = bin_centers
        self.jitter_spectrum_chnl     = jitter_spectrum
        self.jitter_ind_spectrum_chnl = jitter_ind_spectrum
        self.f_MHz                    = array(spectrum_freqs) * 1.e-6
    except:
        pass
    # - Tx output
    try:
        actual_xings = find_crossings(t, rx_in, decision_scaler, mod_type = mod_type)
        (jitter, t_jitter, isi, dcd, pj, rj, jitter_ext, \
            thresh, jitter_spectrum, jitter_ind_spectrum, spectrum_freqs, \
            hist, hist_synth, bin_centers) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_tx                 = isi
        self.dcd_tx                 = dcd
        self.pj_tx                  = pj
        self.rj_tx                  = rj
        self.thresh_tx              = thresh
        self.jitter_tx              = hist
        self.jitter_ext_tx          = hist_synth
        self.jitter_spectrum_tx     = jitter_spectrum
        self.jitter_ind_spectrum_tx = jitter_ind_spectrum
    except:
        self.thresh_tx              = array([])
        self.jitter_ext_tx          = array([])
        self.jitter_tx              = array([])
        self.jitter_spectrum_tx     = array([])
        self.jitter_ind_spectrum_tx = array([])
    # - CTLE output
    try:
        actual_xings = find_crossings(t, ctle_out, decision_scaler, mod_type = mod_type)
        (jitter, t_jitter, isi, dcd, pj, rj, jitter_ext, \
            thresh, jitter_spectrum, jitter_ind_spectrum, spectrum_freqs, \
            hist, hist_synth, bin_centers) = calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_ctle                 = isi
        self.dcd_ctle                 = dcd
        self.pj_ctle                  = pj
        self.rj_ctle                  = rj
        self.thresh_ctle              = thresh
        self.jitter_ctle              = hist
        self.jitter_ext_ctle          = hist_synth
        self.jitter_spectrum_ctle     = jitter_spectrum
        self.jitter_ind_spectrum_ctle = jitter_ind_spectrum
    except:
        self.thresh_ctle              = array([])
        self.jitter_ext_ctle          = array([])
        self.jitter_ctle              = array([])
        self.jitter_spectrum_ctle     = array([])
        self.jitter_ind_spectrum_ctle = array([])
    # - DFE output
    try:
        ignore_until  = (nui - eye_uis) * ui + ui / 2.
        ideal_xings   = array(filter(lambda x: x > ignore_until, list(ideal_xings)))
        min_delay     = ignore_until + conv_dly
        actual_xings  = find_crossings(t, dfe_out, decision_scaler, min_delay = min_delay, mod_type = mod_type, rising_first = False)
        (jitter, t_jitter, isi, dcd, pj, rj, jitter_ext, \
            thresh, jitter_spectrum, jitter_ind_spectrum, spectrum_freqs, \
            hist, hist_synth, bin_centers) = calc_jitter(ui, eye_uis, pattern_len, ideal_xings, actual_xings, rel_thresh)
        self.isi_dfe                 = isi
        self.dcd_dfe                 = dcd
        self.pj_dfe                  = pj
        self.rj_dfe                  = rj
        self.thresh_dfe              = thresh
        self.jitter_dfe              = hist
        self.jitter_ext_dfe          = hist_synth
        self.jitter_spectrum_dfe     = jitter_spectrum
        self.jitter_ind_spectrum_dfe = jitter_ind_spectrum
        self.f_MHz_dfe               = array(spectrum_freqs) * 1.e-6
        skip_factor                  = nbits / eye_bits
        ctle_spec                    = self.jitter_spectrum_ctle
        dfe_spec                     = self.jitter_spectrum_dfe
#        ctle_spec_condensed          = array([ctle_spec.take(range(i, i + skip_factor)).mean() for i in range(0, len(ctle_spec), skip_factor)])
#        window_width                 = len(dfe_spec) / 10
#        self.jitter_rejection_ratio  = moving_average(ctle_spec_condensed, window_width) / moving_average(dfe_spec, window_width) 
        self.jitter_rejection_ratio  = zeros(len(dfe_spec))
    except:
        raise
        self.thresh_dfe              = array([])
        self.jitter_ext_dfe          = array([])
        self.jitter_dfe              = array([])
        self.jitter_spectrum_dfe     = array([])
        self.jitter_ind_spectrum_dfe = array([])
        self.f_MHz_dfe               = array([])
        self.jitter_rejection_ratio  = array([])

    self.jitter_perf = nbits * nspb / (clock() - split_time)
    self.total_perf  = nbits * nspb / (clock() - start_time)
    split_time       = clock()
    self.status      = 'Updating plots...(sweep %d of %d)' % (sweep_num, num_sweeps)

    # Save local variables to class instance for state preservation, performing unit conversion where necessary.
    self.chnl_out    = chnl_out

    self.adaptation = tap_weights
    self.ui_ests    = array(ui_ests) * 1.e12 # (ps)
    self.clocks     = clocks
    self.lockeds    = lockeds
    self.clock_times = clock_times

    # Update plots.
    if(update_plots):
        update_results(self)
        if(not initial_run):
            update_eyes(self)

    self.plotting_perf = nbits * nspb / (clock() - split_time)
    self.status = 'Ready.'

# Plot updating
def update_results(self):
    """Updates all plot data used by GUI."""

    # Copy globals into local namespace.
    ui            = self.ui
    samps_per_ui  = self.nspui
    eye_uis       = self.eye_uis
    num_ui        = self.nui
    clock_times   = self.clock_times
    f             = self.f
    t             = self.t
    t_ns          = self.t_ns
    t_ns_chnl     = self.t_ns_chnl
    conv_dly_ix   = self.conv_dly_ix

    Ts = t[1]
    ignore_until  = (num_ui - eye_uis) * ui

    # Misc.
    f_GHz         = f[:len(f) // 2] / 1.e9
    len_f_GHz     = len(f_GHz)
    self.plotdata.set_data("f_GHz",     f_GHz[1:])
    self.plotdata.set_data("t_ns",      t_ns)
    self.plotdata.set_data("t_ns_chnl", t_ns_chnl)

    # DFE.
    tap_weights = transpose(array(self.adaptation))
    i = 1
    for tap_weight in tap_weights:
        self.plotdata.set_data("tap%d_weights" % i, tap_weight)
        i += 1
    self.plotdata.set_data("tap_weight_index", range(len(tap_weight)))
    self.plotdata.set_data("dfe_out",  self.dfe_out)
    self.plotdata.set_data("ui_ests",  self.ui_ests)
    self.plotdata.set_data("clocks",   self.clocks)
    self.plotdata.set_data("lockeds",  self.lockeds)

    # EQ Tune
    self.plotdata.set_data('ctle_out_h_tune', self.ctle_out_h_tune)
    self.plotdata.set_data('ctle_out_g_tune', self.ctle_out_g_tune)

    # Impulse responses
    self.plotdata.set_data("chnl_h",     self.chnl_h * 1.e-9 / Ts)  # Re-normalize to (V/ns), for plotting.
    self.plotdata.set_data("chnl_g",     self.chnl_g * 1.e-9 / Ts)
    self.plotdata.set_data("tx_h",       self.tx_h * 1.e-9 / Ts)
    self.plotdata.set_data("tx_out_h",   self.tx_out_h * 1.e-9 / Ts)
    self.plotdata.set_data("tx_out_g",   self.tx_out_g * 1.e-9 / Ts)
    self.plotdata.set_data("ctle_h",     self.ctle_h * 1.e-9 / Ts)
    self.plotdata.set_data("ctle_out_h", self.ctle_out_h * 1.e-9 / Ts)
    self.plotdata.set_data("ctle_out_g", self.ctle_out_g * 1.e-9 / Ts)
    self.plotdata.set_data("dfe_h",      self.dfe_h * 1.e-9 / Ts)
    self.plotdata.set_data("dfe_out_h",  self.dfe_out_h * 1.e-9 / Ts)
    self.plotdata.set_data("dfe_out_g",  self.dfe_out_g * 1.e-9 / Ts)

    # Step responses
    self.plotdata.set_data("chnl_s",     self.chnl_s)
    self.plotdata.set_data("tx_s",       self.tx_s)
    self.plotdata.set_data("tx_out_s",   self.tx_out_s)
    self.plotdata.set_data("ctle_s",     self.ctle_s)
    self.plotdata.set_data("ctle_out_s", self.ctle_out_s)
    self.plotdata.set_data("dfe_s",      self.dfe_s)
    self.plotdata.set_data("dfe_out_s",  self.dfe_out_s)

    # Pulse responses
    self.plotdata.set_data("chnl_p",     self.chnl_p)
    self.plotdata.set_data("tx_out_p",   self.tx_out_p)
    self.plotdata.set_data("ctle_out_p", self.ctle_out_p)
    self.plotdata.set_data("dfe_out_p",  self.dfe_out_p)

    # Outputs
    self.plotdata.set_data("ideal_signal",   self.ideal_signal)
    self.plotdata.set_data("chnl_out",   self.chnl_out)
    self.plotdata.set_data("tx_out",     self.rx_in)
    self.plotdata.set_data("ctle_out",   self.ctle_out)
    self.plotdata.set_data("dfe_out",    self.dfe_out)
    self.plotdata.set_data("auto_corr",  self.auto_corr)

    # Frequency responses
    self.plotdata.set_data("chnl_H",     20. * log10(abs(self.chnl_H    [1 : len_f_GHz])))
    self.plotdata.set_data("tx_H",       20. * log10(abs(self.tx_H      [1 : len_f_GHz])))
    self.plotdata.set_data("tx_out_H",   20. * log10(abs(self.tx_out_H  [1 : len_f_GHz])))
    self.plotdata.set_data("ctle_H",     20. * log10(abs(self.ctle_H    [1 : len_f_GHz])))
    self.plotdata.set_data("ctle_out_H", 20. * log10(abs(self.ctle_out_H[1 : len_f_GHz])))
    self.plotdata.set_data("dfe_H",      20. * log10(abs(self.dfe_H     [1 : len_f_GHz])))
    self.plotdata.set_data("dfe_out_H",  20. * log10(abs(self.dfe_out_H [1 : len_f_GHz])))

    # Jitter distributions
    jitter_ext_chnl = self.jitter_ext_chnl # These are used, again, in bathtub curve generation, below.
    jitter_ext_tx   = self.jitter_ext_tx
    jitter_ext_ctle = self.jitter_ext_ctle
    jitter_ext_dfe  = self.jitter_ext_dfe
    self.plotdata.set_data("jitter_bins",     array(self.jitter_bins)     * 1.e12)
    self.plotdata.set_data("jitter_chnl",     self.jitter_chnl)
    self.plotdata.set_data("jitter_ext_chnl", jitter_ext_chnl)
    self.plotdata.set_data("jitter_tx",       self.jitter_tx)
    self.plotdata.set_data("jitter_ext_tx",   jitter_ext_tx)
    self.plotdata.set_data("jitter_ctle",     self.jitter_ctle)
    self.plotdata.set_data("jitter_ext_ctle", jitter_ext_ctle)
    self.plotdata.set_data("jitter_dfe",      self.jitter_dfe)
    self.plotdata.set_data("jitter_ext_dfe",  jitter_ext_dfe)

    # Jitter spectrums
    log10_ui = log10(ui)
    self.plotdata.set_data("f_MHz",     self.f_MHz[1:])
    self.plotdata.set_data("f_MHz_dfe", self.f_MHz_dfe[1:])
    self.plotdata.set_data("jitter_spectrum_chnl",     10. * (log10(self.jitter_spectrum_chnl     [1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_chnl", 10. * (log10(self.jitter_ind_spectrum_chnl [1:]) - log10_ui))
    self.plotdata.set_data("thresh_chnl",              10. * (log10(self.thresh_chnl              [1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_tx",       10. * (log10(self.jitter_spectrum_tx       [1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_tx",   10. * (log10(self.jitter_ind_spectrum_tx   [1:]) - log10_ui))
    self.plotdata.set_data("thresh_tx",                10. * (log10(self.thresh_tx                [1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_ctle",     10. * (log10(self.jitter_spectrum_ctle     [1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_ctle", 10. * (log10(self.jitter_ind_spectrum_ctle [1:]) - log10_ui))
    self.plotdata.set_data("thresh_ctle",              10. * (log10(self.thresh_ctle              [1:]) - log10_ui))
    self.plotdata.set_data("jitter_spectrum_dfe",      10. * (log10(self.jitter_spectrum_dfe      [1:]) - log10_ui))
    self.plotdata.set_data("jitter_ind_spectrum_dfe",  10. * (log10(self.jitter_ind_spectrum_dfe  [1:]) - log10_ui))
    self.plotdata.set_data("thresh_dfe",               10. * (log10(self.thresh_dfe               [1:]) - log10_ui))
    self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])

    # Bathtubs
    half_len = len(jitter_ext_chnl) / 2
    #  - Channel
    bathtub_chnl    = list(cumsum(jitter_ext_chnl[-1 : -(half_len + 1) : -1]))
    bathtub_chnl.reverse()
    bathtub_chnl    = array(bathtub_chnl + list(cumsum(jitter_ext_chnl[:half_len + 1])))
    bathtub_chnl    = where(bathtub_chnl < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_chnl)), bathtub_chnl) # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_chnl", log10(bathtub_chnl))
    #  - Tx
    bathtub_tx    = list(cumsum(jitter_ext_tx[-1 : -(half_len + 1) : -1]))
    bathtub_tx.reverse()
    bathtub_tx    = array(bathtub_tx + list(cumsum(jitter_ext_tx[:half_len + 1])))
    bathtub_tx    = where(bathtub_tx < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_tx)), bathtub_tx) # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_tx", log10(bathtub_tx))
    #  - CTLE
    bathtub_ctle    = list(cumsum(jitter_ext_ctle[-1 : -(half_len + 1) : -1]))
    bathtub_ctle.reverse()
    bathtub_ctle    = array(bathtub_ctle + list(cumsum(jitter_ext_ctle[:half_len + 1])))
    bathtub_ctle    = where(bathtub_ctle < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_ctle)), bathtub_ctle) # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_ctle", log10(bathtub_ctle))
    #  - DFE
    bathtub_dfe    = list(cumsum(jitter_ext_dfe[-1 : -(half_len + 1) : -1]))
    bathtub_dfe.reverse()
    bathtub_dfe    = array(bathtub_dfe + list(cumsum(jitter_ext_dfe[:half_len + 1])))
    bathtub_dfe    = where(bathtub_dfe < MIN_BATHTUB_VAL, 0.1 * MIN_BATHTUB_VAL * ones(len(bathtub_dfe)), bathtub_dfe) # To avoid Chaco log scale plot wierdness.
    self.plotdata.set_data("bathtub_dfe", log10(bathtub_dfe))

    # Eyes
    width    = 2 * samps_per_ui
    xs       = linspace(-ui * 1.e12, ui * 1.e12, width)
    height   = 100
    y_max    = 1.1 * max(abs(array(self.chnl_out)))
    eye_chnl = calc_eye(ui, samps_per_ui, height, self.chnl_out[conv_dly_ix:], y_max)
    y_max    = 1.1 * max(abs(array(self.rx_in)))
    eye_tx   = calc_eye(ui, samps_per_ui, height, self.rx_in[conv_dly_ix:],   y_max)
    y_max    = 1.1 * max(abs(array(self.ctle_out)))
    eye_ctle = calc_eye(ui, samps_per_ui, height, self.ctle_out[conv_dly_ix:], y_max)
    i = 0
    while(clock_times[i] <= ignore_until):
        i += 1
        assert i < len(clock_times), "ERROR: Insufficient coverage in 'clock_times' vector."
    y_max    = 1.1 * max(abs(array(self.dfe_out)))
    eye_dfe  = calc_eye(ui, samps_per_ui, height, self.dfe_out, y_max, clock_times[i:])
    self.plotdata.set_data("eye_index", xs)
    self.plotdata.set_data("eye_chnl",  eye_chnl)
    self.plotdata.set_data("eye_tx",    eye_tx)
    self.plotdata.set_data("eye_ctle",  eye_ctle)
    self.plotdata.set_data("eye_dfe",   eye_dfe)

def update_eyes(self):
    """ Update the heat plots representing the eye diagrams."""

    ui            = self.ui
    samps_per_ui  = self.nspui

    width    = 2 * samps_per_ui
    height   = 100
    xs       = linspace(-ui * 1.e12, ui * 1.e12, width)

    y_max    = 1.1 * max(abs(array(self.chnl_out)))
    ys       = linspace(-y_max, y_max, height)
    self.plots_eye.components[0].components[0].index.set_data(xs, ys)
    self.plots_eye.components[0].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[0].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[0].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[0].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[0].invalidate_draw()

    y_max    = 1.1 * max(abs(array(self.rx_in)))
    ys       = linspace(-y_max, y_max, height)
    self.plots_eye.components[1].components[0].index.set_data(xs, ys)
    self.plots_eye.components[1].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[1].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[1].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[1].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[1].invalidate_draw()

    y_max    = 1.1 * max(abs(array(self.ctle_out)))
    ys       = linspace(-y_max, y_max, height)
    self.plots_eye.components[2].components[0].index.set_data(xs, ys)
    self.plots_eye.components[2].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[2].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[2].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[2].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[2].invalidate_draw()

    y_max    = 1.1 * max(abs(array(self.dfe_out)))
    ys       = linspace(-y_max, y_max, height)
    self.plots_eye.components[3].components[0].index.set_data(xs, ys)
    self.plots_eye.components[3].x_axis.mapper.range.low = xs[0]
    self.plots_eye.components[3].x_axis.mapper.range.high = xs[-1]
    self.plots_eye.components[3].y_axis.mapper.range.low = ys[0]
    self.plots_eye.components[3].y_axis.mapper.range.high = ys[-1]
    self.plots_eye.components[3].invalidate_draw()

    self.plots_eye.request_redraw()

def do_opt_rx(peak_mag, self):
    self.peak_mag_tune = peak_mag
    return self.cost

def do_opt_tx(taps, self):
    taps = list(taps)
    if(self.pretap_tune_enable):
        self.pretap_tune   = taps.pop(0)
    if(self.posttap_tune_enable):
        self.posttap_tune  = taps.pop(0)
    if(self.posttap2_tune_enable):
        self.posttap2_tune = taps.pop(0)
    if(self.posttap3_tune_enable):
        self.posttap3_tune = taps.pop(0)
    return self.cost

