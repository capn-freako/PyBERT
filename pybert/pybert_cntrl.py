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

debug = False

def get_chnl_in(self):
    """Generates the channel input."""

    bits    = self.bits
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
    res         = repeat(2 * bits - 1, nspb)
    ideal_xings = find_crossing_times(t, res, anlg=False)
    # Filter it.
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
    tx_xings                             = find_crossing_times(t, res)
    if(tx_xings[0] < ui):
        tx_xings = tx_xings[1:]
    (jitter, t_jitter, isi, dcd, pj, rj) = calc_jitter(ui, nbits, pattern_len, ideal_xings, tx_xings)

    if(debug):
        print "Tx output jitter:"
        print "\tISI:", isi * 1.e12, "ps"
        print "\tDCD:", dcd * 1.e12, "ps"
        print "\tPj:", pj * 1.e12, "ps"
        print "\tRj:", rj * 1.e12, "ps"

    self.ideal_xings = ideal_xings
    self.tx_xings    = tx_xings
    self.isi_tx      = isi
    self.dcd_tx      = dcd
    self.pj_tx       = pj
    self.rj_tx       = rj

    return res

def find_crossing_times(t, x, anlg=True):
    """
    Finds the zero crossing times of the input signal.

    Inputs:

      - t     Vector of sample times. Intervals do NOT need to be uniform.

      - x     Sampled input vector.

      - anlg  Interpolation flag. When TRUE, use linear interpolation,
              in order to determine zero crossing times more precisely.

    Outputs:

      - xings     The crossing times.

    """

    assert len(t) == len(x), "len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x))

    sign_x      = sign(x)
    sign_x      = where(sign_x, sign_x, ones(len(sign_x))) # "0"s can produce duplicate xings.
    diff_sign_x = diff(sign_x)
    xing_ix     = where(diff_sign_x)[0]
    if(anlg):
        try:
            xings    = [t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) for i in xing_ix]
        except:
            print "len(t):", len(t), "len(x):", len(x), "i:", i
            raise
    else:
        xings    = [t[i + 1] for i in xing_ix]
    return array(xings)

def calc_jitter(ui, nbits, pattern_len, ideal_xings, actual_xings):
    """
    Calculate the jitter in a set of actual zero crossings, given the ideal crossings and unit interval.

    Inputs:

      - ui               : The nominal unit interval.

      - nbits            : The number of unit intervals spanned by the input signal.

      - pattern_len      : The number of unit intervals, before input bit stream repeats.

      - ideal_xings      : The ideal zero crossing locations of the edges.

      - actual_xings     : The actual zero crossing locations of the edges.

    Outputs:

      - jitter   : The jitter values.

      - t_jitter : The times (taken from 'ideal_xings') corresponding to the returned jitter values.

      - isi      : The jitter due to intersymbol interference.

      - dcd      : The jitter due to duty cycle distortion.

      - pj       : The jitter due to uncorrelated periodic sources.

      - rj       : The jitter due to uncorrelated unbounded random sources.

    Notes:

      - It is assumed that the first crossing is a rising edge.

      - Any delay present in the actual crossings, relative to the ideal crossings,
        should be removed, before calling this function.

    """

    jitter   = []
    t_jitter = []
    i        = 0
    # Assemble the TIE track.
    for actual_xing in actual_xings:
        # Check for missed crossings and zero fill, as necessary.
        while(i < len(ideal_xings) and (actual_xing - ideal_xings[i]) > ui):
            for j in range(2): # If we missed one crossing, then we missed two.
                if(i >= len(ideal_xings)):
                    break
                jitter.append(0.)
                t_jitter.append(ideal_xings[i])
                i += 1
        if(i >= len(ideal_xings)):
            break
        jitter.append(actual_xing - ideal_xings[i])
        t_jitter.append(ideal_xings[i])
        i += 1
    jitter  = array(jitter)
    jitter -= mean(jitter)
    # Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    # - We have to be careful to keep the last crossing, in the case where there are an odd number of them,
    #   because we'll be assembling a "repeated average" vector, later, and subtracting it from the original
    #   jitter vector. So, we can't get sloppy, or we'll end up with misalignment between the two.
    xings_per_pattern    = where(ideal_xings > pattern_len * ui)[0][0]
    risings_per_pattern  = int(xings_per_pattern / 2. + 0.5)
    fallings_per_pattern = xings_per_pattern // 2
    num_patterns         = nbits // pattern_len - 1
    jitter = jitter[xings_per_pattern:] # The first pattern period is problematic.
    if(len(jitter) < xings_per_pattern * num_patterns):
        jitter = array(list(jitter).append(0.))
    try:
        t_jitter = t_jitter[:len(jitter)]
    except:
        print "jitter:", jitter
    try:
        tie_risings          = reshape(jitter.take(range(0, num_patterns * risings_per_pattern * 2, 2)),  (num_patterns, risings_per_pattern))
        tie_fallings         = reshape(jitter.take(range(1, num_patterns * fallings_per_pattern * 2, 2)), (num_patterns, fallings_per_pattern))
    except:
        print "ideal_xings[xings_per_pattern - 1]:", ideal_xings[xings_per_pattern - 1], "ideal_xings[-1]:", ideal_xings[-1]
        print "num_patterns:", num_patterns, "risings_per_pattern:", risings_per_pattern, "fallings_per_pattern:", fallings_per_pattern, "len(jitter):", len(jitter)
        print "nbits:", nbits, "pattern_len:", pattern_len
        raise
    assert len(filter(lambda x: x == None, tie_risings)) == 0, "num_patterns: %d, risings_per_pattern: %d, len(jitter): %d" % \
                                           (num_patterns, risings_per_pattern, len(jitter))
    assert len(filter(lambda x: x == None, tie_fallings)) == 0, "num_patterns: %d, fallings_per_pattern: %d, len(jitter): %d" % \
                                           (num_patterns, fallings_per_pattern, len(jitter))
    # Do the jitter decomposition.
    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    tie_risings_ave  = tie_risings.mean(axis=0)
    tie_fallings_ave = tie_fallings.mean(axis=0)
    try:
        isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    except:
        print "tie_risings_ave:", tie_risings_ave, "\ntie_fallings_ave:", tie_fallings_ave
        raise
    dcd = abs(mean(tie_risings_ave) - mean(tie_fallings_ave))
    # - Subtract the data dependent jitter from the original TIE track.
    tie_ave  = concatenate(zip(tie_risings_ave, tie_fallings_ave))
    if(xings_per_pattern % 2): # Correct for odd number of crossings per pattern.
        #print "Odd # of xings per pattern detected."
        tmp = list(tie_ave)
        tmp.append(tie_risings_ave[-1])
        tie_ave = array(tmp)
    tie_ave  = resize(tie_ave, len(jitter))
    try:
        tie_ind  = jitter - tie_ave
    except:
        print "tie_ave:", tie_ave
        raise

    if(debug):
        #print "jitter:", jitter, "tie_ave:", tie_ave
        plot(jitter,                                               label="jitter",          color="b")
        plot(concatenate(zip(tie_risings_ave, tie_fallings_ave)),  label="rise/fall aves.", color="r")
        title("Original TIE track & Average over one pattern period")
        legend()
        show()
        plot(t_jitter, tie_ind)
        title("Data Independent Jitter")
        show()

    # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
    y        = fft(make_uniform(t_jitter, tie_ind, ui, nbits))
    y_mag    = abs(y)
    y_sigma  = sqrt(mean((y_mag - mean(y_mag)) ** 2))
    # - We'll call any spectral component with a magnitude > 3-sigma a "peak".
    thresh   = 6 * y_sigma
    y_per    = where(y_mag > thresh, y, zeros(len(y)))

    #if(debug):
    if(True):
        print "# of spectral peaks detected:", len(where(y_per)[0])
        print "thresh:", thresh, "max(y_mag):", max(y_mag)

    tie_per  = real(ifft(y_per))
    pj       = tie_per.ptp()

    if(debug):
        plot(tie_per)
        title("Periodic Jitter")
        show()

    # - Subtract the periodic jitter and calculate the standard deviation of what's left.
    tie_rnd  = make_uniform(ideal_xings[:len(tie_ind)], tie_ind, ui, nbits) - tie_per
    rj       = sqrt(mean((tie_rnd - mean(tie_rnd)) ** 2))

    return (jitter, t_jitter, isi, dcd, pj, rj)

def make_uniform(t, jitter, ui, nbits):
    """
    Make the jitter vector uniformly sampled in time, by zero-filling where necessary.

    The trick, here, is creating a uniformly sampled input vector for the FFT operation,
    since the jitter samples are almost certainly not uniformly sampled.
    We do this by simply zero padding the missing samples.

    Inputs:

    - t      : The sample times for the 'jitter' vector.

    - jitter : The input jitter samples.

    - ui     : The nominal unit interval.

    - nbits  : The desired number of unit intervals, in the time domain.

    Output:

    - y      : The FFT of the input jitter.

    """

    assert len(t) == len(jitter)

    half_n         = nbits / 2
    run_lengths    = map(int, diff(t) / ui + 0.5)
    missing        = where(array(run_lengths) > 1)[0]
    num_insertions = 0
    jitter         = list(jitter) # Because we use 'insert'.
    for i in missing:
        for j in range(run_lengths[i] - 1):
            jitter.insert(i + 1 + num_insertions, 0.)
            num_insertions += 1
    if(len(jitter) < nbits):
        jitter.extend([0.] * (nbits - len(jitter)))
    if(len(jitter) > nbits):
        jitter = jitter[:nbits]
    return jitter

def calc_jitter_spectrum(t, jitter, ui, nbits):
    """
    Calculate the spectral magnitude estimate of the input jitter samples.

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

    half_n = nbits / 2
    f0     = 1. / (ui * nbits)
    f      = [i * f0 for i in range(half_n)]
    y      = fft(make_uniform(t, jitter, ui, nbits))
    y      = array(abs(y[:half_n])) / half_n / ui
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
        
