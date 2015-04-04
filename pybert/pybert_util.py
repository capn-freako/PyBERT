"""
General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>
Original date:   September 27, 2014 (Copied from `pybert_cntrl.py'.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from numpy          import sign, sin, pi, array, linspace, float, zeros, ones, repeat, where, diff, append, pad, real, histogram
from numpy          import log10, sqrt, power, exp, cumsum, mean, power, convolve, correlate, reshape, resize, insert
from numpy.random   import normal
from numpy.fft      import fft, ifft
from scipy.signal   import lfilter, iirfilter, invres, freqs, medfilt
from scipy.optimize import minimize
from scipy.stats    import norm
from dfe            import DFE
from cdr            import CDR

debug          = False
gDebugOptimize = False

def moving_average(a, n=3) :
    """Calculates a sliding average over the input vector."""

    ret     = cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return insert(ret[n - 1:], 0, ret[n - 1] * ones(n - 1)) / n

def find_crossing_times(t, x, min_delay=0., rising_first=True, min_init_dev=0.1, thresh = 0.):
    """
    Finds the threshold crossing times of the input signal.

    Inputs:

      - t          Vector of sample times. Intervals do NOT need to be uniform.

      - x          Sampled input vector.

      - min_delay  Minimum delay required, before allowing crossings.
                   (Helps avoid false crossings at beginning of signal.)
                   Optional. Default = 0.

      - rising_first When True, start with the first rising edge found.
                     Optional. Default = True.
                     When this option is True, the first rising edge crossing
                     is the first crossing returned. This is the desired
                     behavior for PyBERT, because we always initialize the
                     bit stream with [0, 1, 1], in order to provide a known
                     synchronization point for jitter analysis.

      - min_init_dev The minimum initial deviation from zero, which must
                     be detected, before searching for crossings.
                     Normalized to maximum input signal magnitude.
                     Optional. Default = 0.1.

      - thresh       Vertical crossing threshold.

    Outputs:

      - xings      The crossing times.

    """

    assert len(t) == len(x), "len(t) (%d) and len(x) (%d) need to be the same." % (len(t), len(x))

    t = array(t)
    x = array(x)

    max_mag_x = max(abs(x))
    min_mag_x = min_init_dev * max_mag_x
    i = 0
    while(abs(x[i]) < min_mag_x):
        i += 1
        assert i < len(x), "ERROR: find_crossing_times(): Input signal minimum deviation not detected!"
    x = x[i:] - thresh
    t = t[i:]

    sign_x      = sign(x)
    sign_x      = where(sign_x, sign_x, ones(len(sign_x))) # "0"s can produce duplicate xings.
    diff_sign_x = diff(sign_x)
    xing_ix     = where(diff_sign_x)[0]
    xings       = [t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) for i in xing_ix]

    if(not xings):
        return(array([]))

    min_time = t[0]
    if(min_delay):
        try:
            assert min_delay < t[-1], "Error: min_delay must be less than final time value."
        except:
            print "min_delay:", min_delay, "t[-3:]:", t[-3:]
            raise
        i = 0
        while(i < len(t) and t[i] < min_delay):
            i += 1
        min_time = t[i]

    i = 0
    try:
        while(xings[i] < min_time):
            i += 1
    except:
        print "i:", i, "min_time:", min_time, "xings[i-3:]:", xings[i-3:]
        print "xing_ix:", xing_ix
        raise

    if(rising_first and diff_sign_x[xing_ix[i]] < 0.):
        i += 1

    if(debug):
        print "find_crossing_times(): min_delay:", min_delay, "; first crossing returned:", xings[i], "rising_first:", rising_first

    return array(xings[i:])

def find_crossings(t, x, amplitude, min_delay = 0., rising_first = True, min_init_dev = 0.1, mod_type = 0):
    """
    Finds the crossing times in a signal, according to the modulation type.

    Inputs:

      Required:

      - t:                   The times associated with each signal sample.

      - x:                   The signal samples.

      - amplitude:           The nominal signal amplitude.
                             (Used for determining thresholds, in the case of some modulation types.)

      Optional:

      - min_delay:           The earliest possible sample time we want returned.
                             Default = 0.

      - rising_first         When True, start with the first rising edge found.
                             When this option is True, the first rising edge crossing
                             is the first crossing returned. This is the desired
                             behavior for PyBERT, because we always initialize the
                             bit stream with [0, 1, 1], in order to provide a known
                             synchronization point for jitter analysis.
                             Default = True.

      - min_init_dev         The minimum initial deviation from zero, which must
                             be detected, before searching for crossings.
                             Normalized to maximum input signal magnitude.
                             Default = 0.1.

      - mod_type:            The modulation type. Allowed values are: (Default = 0.)
                               - 0: NRZ
                               - 1: Duo-binary
                               - 2: PAM-4

    Outputs:

      - xings:               The crossing times.

    """

    xings = find_crossing_times(t, x, min_delay = min_delay, rising_first = rising_first, min_init_dev = min_init_dev)
    return array(xings)

def calc_jitter(ui, nui, pattern_len, ideal_xings, actual_xings, rel_thresh=6, num_bins=99, zero_mean=True):
    """
    Calculate the jitter in a set of actual zero crossings, given the ideal crossings and unit interval.

    Inputs:

      - ui               : The nominal unit interval.

      - nui              : The number of unit intervals spanned by the input signal.

      - pattern_len      : The number of unit intervals, before input symbol stream repeats.

      - ideal_xings      : The ideal zero crossing locations of the edges.

      - actual_xings     : The actual zero crossing locations of the edges.

      - rel_thresh       : (optional) The threshold for determining periodic jitter spectral components (sigma).

      - num_bins         : (optional) The number of bins to use, when forming histograms.

      - zero_mean        : (optional) Force the mean jitter to zero, when True.

    Outputs:

      - jitter   : The total jitter.

      - t_jitter : The times (taken from 'ideal_xings') corresponding to the returned jitter values.

      - isi      : The peak to peak jitter due to intersymbol interference.

      - dcd      : The peak to peak jitter due to duty cycle distortion.

      - pj       : The peak to peak jitter due to uncorrelated periodic sources.

      - rj       : The standard deviation of the jitter due to uncorrelated unbounded random sources.

      - tie_ind  : The data independent jitter.

      - thresh   : Threshold for determining periodic components.

      - jitter_spectrum  : The spectral magnitude of the total jitter.

      - tie_ind_spectrum : The spectral magnitude of the data independent jitter.

      - spectrum_freqs   : The frequencies corresponding to the spectrum components.

      - hist        : The histogram of the actual jitter.

      - hist_synth  : The histogram of the extrapolated jitter.

      - bin_centers : The bin center values for both histograms.

    """

    def my_hist(x):
        """
        Calculates the probability mass function (PMF) of the input vector,
        enforcing an output range of [-UI/2, +UI/2], sweeping everything in [-UI, -UI/2] into the first bin,
        and everything in [UI/2, UI] into the last bin.
        """
        hist, bin_edges      = histogram(x, [-ui] + [-ui / 2. + i * ui / (num_bins - 2) for i in range(num_bins - 1)] + [ui])
        bin_centers          = [-ui / 2.] + [mean([bin_edges[i + 1], bin_edges[i + 2]]) for i in range(len(bin_edges) - 3)] + [ui / 2.]
        return (array(map(float, hist)) / sum(hist), bin_centers)

    # Assemble the TIE track.
    jitter   = []
    t_jitter = []
    i        = 0
    ideal_xings  = array(ideal_xings)  - (ideal_xings[0] - ui / 2.)
    if(len(actual_xings)):
        actual_xings = array(actual_xings) - (actual_xings[0] - ideal_xings[0])

    skip_next_ideal_xing = False
    pad_ixs = []
    for ideal_xing in ideal_xings:
        if(skip_next_ideal_xing):
            t_jitter.append(ideal_xing)
            skip_next_ideal_xing = False
            continue
        # Find the closest actual crossing, occuring within [-ui, ui],
        # to the ideal crossing, checking for missing crossings.
        min_t = ideal_xing - ui
        max_t = ideal_xing + ui
        while(i < len(actual_xings) and actual_xings[i] < min_t):
            i += 1
        if(i == len(actual_xings)):              # We've exhausted the list of actual crossings; we're done.
            break
        if(actual_xings[i] > max_t):             # Means the xing we're looking for didn't occur, in the actual signal.
            pad_ixs.append(len(jitter) + 2 * len(pad_ixs))
            skip_next_ideal_xing = True          # If we missed one, we missed two.
        else:
            candidates = []
            j = i
            while(j < len(actual_xings) and actual_xings[j] <= max_t):
                candidates.append(actual_xings[j])
                j += 1
            ties     = array(candidates) - ideal_xing
            tie_mags = abs(ties)
            best_ix  = where(tie_mags == min(tie_mags))[0][0]
            tie      = ties[best_ix]
            jitter.append(tie)
            i += best_ix + 1
        t_jitter.append(ideal_xing)
    jitter  = array(jitter)

    if(debug):
        print "mean(jitter):", mean(jitter)
        print "len(jitter):", len(jitter)

    if(zero_mean):
        jitter -= mean(jitter)

    jitter = list(jitter)
    for pad_ix in pad_ixs:
        jitter.insert(pad_ix, -3. * ui / 4.)         # Pad the jitter w/ alternating +/- 3UI/4. (Will get pulled into [-UI/2, UI/2], later.
        jitter.insert(pad_ix,  3. * ui / 4.)
    jitter = array(jitter)

    # Do the jitter decomposition.
    # - Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    xings_per_pattern    = where(ideal_xings >= pattern_len * ui)[0][0]
    fallings_per_pattern = xings_per_pattern // 2
    risings_per_pattern  = xings_per_pattern - fallings_per_pattern
    num_patterns         = nui // pattern_len

    # -- Check and adjust vector lengths, reporting out if any modifications were necessary.
    if(False):
        if(len(jitter) < xings_per_pattern * num_patterns):
            print "Added %d zeros to 'jitter'." % (xings_per_pattern * num_patterns - len(jitter))
            jitter = append(jitter, zeros(xings_per_pattern * num_patterns - len(jitter)))
        try:
            t_jitter = t_jitter[:len(jitter)]
            if(len(jitter) > len(t_jitter)):
                jitter = jitter[:len(t_jitter)]
                print "Had to shorten 'jitter', due to 't_jitter'."
        except:
            print "jitter:", jitter
            raise

    # -- Do the reshaping and check results thoroughly.
    try:
        tie_risings  = jitter.take(range(0, len(jitter), 2))
        tie_fallings = jitter.take(range(1, len(jitter), 2))
        tie_risings.resize (num_patterns * risings_per_pattern)
        tie_fallings.resize(num_patterns * fallings_per_pattern)
        tie_risings  = reshape(tie_risings, (num_patterns, risings_per_pattern))
        tie_fallings = reshape(tie_fallings,(num_patterns, fallings_per_pattern))
    except:
        print "ideal_xings[xings_per_pattern - 1]:", ideal_xings[xings_per_pattern - 1], "ideal_xings[-1]:", ideal_xings[-1]
        print "num_patterns:", num_patterns, "risings_per_pattern:", risings_per_pattern, "fallings_per_pattern:", fallings_per_pattern, "len(jitter):", len(jitter)
        print "nui:", nui, "pattern_len:", pattern_len
        raise

    # - Use averaging to remove the uncorrelated components, before calculating data dependent components.
    tie_risings_ave  = tie_risings.mean(axis=0)
    tie_fallings_ave = tie_fallings.mean(axis=0)
    isi = max(tie_risings_ave.ptp(), tie_fallings_ave.ptp())
    isi = min(isi, ui) # Cap the ISI at the unit interval.
    dcd = abs(mean(tie_risings_ave) - mean(tie_fallings_ave))

    # - Subtract the data dependent jitter from the original TIE track, in order to yield the data independent jitter.
    tie_ave  = sum(zip(tie_risings_ave, tie_fallings_ave), ())
    tie_ave  = resize(tie_ave, len(jitter))
    tie_ind  = jitter - tie_ave

    # - Use spectral analysis to help isolate the periodic components of the data independent jitter.
    # -- Calculate the total jitter spectrum, for display purposes only.
    # --- Make vector uniformly sampled in time, via zero padding where necessary.
    # --- (It's necessary to keep track of those elements in the resultant vector, which aren't paddings; hence, 'valid_ix'.)
    x, valid_ix     = make_uniform(t_jitter, jitter, ui, nui)
    y               = fft(x)
    jitter_spectrum = abs(y[:len(y) / 2]) / sqrt(len(jitter)) # Normalized, in order to make power correct.
    f0              = 1. / (ui * nui)
    spectrum_freqs  = [i * f0 for i in range(len(y) / 2)]

    # -- Use the data independent jitter spectrum for our calculations.
    tie_ind_uniform, valid_ix = make_uniform(t_jitter, tie_ind, ui, nui)

    # --- Normalized, in order to make power correct, since we grab Rj from the freq. domain.
    # --- (I'm using the length of the vector before zero padding, because zero padding doesn't add energy.)
    # --- (This has the effect of making our final Rj estimate more conservative.)
    y        = fft(tie_ind_uniform) / sqrt(len(tie_ind))
    y_mag    = abs(y)
    y_mean   = moving_average(y_mag, n = len(y_mag) / 10)
    y_var    = moving_average((y_mag - y_mean) ** 2, n = len(y_mag) / 10)
    y_sigma  = sqrt(y_var)
    thresh   = y_mean + rel_thresh * y_sigma
    y_per    = where(y_mag > thresh, y,             zeros(len(y)))   # Periodic components are those lying above the threshold.
    y_rnd    = where(y_mag > thresh, zeros(len(y)), y)               # Random components are those lying below.
    y_rnd    = abs(y_rnd)
    rj       = sqrt(mean((y_rnd - mean(y_rnd)) ** 2))
    tie_per  = real(ifft(y_per)).take(valid_ix) * sqrt(len(tie_ind)) # Restoring shape of vector to its original, non-uniformly sampled state.
    pj       = tie_per.ptp()

    # --- Save the spectrum, for display purposes.
    tie_ind_spectrum = y_mag[:len(y_mag) / 2]

    # - Reassemble the jitter, excluding the Rj.
    # -- Here, we see why it was necessary to keep track of the non-padded elements with 'valid_ix':
    # -- It was so that we could add the average and periodic components back together, maintaining correct alignment between them.
    if(len(tie_per) > len(tie_ave)):
        tie_per = tie_per[:len(tie_ave)]
    if(len(tie_per) < len(tie_ave)):
        tie_ave = tie_ave[:len(tie_per)]
    jitter_synth = tie_ave + tie_per

    # - Calculate the histogram of original, for comparison.
    hist,       bin_centers = my_hist(jitter)

    # - Calculate the histogram of everything, except Rj.
    hist_synth, bin_centers = my_hist(jitter_synth)

    # - Extrapolate the tails by convolving w/ complete Gaussian.
    rv         = norm(loc = 0., scale = rj)
    rj_pdf     = rv.pdf(bin_centers)
    rj_pmf     = (rj_pdf / sum(rj_pdf))
    hist_synth = convolve(hist_synth, rj_pmf)
    tail_len   = (len(bin_centers) - 1) / 2
    hist_synth = [sum(hist_synth[: tail_len + 1])] + list(hist_synth[tail_len + 1 : len(hist_synth) - tail_len - 1]) + [sum(hist_synth[len(hist_synth) - tail_len - 1 :])]

    return (jitter, t_jitter, isi, dcd, pj, rj, tie_ind,
            thresh[:len(thresh) / 2], jitter_spectrum, tie_ind_spectrum, spectrum_freqs,
            hist, hist_synth, bin_centers)

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

    - y      : The uniformly sampled, zero padded jitter vector.

    - y_ix   : The indices where y is valid (i.e. - not zero padded).

    """

    if(len(t) < len(jitter)):
        jitter = jitter[:len(t)]

    run_lengths    = map(int, diff(t) / ui + 0.5)
    valid_ix       = [0] + list(cumsum(run_lengths))
    valid_ix       = filter(lambda x: x < nbits, valid_ix)
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

    return jitter, valid_ix

def calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, ws):
    """
    Calculates propagation constant from cross-sectional parameters.

    The formula's applied are taken from Howard Johnson's "Metallic Transmission Model"
    (See "High Speed Signal Propagation", Sec. 3.1.)

    Inputs:
      - R0          skin effect resistance (Ohms/m)
      - w0          cross-over freq.
      - Rdc         d.c. resistance (Ohms/m)
      - Z0          characteristic impedance in LC region (Ohms)
      - v0          propagation velocity (m/s)
      - Theta0      loss tangent
      - ws          frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    w = array(ws).copy()

    # Guard against /0.
    if(w[0] == 0):
        w[0] = 1.e-12

    Rac   = R0 * sqrt(2 * 1j * w / w0)                        # AC resistance vector
    R     = sqrt(power(Rdc, 2) + power(Rac, 2))               # total resistance vector
    L0    = Z0 / v0                                           # "external" inductance per unit length (H/m)
    C0    = 1. / (Z0 * v0)                                    # nominal capacitance per unit length (F/m)
    C     = C0 * power((1j * w / w0), (-2. * Theta0 / pi))    # complex capacitance per unit length (F/m)
    gamma = sqrt((1j * w * L0 + R) * (1j * w * C))            # propagation constant (nepers/m)
    Zc    = sqrt((1j * w * L0 + R) / (1j * w * C))            # characteristic impedance (Ohms)

    return (gamma, Zc)

def calc_G(H, Rs, Cs, Zc, RL, Cp, CL, ws):
    """
    Calculates fully loaded transfer function of complete channel.

    Inputs:
      - H     unloaded transfer function of interconnect
      - Rs    source series resistance
      - Cs    source parallel (parasitic) capacitance
      - Zc    frequency dependent characteristic impedance of the interconnect
      - RL    load resistance (differential)
      - Cp    load parallel (parasitic) capacitance (single ended)
      - CL    load series (d.c. blocking) capacitance (single ended)
      - ws    frequency sample points vector

    Outputs:
      - G     frequency dependent transfer function of channel
    """

    w = array(ws).copy()

    # Guard against /0.
    if(w[0] == 0):
        w[0] = 1.e-12

    # Impedance looking back into the Tx output is a simple parallel RC network.
    Zs = Rs / (1. + 1j * w * Rs * Cs)
    # Rx load impedance is 2 series, a.c.-coupling capacitors, in series w/ parallel comb. of Rterm & parasitic cap.
    # (The two parasitic capacitances are in series.)
    ZL = 2. * 1. / (1j * w * CL) + RL / (1. + 1j * w * RL * Cp / 2)
    # Admittance into the interconnect is (Cs || Zc) / (Rs + (Cs || Zc)).
    Cs_par_Zc = Zc / (1. + 1j * w * Zc * Cs)
    A         = Cs_par_Zc / (Rs + Cs_par_Zc)
    # Reflection coefficient at Rx:
    R1        = (ZL - Zc) / (ZL + Zc)
    # Reflection coefficient at Tx:
    R2        = (Zs - Zc) / (Zs + Zc)
    # Fully loaded channel transfer function:
    G = A * H * (1 + R1) / (1 - R1 * R2 * H**2)
    G = G * (((RL/(1j*w*Cp/2))/(RL + 1/(1j*w*Cp/2))) / ZL) # Corrected for divider action.
                                                           # (i.e. - We're interested in what appears across RL.)
    return G

def calc_eye(ui, samps_per_ui, height, ys, y_max, clock_times=None):
    """
    Calculates the "eye" diagram of the input signal vector.

    Inputs:
      - ui             unit interval (s)
      - samps_per_ui   # of samples per unit interval
      - height         height of output image data array
      - ys             signal vector of interest
      - y_max          max. +/- vertical extremity of plot
      - clock_times    (optional)
                       vector of clock times to use for eye centers.
                       If not provided, just use mean zero-crossing and
                       assume constant UI and no phase jumps.
                       (This allows the same function to be used for
                       eye diagram creation,
                       for both pre and post-CDR signals.)

    Outputs:
      - img_array      The "heat map" representing the eye diagram.
                       Each grid location contains a value indicating
                       the number of times the signal passed through
                       that location.

    """

    # List/array necessities.
    ys = array(ys)

    # Intermediate variable calculation.
    tsamp = ui / samps_per_ui

    # Adjust the scaling.
    width    = 2 * samps_per_ui
    y_scale  = height / (2 * y_max)          # (pixels/V)
    y_offset = height / 2                    # (pixels)

    # Generate the "heat" picture array.
    img_array = zeros([height, width])
    if(clock_times):
        for clock_time in clock_times:
            start_time = clock_time - ui
            stop_time  = clock_time + ui
            start_ix   = int(start_time / tsamp)
            if(start_ix + 2 * samps_per_ui > len(ys)):
                break
            interp_fac = (start_time - start_ix * tsamp) / tsamp
            last_y     = ys[start_ix]
            i = 0
            for (samp1, samp2) in zip(ys[start_ix : start_ix + 2 * samps_per_ui],
                                      ys[start_ix + 1 : start_ix + 1 + 2 * samps_per_ui]):
                y = samp1 + (samp2 - samp1) * interp_fac
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                last_y = y
                i += 1
    else:
        start_ix      = (where(diff(sign(ys)))[0] % samps_per_ui).mean() + samps_per_ui // 2 
        last_start_ix = len(ys) - 2 * samps_per_ui
        while(start_ix < last_start_ix):
            i = 0
            for y in ys[start_ix : start_ix + 2 * samps_per_ui]:
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                i += 1
            start_ix += samps_per_ui

    return img_array

def make_ctle(rx_bw, peak_freq, peak_mag, w):
    """
    Generate the frequency response of a continuous time linear
    equalizer (CTLE), given the:

    - signal path bandwidth,
    - peaking specification, and
    - list of frequencies of interest.

    We use the 'invres()' function from scipy.signal, as it suggests
    itself as a natural approach, given our chosen use model of having
    the user provide the peaking frequency and degree of peaking.

    That is, we define our desired frequency response using one zero
    and two poles, where:

    - The pole locations are equal to:
       - the signal path natural bandwidth, and
       - the user specified peaking frequency.

    - The zero location is chosen, so as to provide the desired degree
      of peaking.

    Inputs:

      - rx_bw        The natural (or, unequalized) signal path bandwidth (Hz).

      - peak_freq    The location of the desired peak in the frequency
                     response (Hz).

      - peak_mag     The desired relative magnitude of the peak (dB). (mag(H(0)) = 1)

      - w            The list of frequencies of interest (rads./s).

    Outputs:

      - w, H         The resultant complex frequency response, at the
                     given frequencies.

    """

    p2   = -2. * pi * rx_bw
    p1   = -2. * pi * peak_freq
    z    = p1 / pow(10., peak_mag / 20.)
    if(p2 != p1):
        r1   = (z - p1) / (p2 - p1)
        r2   = 1 - r1
    else:
        r1   = -1.
        r2   = z - p1
    b, a = invres([r1, r2], [p1, p2], [])

    return freqs(b, a, w)

def trim_impulse(g, Ts=0, chnl_dly=0, min_len=0, max_len=1000000):
    """
    Trim impulse response, for more useful display, by:
      - eliminating 90% of the overall delay from the beginning, and
      - clipping off the tail, after 99.8% of the total power has been captured.
        (Using 99.9% was causing problems; I don't know why.)

    Inputs:
    
      - g         impulse response

      - Ts        (optional) sample interval (same units as 'chnl_dly')

      - chnl_dly  (optional) channel delay

      - min_len   (optional) minimum length of returned vector

      - max_len   (optional) maximum length of returned vector

    Outputs:
    
      - g_trim    trimmed impulse response

      - start_ix  index of first returned sample

    """

    g         = array(g)

    if(Ts and chnl_dly):
        start_ix  = int(0.9 * chnl_dly / Ts)
    else:
        min_mag = 0.01 * max(abs(g))
        i = 0
        while(abs(g[i]) < min_mag):
            i += 1
        start_ix = int(0.9 * i)

    Pt        = 0.998 * sum(g[start_ix:] ** 2)
    i         = start_ix
    P         = 0
    while(P < Pt):
        P += g[i] ** 2
        i += 1

    vec_len = i - start_ix
    if(vec_len < min_len):
        i = min(len(g), start_ix + min_len)
    if(vec_len > max_len):
        i = start_ix + max_len

    return (g[start_ix : i], start_ix)

def import_qucs_csv(filename, sample_per):
    """ Read in a CSV waveform file exported by QUCS,
        resampling as appropriate, via linear interpolation.

        Inputs:
        - filename:   Name of waveform file to read in.
        - sample_per: New sample interval

        Outputs:
        - res:        Resampled waveform.
    """

    # Read in original data from file.
    ts = []
    xs = []
    with open(filename, mode='rU') as csv_file:
        line = csv_file.readline()               # We don't use the header.
        for line in csv_file:
            tmp = map (float, line.split(';'))   # QUCS uses the semicolon as the field separator.
            ts.append(tmp[0])
            xs.append(tmp[1])

    # Resample data, using linear interpolation.
    tmax = ts[-1]
    res  = []
    t    = 0.0
    i    = 0
    while(t < tmax):
        while(ts[i] <= t):
            i = i + 1
        res.append(xs[i - 1] + (xs[i] - xs[i - 1]) * (t - ts[i - 1]) / (ts[i] - ts[i - 1]))
        t += sample_per

    return array(res)

def trim_shift_scale(ideal_h, actual_h, use_corr = False):
    if(use_corr):
        corr_res = correlate(ideal_h, actual_h, mode='valid')
        offset   = int(sum([i * corr_res[i] ** 2 for i in range(len(corr_res))]) / sum(corr_res ** 2))
    else:
        offset   = mean(where(ideal_h == max(ideal_h))[0]) - mean(where(actual_h == max(actual_h))[0])

    res = ideal_h[offset:].copy()
    res.resize(len(actual_h))                    # Clips, if too long; pads w/ zeros, if too short.
    return res * max(actual_h) / max(res)

def opt_tx(old_taps, cursor_pos, nspui, chnl_h, ideal_h, use_pulse, ctle_h = None):
    """
    """

    cons     = ({'type': 'ineq',
                    'fun' : lambda x: array([1 - sum(abs(x))])})
    if(gDebugOptimize):
        res  = minimize(ffe_cost, old_taps, args=(cursor_pos, nspui, chnl_h, ideal_h, use_pulse, ctle_h), constraints=cons, options={'disp' : True})
    else:
        res  = minimize(ffe_cost, old_taps, args=(cursor_pos, nspui, chnl_h, ideal_h, use_pulse, ctle_h), constraints=cons, options={'disp' : False})

    return res.x

def ffe_cost(taps, cursor_pos, nspui, chnl_h, ideal_h, use_pulse, ctle_h = None):
    """
    """

    # Assemble the link impulse response.
    main_tap = 1. - sum(abs(taps))
    taps     = list(taps)
    taps.insert(cursor_pos, main_tap)
    tx_h     = sum([[x] + list(zeros(nspui - 1)) for x in taps], [])
    out_h    = convolve(tx_h, chnl_h)
    if ctle_h is not None:
        out_h = convolve(ctle_h, out_h)
    
    cost, ideal_resp, actual_resp = calc_cost(out_h, ideal_h, nspui, use_pulse)

    return cost

def calc_cost(actual_h, ideal_h, nspui, use_pulse):
    """
    Calculates the cost function for link equalization optimization.

    Inputs:

      - actual_h   actual impulse response

      - ideal_h    ideal impulse response

      - nspui      number of samples per unit interval

      - use_pulse  Boolean flag; use pulse response when true.

    Outputs:

      - cost        cost function value

      - ideal_resp  ideal response function

      - actual_resp actual response function

    """

    # Calculate the cost function.
    if(use_pulse):
        ideal_h  = trim_shift_scale(ideal_h, actual_h, use_corr = True)
        actual_s = actual_h.cumsum()
        actual   = actual_s - pad(actual_s[:-nspui], (nspui,0), 'constant', constant_values=(0,0))
        ideal_s  = ideal_h.cumsum()
        ideal    = ideal_s - pad(ideal_s[:-nspui], (nspui,0), 'constant', constant_values=(0,0))
        assert max(ideal) == max(abs(ideal)), "pybert_util.calc_cost(): ERROR: Ideal response peak is negative!"
        min_thresh   = 0.01 * max(ideal)
        pulse_center = int(sum([i * ideal[i] ** 2 for i in range(len(ideal))]) / sum(ideal ** 2))
        i = pulse_center
        while(i > 0 and ideal[i] > min_thresh):
            i -= 1
        pulse_left = i
        i = pulse_center
        while(i < len(ideal) and ideal[i] > min_thresh):
            i += 1
        pulse_right = i
        pwr_outside = sum(actual[:pulse_left + 1] ** 2) + sum(actual[pulse_right:] ** 2) 
        pwr_inside  = sum(actual[pulse_left + 1 : pulse_right] ** 2)
        cost        = pwr_outside / pwr_inside
    else:
        ideal_h  = trim_shift_scale(ideal_h, actual_h, use_corr = True)
        actual   = actual_h
        ideal    = ideal_h
        cost     = sum((actual - ideal) ** 2)

    return cost, ideal, actual

def lfsr_bits(taps, seed):
    """
    """

    val      = int(seed)
    num_taps = max(taps)
    mask     = (1 << num_taps) - 1

    while(True):
        xor_res = reduce(lambda x, b: x ^ b, [bool(val & (1 << (tap - 1))) for tap in taps])
        val     = (val << 1) & mask  # Just to keep 'val' from growing without bound.
        if(xor_res):
            val += 1
        yield(val & 1)

def safe_log10(x):
    "Guards against pesky 'Divide by 0' error messages."

    if hasattr(x, "__len__"):
        x = where(x == 0, 1.e-20 * ones(len(x)), x)
    else:
        if(x == 0):
            x = 1.e-20

    return log10(x)

