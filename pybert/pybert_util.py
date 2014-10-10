# General purpose utilities for PyBERT.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   September 27, 2014 (Copied from `pybert_cntrl.py'.)
#
# Copyright (c) 2014 David Banas; all rights reserved World wide.

from numpy        import sign, sin, pi, array, linspace, float, zeros, ones, repeat, where, diff, log10, sqrt, power, exp
from numpy.random import normal
from numpy.fft    import fft
from scipy.signal import lfilter, iirfilter
from dfe          import DFE
from cdr          import CDR
import time
from pylab import *
import numpy as np

debug = False

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

      - jitter   : The total jitter.

      - t_jitter : The times (taken from 'ideal_xings') corresponding to the returned jitter values.

      - isi      : The peak to peak jitter due to intersymbol interference.

      - dcd      : The peak to peak jitter due to duty cycle distortion.

      - pj       : The peak to peak jitter due to uncorrelated periodic sources.

      - rj       : The standard deviation of the jitter due to uncorrelated unbounded random sources.

      - tie_ind  : The data independent jitter.

    Notes:

      - Any delay present in the actual crossings, relative to the ideal crossings,
        should be removed, before calling this function.

    """

    jitter   = []
    t_jitter = []
    i        = 0
    # Assemble the TIE track.
    for actual_xing in actual_xings:
        # Check for multiple crossings and skip them.
        if(actual_xing < (ideal_xings[i] - ui / 2.)):
            continue
        # Check for missed crossings and zero fill, as necessary.
        while(i < len(ideal_xings) and (actual_xing - ideal_xings[i]) > (ui / 2.)):
            for j in range(2): # If we missed one crossing, then we missed two.
                if(i >= len(ideal_xings)):
                    if(debug):
                        print "Oops! Ran out of 'ideal_xings' entries while correcting for missed crossings."
                    break
                jitter.append(0.)
                t_jitter.append(ideal_xings[i])
                i += 1
        if(i < len(ideal_xings)):
            jitter.append(actual_xing - ideal_xings[i])
            t_jitter.append(ideal_xings[i])
        i += 1
        if(i >= len(ideal_xings)):
            if(debug):
                print "Oops! Ran out of 'ideal_xings' entries. (i = %d, len(ideal_xings) = %d, len(jitter) = %d, len(actual_xings) = %d)" \
                        % (i, len(ideal_xings), len(jitter), len(actual_xings))
                print "\tLast ideal xing: %e;   last actual xing: %e." % (ideal_xings[-1], actual_xings[-1])
            break
    jitter  = array(jitter)
    if(debug):
        print "mean(jitter):", mean(jitter)
        print "len(jitter):", len(jitter)
    jitter -= mean(jitter)
    # Separate the rising and falling edges, shaped appropriately for averaging over the pattern period.
    # - We have to be careful to keep the last crossing, in the case where there are an odd number of them,
    #   because we'll be assembling a "repeated average" vector, later, and subtracting it from the original
    #   jitter vector. So, we can't get sloppy, or we'll end up with misalignment between the two.
    try:
        xings_per_pattern    = where(ideal_xings > pattern_len * ui)[0][0]
    except:
        print "ideal_xings:", ideal_xings
        raise
    assert not (xings_per_pattern % 2), "Odd number of crossings per pattern detected!"
    risings_per_pattern  = xings_per_pattern // 2
    fallings_per_pattern = xings_per_pattern // 2
    num_patterns         = nbits // pattern_len - 1
    jitter = jitter[xings_per_pattern:] # The first pattern period is problematic.
    if(len(jitter) < xings_per_pattern * num_patterns):
        jitter = np.append(jitter, zeros(xings_per_pattern * num_patterns - len(jitter)))
    try:
        t_jitter = t_jitter[:len(jitter)]
    except:
        print "jitter:", jitter
        raise
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
    tie_ave  = resize(tie_ave, len(jitter))
    try:
        tie_ind  = jitter - tie_ave
    except:
        print "tie_ave:", tie_ave
        raise

    if(debug):
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
    # - We'll call any spectral component with a magnitude > 6-sigma a "peak".
    thresh   = 6 * y_sigma
    y_per    = where(y_mag > thresh, y, zeros(len(y)))

    if(debug):
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

    return (jitter, t_jitter, isi, dcd, pj, rj, tie_ind)

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

    assert len(t) == len(jitter), "Length of t (%d) and jitter (%d) must be equal!" % (len(t), len(jitter))

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

def calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w):
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
      - w           frequency sample points vector

    Outputs:
      - gamma       frequency dependent propagation constant
      - Zc          frequency dependent characteristic impedance
    """

    # Guard against /0.
    if(w[0] == 0):
        w[0] = 1.e-12

    Rac   = R0 * sqrt(2 * 1j * w / w0)                        # AC resistance vector
    R     = sqrt(np.power(Rdc, 2) + np.power(Rac, 2))         # total resistance vector
    L0    = Z0 / v0                                           # "external" inductance per unit length (H/m)
    C0    = 1. / (Z0 * v0)                                    # nominal capacitance per unit length (F/m)
    C     = C0 * np.power((1j * w / w0), (-2. * Theta0 / pi)) # complex capacitance per unit length (F/m)
    gamma = sqrt((1j * w * L0 + R) * (1j * w * C))            # propagation constant (nepers/m)
    Zc    = sqrt((1j * w * L0 + R) / (1j * w * C))            # characteristic impedance (Ohms)

    return (gamma, Zc)

def calc_G(H, Rs, Cs, Zc, RL, Cp, CL, w):
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
      - w     frequency sample points vector

    Outputs:
      - G     frequency dependent transfer function of channel
    """

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

def calc_eye(ui, samps_per_bit, height, ys, clock_times=None):
    """
    Calculates the "eye" diagram of the input signal vector.

    Inputs:
      - ui             unit interval (s)
      - samps_per_bit  # of samples per bit
      - height         height of output image data array
      - ys             signal vector of interest
      - clock_times    (optional) vector of clock times to use for eye centers.
                       If not provided, just locate second zero-crossing (The first can be problematic.),
                       and assume constant UI and no phase jumps.
                       (This allows the same function to be used for eye diagram creation, for both pre and post-CDR signals.)

    Outputs:
      - img_array      The "heat map" representing the eye diagram.
                       Each grid location contains a value indicating
                       the number of times the signal passed through
                       that location.

    """

    # Intermediate variable calculation.
    tsamp = ui / samps_per_bit

    # Adjust the scaling.
    width    = 2 * samps_per_bit
    y_max    = 1.1 * max(abs(ys))
    y_scale  = height / (2 * y_max)          # (pixels/V)
    y_offset = height / 2                    # (pixels)

    # Generate the "heat" picture array.
    img_array = zeros([height, width])
    if(clock_times):
        for clock_time in clock_times:
            start_time = clock_time - ui
            stop_time  = clock_time + ui
            start_ix   = int(start_time / tsamp)
            interp_fac = (start_time - start_ix * tsamp) / tsamp
            last_y     = ys[start_ix]
            i = 0
            for (samp1, samp2) in zip(ys[start_ix : start_ix + 2 * samps_per_bit],
                                      ys[start_ix + 1 : start_ix + 1 + 2 * samps_per_bit]):
                y = samp1 + (samp2 - samp1) * interp_fac
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                if(sign(y) != sign(last_y)): # Trap zero crossings.
                    img_array[y_offset, int(i - 1 + y / (y - last_y) + 0.5)] += 1
                last_y = y
                i += 1
    else:
        start_ix      = where(diff(sign(ys)))[0][1] + 1 + samps_per_bit // 2 # The first crossing can be "off"; so, I use the second.
        last_start_ix = len(ys) - 2 * samps_per_bit
        while(start_ix < last_start_ix):
            last_y = ys[start_ix]
            i      = 0
            for y in ys[start_ix : start_ix + 2 * samps_per_bit]:
                img_array[int(y * y_scale + 0.5) + y_offset, i] += 1
                if(sign(y) != sign(last_y)): # Trap zero crossings.
                    img_array[y_offset, int(i - 1 + y / (y - last_y) + 0.5)] += 1
                last_y = y
                i += 1
            start_ix += samps_per_bit

    return img_array

