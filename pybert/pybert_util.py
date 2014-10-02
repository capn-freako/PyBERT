# General purpose utilities for PyBERT.
#
# Original author: David Banas <capn.freako@gmail.com>
# Original date:   September 27, 2014 (Copied from `pybert_cntrl.py'.)
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
        # Check for missed crossings and zero fill, as necessary.
        while(i < len(ideal_xings) and (actual_xing - ideal_xings[i]) > ui):
            print "Fixing a missed crossing."
            for j in range(2): # If we missed one crossing, then we missed two.
                if(i >= len(ideal_xings)):
                    print "Oops! Ran out of 'ideal_xings' entries while correcting for missed crossings."
                    break
                jitter.append(0.)
                t_jitter.append(ideal_xings[i])
                i += 1
        if(i >= len(ideal_xings)):
            print "Oops! Ran out of 'ideal_xings' entries."
            break
        jitter.append(actual_xing - ideal_xings[i])
        t_jitter.append(ideal_xings[i])
        i += 1
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
    if(False):
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

    #if(debug):
    if(False):
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


