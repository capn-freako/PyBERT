#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

The application source is divided among several files, as follows:

    pybert.py       - This file. The M in MVC, it contains:
                      - independent variable declarations
                      - default initialization
                      - the definitions of those dependent variables, which are handled
                        automatically by the Traits/UI machinery.
                
    pybert_view.py  - The V in MVC, it contains the main window layout definition, as
                      well as the definitions of user invoked actions
                      (i.e.- buttons).

    pybert_cntrl.py - The C in MVC, it contains the definitions for those dependent
                      variables, which are updated not automatically by
                      the Traits/UI machinery, but rather by explicit
                      user action (i.e. - button clicks).

    pybert_plot.py  - Contains all plot definitions.

    pybert_util.py  - Contains general purpose utility functionality.

    dfe.py          - Contains the decision feedback equalizer model.

    cdr.py          - Contains the clock data recovery unit model.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from pylab           import *

from traits.api      import HasTraits, Array, Range, Float, Int, Property, String, cached_property, Instance, HTML, List, Bool, File
from traitsui.api    import View, Item, Group
from enable.component_editor import ComponentEditor
from chaco.api       import Plot, ArrayPlotData, VPlotContainer, GridPlotContainer, ColorMapper, Legend, OverlayPlotContainer, PlotAxis
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom
from numpy           import array, linspace, zeros, histogram, mean, diff, log10, transpose, shape, exp, real
from numpy.fft       import fft, ifft
from numpy.random    import randint
from scipy.signal    import lfilter, iirfilter

from pybert_view     import traits_view
from pybert_cntrl    import my_run_simulation, update_results, update_eyes
from pybert_util     import calc_gamma, calc_G, trim_impulse, import_qucs_csv, make_ctle, trim_shift_scale
from pybert_plot     import make_plots

debug = False

# Default model parameters - Modify these to customize the default simulation.
# - Simulation Control
gUI             = 100     # (ps)
gNbits          = 8000    # number of bits to run
gPatLen         = 127     # repeating bit pattern length
gNspb           = 32      # samples per bit
gNumAve         = 1       # Number of bit error samples to average, when sweeping.
# - Channel Control
#     - parameters for Howard Johnson's "Metallic Transmission Model"
#     - (See "High Speed Signal Propagation", Sec. 3.1.)
#     - ToDo: These are the values for 24 guage twisted copper pair; need to add other options.
gRdc            = 0.1876  # Ohms/m
gw0             = 10.e6   # 10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
gR0             = 1.452   # skin-effect resistance (Ohms/m)
gTheta0         = .02     # loss tangent
gZ0             = 100.    # characteristic impedance in LC region (Ohms)
gv0             = 0.67    # relative propagation velocity (c)
gl_ch           = 1.0     # cable length (m)
gRn             = 0.01    # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)
# - Tx
gVod            = 1.0     # output drive strength (Vp)
gRs             = 100     # differential source impedance (Ohms)
gCout           = 0.50    # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gPnMag          = 0.1     # magnitude of periodic noise (V)
gPnFreq         = 0.437   # frequency of periodic noise (MHz)
# - Rx
gRin            = 100     # differential input resistance
gCin            = 0.50    # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gCac            = 1.      # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
gBW             = 12.     # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseCtle        = True    # Include CTLE when running simulation.
gUseDfe         = True    # Include DFE when running simulation.
gDfeIdeal       = True    # DFE ideal summing node selector
gPeakFreq       = 5.      # CTLE peaking frequency (GHz)
gPeakMag        = 10.     # CTLE peaking magnitude (dB)
# - DFE
gDecisionScaler = 0.5
gNtaps          = 5
gGain           = 0.1
gNave           = 100
gDfeBW          = 12.     # DFE summing node bandwidth (GHz)
# - CDR
gDeltaT         = 0.1     # (ps)
gAlpha          = 0.01
gNLockAve       = 500     # number of UI used to average CDR locked status.
gRelLockTol     = .1      # relative lock tolerance of CDR.
gLockSustain    = 500
# - Analysis
gThresh         = 6       # threshold for identifying periodic jitter spectral elements (sigma)

class PyBERT(HasTraits):
    """
    A serial communication link bit error rate tester (BERT) simulator with a GUI interface.
    
    Useful for exploring the concepts of serial communication link design.
    """

    # Independent variables
    # - Simulation Control
    ui              = Float(gUI)                                            # (ps)
    nbits           = Int(gNbits)
    pattern_len     = Int(gPatLen)
    nspb            = Int(gNspb)
    eye_bits        = Int(gNbits // 5)
    mod_type        = List([0])
    num_sweeps      = Int(1)
    sweep_num       = Int(1)
    sweep_aves      = Int(gNumAve)
    do_sweep        = Bool(False)
    # - Channel Control
    use_ch_file     = Bool(False)
    ch_file         = File('', entries=5, filter=['*.csv'])
    Rdc             = Float(gRdc)
    w0              = Float(gw0)
    R0              = Float(gR0)
    Theta0          = Float(gTheta0)
    Z0              = Float(gZ0)
    v0              = Float(gv0)
    l_ch            = Float(gl_ch)
    # - Tx
    vod             = Float(gVod)                                           # (V)
    rs              = Float(gRs)                                            # (Ohms)
    cout            = Float(gCout)                                          # (pF)
    pn_mag          = Float(gPnMag)                                         # (ps)
    pn_freq         = Float(gPnFreq)                                        # (MHz)
    rn              = Float(gRn)                                            # (V)
    pretap          = Float(-0.05)
    pretap_sweep    = Bool(False)
    pretap_final    = Float(-0.05)
    pretap_steps    = Int(5)
    posttap         = Float(-0.10)
    posttap_sweep   = Bool(False)
    posttap_final   = Float(-0.10)
    posttap_steps   = Int(10)
    posttap2        = Float(0.0)
    posttap2_sweep  = Bool(False)
    posttap2_final  = Float(0.0)
    posttap2_steps  = Int(10)
    posttap3        = Float(0.0)
    posttap3_sweep  = Bool(False)
    posttap3_final  = Float(0.0)
    posttap3_steps  = Int(10)
    pretap_tune     = Float(0.0)
    posttap_tune    = Float(0.0)
    posttap2_tune   = Float(0.0)
    posttap3_tune   = Float(0.0)
    # - Rx
    rin             = Float(gRin)                                           # (Ohmin)
    cin             = Float(gCin)                                           # (pF)
    cac             = Float(gCac)                                           # (uF)
    rx_bw           = Float(gBW)                                            # (GHz)
    use_agc         = Bool(True)
    use_dfe         = Bool(gUseDfe)
    sum_ideal       = Bool(gDfeIdeal)
    peak_freq       = Float(gPeakFreq)                                      # CTLE peaking frequency (GHz)
    peak_mag        = Float(gPeakMag)                                       # CTLE peaking magnitude (dB)
    rx_bw_tune      = Float(gBW)
    peak_freq_tune  = Float(gPeakFreq)
    peak_mag_tune   = Float(gPeakMag)
    # - DFE
    decision_scaler = Float(gDecisionScaler)
    gain            = Float(gGain)
    n_ave           = Float(gNave)
    n_taps          = Int(gNtaps)
    sum_bw          = Float(gDfeBW)                                         # (GHz)
    # - CDR
    delta_t         = Float(gDeltaT)                                        # (ps)
    alpha           = Float(gAlpha)
    n_lock_ave      = Int(gNLockAve)
    rel_lock_tol    = Float(gRelLockTol)
    lock_sustain    = Int(gLockSustain)
    # - Analysis
    thresh          = Int(gThresh)
    # - Plots (plot containers, actually)
    plotdata          = ArrayPlotData()
    plots_h           = Instance(GridPlotContainer)
    plots_s           = Instance(GridPlotContainer)
    plots_p           = Instance(GridPlotContainer)
    plots_H           = Instance(GridPlotContainer)
    plots_dfe         = Instance(GridPlotContainer)
    plots_eye         = Instance(GridPlotContainer)
    plots_jitter_dist = Instance(GridPlotContainer)
    plots_jitter_spec = Instance(GridPlotContainer)
    plots_bathtub     = Instance(GridPlotContainer)
    # - Status
    status          = String("Ready.")
    jitter_perf     = Float(0.)
    total_perf      = Float(0.)
    sweep_results   = List([])
    # - About
    ident  = String('PyBERT v1.5 - a serial communication link design tool, written in Python\n\n \
    David Banas\n \
    March 22, 2015\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')
    # - Help
    instructions = Property()

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
    #
    # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
    #         This will slow the GUI down noticeably.
    jitter_info     = Property(HTML,    depends_on=['jitter_perf'])
    perf_info       = Property(HTML,    depends_on=['total_perf'])
    status_str      = Property(String,  depends_on=['status'])
    sweep_info      = Property(HTML,    depends_on=['sweep_results'])
    tx_h_tune       = Property(Array,   depends_on=['pretap_tune', 'posttap_tune', 'posttap2_tune', 'posttap3_tune'])
    ctle_h_tune     = Property(Array,   depends_on=['peak_freq_tune', 'peak_mag_tune', 'rx_bw_tune'])
    ctle_out_h_tune = Property(Array,   depends_on=['tx_h_tune', 'ctle_h_tune'])
    t               = Property(Array,   depends_on=['ui', 'nspb', 'nbits'])
    t_ns            = Property(Array,   depends_on=['t'])
    f               = Property(Array,   depends_on=['t'])
    w               = Property(Array,   depends_on=['f'])
    bits            = Property(Array,   depends_on=['pattern_len', 'nbits'])
    symbols         = Property(Array,   depends_on=['bits', 'mod_type', 'vod'])
    ffe             = Property(Array,   depends_on=['pretap', 'posttap', 'posttap2', 'posttap3'])

    # Default initialization
    def __init__(self, run_simulation = True):
        """
        Initial plot setup occurs here.

        In order to populate the data structure we need to
        construct the plots, we must run the simulation.

        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(PyBERT, self).__init__()

        if(run_simulation):
            # Running the simulation will fill in the required data structure.
            my_run_simulation(self, initial_run=True)

            # Once the required data structure is filled in, we can create the plots.
            make_plots(self, n_dfe_taps = gNtaps)

    # Dependent variable definitions
    @cached_property
    def _get_t(self):
        """
        Calculate the system time vector, in seconds.

        """

        ui    = self.ui * 1.e-12
        nspb  = self.nspb
        nbits = self.nbits

        t0   = ui / nspb
        npts = nbits * nspb

        return array([i * t0 for i in range(npts)])
    
    @cached_property
    def _get_t_ns(self):
        """
        Calculate the system time vector, in ns.
        """

        return self.t * 1.e9
    
    @cached_property
    def _get_f(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in Hz.
        # (i.e. - [0, f0, 2 * f0, ... , fN] + [-(fN - f0), -(fN - 2 * f0), ... , -f0]
        """

        t = self.t

        npts      = len(t)
        f0        = 1. / (t[1] * npts)
        half_npts = npts // 2

        return array([i * f0 for i in range(half_npts + 1)] + [(half_npts - i) * -f0 for i in range(1, half_npts)])

    @cached_property
    def _get_w(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in rads./sec.
        """

        return 2 * pi * self.f
    
    @cached_property
    def _get_bits(self):
        """
        Generate the bit stream.
        """
        
        pattern_len     = self.pattern_len
        nbits           = self.nbits

        return resize(array([0, 1, 1] + [randint(2) for i in range(pattern_len - 3)]), nbits)

    @cached_property
    def _get_symbols(self):
        """
        Generate the symbol stream.
        """
        
        mod_type        = self.mod_type[0]
        vod             = self.vod
        bits            = self.bits

        if  (mod_type == 0):                         # NRZ
            symbols = 2 * bits - 1
        elif(mod_type == 1):                         # Duo-binary
            symbols = [bits[0]]
            for bit in bits[1:]:                       # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = array(symbols) - 0.5
        elif(mod_type == 2):                        # PAM-4
            symbols = []
            for bits in zip(bits[0::2], bits[1::2]):
                if(bits == [0,0]):
                    symbols.append(-1.)
                elif(bits == [0,1]):
                    symbols.append(-1./3.)
                elif(bits == [1,0]):
                    symbols.append(1.)
                else:
                    symbols.append(1./3.)
            symbols = repeat(array(symbols), 2)
        else:
            raise Exception("ERROR: _get_symbols(): Unknown modulation type requested!")

        return symbols * vod

    @cached_property
    def _get_ffe(self):
        """
        Generate the Tx pre-emphasis FIR numerator.
        """
        
        pretap   = self.pretap
        posttap  = self.posttap
        posttap2 = self.posttap2
        posttap3 = self.posttap3

        main_tap = 1.0 - abs(pretap) - abs(posttap) - abs(posttap2) - abs(posttap3)

        return [pretap, main_tap, posttap, posttap2, posttap3]

    @cached_property
    def _get_jitter_info(self):

        isi_chnl      = self.isi_chnl * 1.e12
        dcd_chnl      = self.dcd_chnl * 1.e12
        pj_chnl       = self.pj_chnl  * 1.e12
        rj_chnl       = self.rj_chnl  * 1.e12
        isi_tx        = self.isi_tx   * 1.e12
        dcd_tx        = self.dcd_tx   * 1.e12
        pj_tx         = self.pj_tx    * 1.e12
        rj_tx         = self.rj_tx    * 1.e12
        isi_ctle      = self.isi_ctle * 1.e12
        dcd_ctle      = self.dcd_ctle * 1.e12
        pj_ctle       = self.pj_ctle  * 1.e12
        rj_ctle       = self.rj_ctle  * 1.e12
        isi_dfe       = self.isi_dfe  * 1.e12
        dcd_dfe       = self.dcd_dfe  * 1.e12
        pj_dfe        = self.pj_dfe   * 1.e12
        rj_dfe        = self.rj_dfe   * 1.e12

        isi_rej_tx    = 1.e20
        dcd_rej_tx    = 1.e20
        pj_rej_tx     = 1.e20
        rj_rej_tx     = 1.e20
        isi_rej_ctle  = 1.e20
        dcd_rej_ctle  = 1.e20
        pj_rej_ctle   = 1.e20
        rj_rej_ctle   = 1.e20
        isi_rej_dfe   = 1.e20
        dcd_rej_dfe   = 1.e20
        pj_rej_dfe    = 1.e20
        rj_rej_dfe    = 1.e20
        isi_rej_total = 1.e20
        dcd_rej_total = 1.e20
        pj_rej_total  = 1.e20
        rj_rej_total  = 1.e20

        if(isi_tx):
            isi_rej_tx = isi_chnl / isi_tx
        if(dcd_tx):
            dcd_rej_tx = dcd_chnl / dcd_tx
        if(pj_tx):
            pj_rej_tx  = pj_chnl  / pj_tx
        if(rj_tx):
            rj_rej_tx  = rj_chnl  / rj_tx
        if(isi_ctle):
            isi_rej_ctle = isi_tx / isi_ctle
        if(dcd_ctle):
            dcd_rej_ctle = dcd_tx / dcd_ctle
        if(pj_ctle):
            pj_rej_ctle  = pj_tx  / pj_ctle
        if(rj_ctle):
            rj_rej_ctle  = rj_tx  / rj_ctle
        if(isi_dfe):
            isi_rej_dfe = isi_ctle / isi_dfe
        if(dcd_dfe):
            dcd_rej_dfe = dcd_ctle / dcd_dfe
        if(pj_dfe):
            pj_rej_dfe  = pj_ctle  / pj_dfe
        if(rj_dfe):
            rj_rej_dfe  = rj_ctle  / rj_dfe
        if(isi_dfe):
            isi_rej_total = isi_chnl / isi_dfe
        if(dcd_dfe):
            dcd_rej_total = dcd_chnl / dcd_dfe
        if(pj_dfe):
            pj_rej_total  = pj_tx  / pj_dfe
        if(rj_dfe):
            rj_rej_total  = rj_tx  / rj_dfe

        info_str = '<H1>Jitter Rejection by Equalization Component</H1>\n'

        info_str += '<H2>Tx Preemphasis</H2>\n'
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (isi_chnl, isi_tx, 10. * log10(isi_rej_tx))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (dcd_chnl, dcd_tx, 10. * log10(dcd_rej_tx))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % \
                      (pj_chnl, pj_tx)
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % \
                      (rj_chnl, rj_tx)
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += '<H2>CTLE</H2>\n'
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (isi_tx, isi_ctle, 10. * log10(isi_rej_ctle))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (dcd_tx, dcd_ctle, 10. * log10(dcd_rej_ctle))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (pj_tx, pj_ctle, 10. * log10(pj_rej_ctle))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (rj_tx, rj_ctle, 10. * log10(rj_rej_ctle))
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += '<H2>DFE</H2>\n'
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (isi_ctle, isi_dfe, 10. * log10(isi_rej_dfe))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (dcd_ctle, dcd_dfe, 10. * log10(dcd_rej_dfe))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (pj_ctle, pj_dfe, 10. * log10(pj_rej_dfe))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (rj_ctle, rj_dfe, 10. * log10(rj_rej_dfe))
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += '<H2>TOTAL</H2>\n'
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (isi_chnl, isi_dfe, 10. * log10(isi_rej_total))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (dcd_chnl, dcd_dfe, 10. * log10(dcd_rej_total))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (pj_tx, pj_dfe, 10. * log10(pj_rej_total))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (rj_tx, rj_dfe, 10. * log10(rj_rej_total))
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        return info_str
    
    @cached_property
    def _get_perf_info(self):

        info_str  = '<H2>Performance by Component</H2>\n'
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += '      <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>\n'
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Channel</TD><TD>%6.3f</TD>\n'         % (self.channel_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Tx Preemphasis</TD><TD>%6.3f</TD>\n'  % (self.tx_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">CTLE</TD><TD>%6.3f</TD>\n'            % (self.ctle_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">DFE</TD><TD>%6.3f</TD>\n'             % (self.dfe_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Jitter Analysis</TD><TD>%6.3f</TD>\n' % (self.jitter_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center"><strong>TOTAL</strong></TD><TD><strong>%6.3f</strong></TD>\n'           % (self.total_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Plotting</TD><TD>%6.3f</TD>\n'        % (self.plotting_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '  </TABLE>\n'

        return info_str

    @cached_property
    def _get_sweep_info(self):

        sweep_results = self.sweep_results

        info_str  = '<H2>Sweep Results</H2>\n'
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += '      <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>\n'
        info_str += '    </TR>\n'

        for item in sweep_results:
            info_str += '    <TR align="center">\n'
            info_str += '      <TD>%+06.3f</TD><TD>%+06.3f</TD><TD>%d</TD><TD>%d</TD>\n' % (item[0], item[1], item[2], item[3])
            info_str += '    </TR>\n'

        info_str += '  </TABLE>\n'

        return info_str

    @cached_property
    def _get_status_str(self):

        perf_str = "%-20s | Perf. (Msmpls/min.):    %4.1f" % (self.status, self.total_perf * 60.e-6)
        jit_str  = "         | Jitter (ps):    ISI=%6.3f    DCD=%6.3f    Pj=%6.3f    Rj=%6.3f" % \
                     (self.isi_dfe * 1.e12, self.dcd_dfe * 1.e12, self.pj_dfe * 1.e12, self.rj_dfe * 1.e12)
        dly_str  = "         | Channel Delay (ns):    %5.3f" % (self.chnl_dly * 1.e9)
        err_str  = "         | Bit errors detected: %d" % self.bit_errs
        return perf_str + dly_str + jit_str + err_str

    @cached_property
    def _get_instructions(self):

        help_str  = "<H2>PyBERT User's Guide</H2>\n"
        help_str += "  <H3>Note to developers</H3>\n"
        help_str += "    This is NOT for you. Instead, open 'pybert/doc/build/html/index.html' in a browser.\n"
        help_str += "  <H3>PyBERT User Help Options</H3>\n"
        help_str += "    <UL>\n"
        help_str += "      <LI>Hover over any user-settable value in the <em>Config.</em> tab, for help message.</LI>\n"
        help_str += '      <LI>Visit the PyBERT FAQ at: https://github.com/capn-freako/PyBERT/wiki/pybert_faq.</LI>\n'
        help_str += '      <LI>Send e-mail to David Banas at capn.freako@gmail.com.</LI>\n'
        help_str += "    </UL>\n"

        return help_str

    @cached_property
    def _get_tx_h_tune(self):

        nspui     = self.nspui
        pretap    = self.pretap_tune
        posttap   = self.posttap_tune
        posttap2  = self.posttap2_tune
        posttap3  = self.posttap3_tune

        main_tap = 1.0 - abs(pretap) - abs(posttap) - abs(posttap2) - abs(posttap3)
        ffe      = [pretap, main_tap, posttap, posttap2, posttap3]                    # FIR filter numerator, for fs = fbit.

        return concatenate([[x] + list(zeros(nspui - 1)) for x in ffe])

    @cached_property
    def _get_ctle_h_tune(self):

        w         = self.w
        chnl_h    = self.chnl_h
        rx_bw     = self.rx_bw_tune     * 1.e9
        peak_freq = self.peak_freq_tune * 1.e9
        peak_mag  = self.peak_mag_tune

        w_dummy, H = make_ctle(rx_bw, peak_freq, peak_mag, w)
        ctle_H     = H / abs(H[0])              # Scale to force d.c. component of '1'.

        return real(ifft(ctle_H))[:len(chnl_h)]

    @cached_property
    def _get_ctle_out_h_tune(self):

        ideal_h   = self.ideal_h
        chnl_h    = self.chnl_h
        tx_h      = self.tx_h_tune.copy()
        ctle_h    = self.ctle_h_tune

        tx_h.resize(len(chnl_h))
        tx_out_h   = convolve(tx_h,   chnl_h)  [:len(chnl_h)]
        ctle_out_h = convolve(ctle_h, tx_out_h)[:len(chnl_h)]

        self.ctle_out_g_tune = trim_shift_scale(ideal_h, ctle_out_h)

        return ctle_out_h

    # Changed property handlers.
    def _ctle_out_h_tune_changed(self):

        self.plotdata.set_data('ctle_out_h_tune', self.ctle_out_h_tune)
        self.plotdata.set_data('ctle_out_g_tune', self.ctle_out_g_tune)

    # These getters have been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control their times of execution. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    def calc_chnl_h(self):
        """
        Calculates the channel impulse response.

        Also sets, in 'self':
         - chnl_dly     group delay of channel
         - start_ix     first element of trimmed response
         - t_ns_chnl    the x-values, in ns, for plotting 'chnl_h'
         - chnl_H       channel frequency response
         - chnl_s       channel step response
         - chnl_p       channel pulse response

        """

        t                    = self.t
        nspui                = self.nspui

        if(self.use_ch_file):
            chnl_h           = import_qucs_csv(self.ch_file, self.Ts)
            chnl_dly         = t[where(chnl_h == max(chnl_h))[0][0]]
            chnl_h.resize(len(t))
            chnl_H           = fft(chnl_h)
            chnl_H          /= abs(chnl_H[0])
            chnl_h, start_ix = trim_impulse(chnl_h)
        else:
            l_ch             = self.l_ch
            v0               = self.v0 * 3.e8
            R0               = self.R0
            w0               = self.w0
            Rdc              = self.Rdc
            Z0               = self.Z0
            Theta0           = self.Theta0
            w                = self.w
            Rs               = self.rs
            Cs               = self.cout * 1.e-12
            RL               = self.rin
            Cp               = self.cin * 1.e-12
            CL               = self.cac * 1.e-6
            Ts               = self.Ts

            chnl_dly         = l_ch / v0
            gamma, Zc        = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            H                = exp(-l_ch * gamma)
            chnl_H           = 2. * calc_G(H, Rs, Cs, Zc, RL, Cp, CL, w) # Compensating for nominal /2 divider action.
            chnl_h           = real(ifft(chnl_H)) * sqrt(len(chnl_H))    # Correcting for '1/N' scaling in ifft().
            chnl_h, start_ix = trim_impulse(chnl_h, Ts, chnl_dly)

        chnl_h   /= sum(chnl_h)                                          # a temporary crutch.
        chnl_s    = chnl_h.cumsum()
        chnl_p    = chnl_s[nspui:] - chnl_s[:-nspui] 

        self.chnl_h          = chnl_h
        self.chnl_dly        = chnl_dly
        self.chnl_H          = chnl_H
        self.start_ix        = start_ix
        self.t_ns_chnl       = array(t[start_ix : start_ix + len(chnl_h)]) * 1.e9
        self.chnl_s          = chnl_s
        self.chnl_p          = chnl_p

        return chnl_h

if __name__ == '__main__':
    PyBERT().configure_traits(view = traits_view)

