#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

The application source is divided among several files, as follows:

    pybert.py       - This file. It contains:
                      - independent variable declarations
                      - default initialization
                      - the definitions of those dependent variables, which are handled
                        automatically by the Traits/UI machinery.
                
    pybert_view.py  - Contains the main window layout definition, as
                      well as the definitions of user invoked actions
                      (i.e.- buttons).

    pybert_cntrl.py - Contains the definitions for those dependent
                      variables, which are updated not automatically by
                      the Traits/UI machinery, but rather by explicit
                      user action (i.e. - button clicks).

    pybert_util.py  - Contains general purpose utility functionality.

    dfe.py          - Contains the decision feedback equalizer model.

    cdr.py          - Contains the clock data recovery unit model.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from traits.api      import HasTraits, Array, Range, Float, Int, Property, String, cached_property, Instance, HTML, List, Bool, File
from chaco.api       import Plot, ArrayPlotData, VPlotContainer, GridPlotContainer, ColorMapper, Legend, OverlayPlotContainer, PlotAxis
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom
from numpy           import array, linspace, zeros, histogram, mean, diff, log10, transpose, shape
from numpy.fft       import fft
from numpy.random    import randint
from scipy.signal    import lfilter, iirfilter

from pybert_view     import traits_view
from pybert_cntrl    import my_run_simulation, update_results, update_eyes
from pybert_util     import calc_gamma, calc_G, trim_impulse, import_qucs_csv
from pybert_plot     import make_plots

debug = False

# Default model parameters - Modify these to customize the default simulation.
# - Simulation Control
gUI             = 100     # (ps)
gNbits          = 8000    # number of bits to run
gPatLen         = 127     # repeating bit pattern length
gNspb           = 32      # samples per bit
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
    posttap         = Float(-0.10)
    # - Rx
    rin             = Float(gRin)                                           # (Ohmin)
    cin             = Float(gCin)                                           # (pF)
    cac             = Float(gCac)                                           # (uF)
    rx_bw           = Float(gBW)                                            # (GHz)
    use_dfe         = Bool(gUseDfe)
    sum_ideal       = Bool(gDfeIdeal)
    peak_freq       = Float(gPeakFreq)                                      # CTLE peaking frequency (GHz)
    peak_mag        = Float(gPeakMag)                                       # CTLE peaking magnitude (dB)
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
    # - About
    ident  = String('PyBERT v1.4 - a serial communication link design tool, written in Python\n\n \
    David Banas\n \
    February 15, 2015\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')
    # - Help
    instructions = Property(HTML)

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
    jitter_info     = Property(HTML,    depends_on=['jitter_perf'])
    perf_info       = Property(HTML,    depends_on=['total_perf'])
    status_str      = Property(String,  depends_on=['status'])
    # - Handled by pybert_cntrl.py, upon user button clicks. (May contain "large overhead" variables.)
    #   - These are dependencies. So, they must be Array()s.
    #   - These are not.
    # Note: Everything has been moved to pybert_cntrl.py.
    #       I was beginning to suspect flaky initialization behavior,
    #       due to the way in which I was splitting up the initialization.
    #       Also, this guarantees no GUI freeze-up.
    # This is an experiment at bringing channel impulse definition back.
#    chnl_h          = Property(Array, depends_on=['use_ch_file', 'ch_file', 'Rdc', 'w0', 'R0', 'Theta0', 'Z0', 'v0', 'l_ch'])
#    t_ns_chnl       = Property(Array, depends_on=['t_ns', 'chnl_h'])
    chnl_h          = Property(Array)
    t_ns_chnl       = Property(Array)

    # Default initialization
    def __init__(self):
        """Plot setup occurs here."""

        super(PyBERT, self).__init__()

        plotdata = self.plotdata

        # Running the simulation will fill in the 'plotdata' structure.
        my_run_simulation(self, initial_run=True)

        # Once the 'plotdata' structure is filled in, we can create the plots.
        make_plots(self)

    # Dependent variable definitions
#    @cached_property
    def _get_chnl_h(self):
        print "Just entered _get_chnl_h()."
        if(self.use_ch_file):
            t                = self.t

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
            chnl_h, start_ix = trim_impulse(real(ifft(chnl_H)), Ts, chnl_dly)

        chnl_h   /= sum(chnl_h)

        self.chnl_dly        = chnl_dly
        self.chnl_H          = chnl_H
        self.start_ix        = start_ix

        return chnl_h

    @cached_property
    def _get_t_ns_chnl(self):
        start_ix  = self.start_ix
        t_ns      = self.t_ns
        chnl_h    = self.chnl_h

        return t_ns[start_ix : start_ix + len(chnl_h)]

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
        info_str += '      <TD align="center">Plotting</TD><TD>%6.3f</TD>\n'        % (self.plotting_perf * 60.e-6)
        info_str += '    </TR>\n'
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">TOTAL</TD><TD>%6.3f</TD>\n'           % (self.total_perf * 60.e-6)
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

if __name__ == '__main__':
    PyBERT().configure_traits(view=traits_view)

