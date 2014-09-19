#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from traits.api      import HasTraits, Array, Range, Float, Int, Property, String, cached_property, Instance
from chaco.api       import Plot, ArrayPlotData, VPlotContainer, GridPlotContainer, ColorMapper
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom
from numpy           import array, linspace, zeros, histogram, mean, diff, log10, transpose, shape
from numpy.fft       import fft
from numpy.random    import randint
from scipy.signal    import lfilter, iirfilter

from pybert_view  import *
from pybert_cntrl import *

# Default model parameters - Modify these to customize the default simulation.
gNtaps          = 5
gGain           = 0.1
gNave           = 100
gDeltaT         = 0.1     # (ps)
gAlpha          = 0.01
gNLockAve       = 500     # number of UI used to average CDR locked status.
gRelLockTol     = .1      # relative lock tolerance of CDR.
gLockSustain    = 500
gUI             = 100     # (ps)
gDecisionScaler = 0.5
gNbits          = 10000   # number of bits to run
gPatLen         = 127     # repeating bit pattern length
gNspb           = 100      # samples per bit
gFc             = 100.0     # default channel cut-off frequency (GHz)
gFc_min         = 0.001   # min. channel cut-off frequency (GHz)
gFc_max         = 100.0   # max. channel cut-off frequency (GHz)
gNch_taps       = 3       # number of taps in IIR filter representing channel
gRj             = 0.001   # standard deviation of Gaussian random jitter (ps)
gSjMag          = 5.      # magnitude of periodic jitter (ps)
gSjFreq         = 5.    # frequency of periodic jitter (MHz)

class PyBERT(HasTraits):
    """
    A serial communication link bit error rate tester (BERT) simulator with a GUI interface.
    
    Useful for exploring the concepts of serial communication link design.
    """

    # Independent variables
    ui     = Float(gUI)                                        # (ps)
    gain   = Float(gGain)
    n_ave  = Float(gNave)
    n_taps = Int(gNtaps)
    nbits  = Int(gNbits)
    pattern_len = Int(gPatLen)
    nspb   = Int(gNspb)
    decision_scaler = Float(gDecisionScaler)
    delta_t         = Float(gDeltaT)                           # (ps)
    alpha           = Float(gAlpha)
    n_lock_ave      = Int(gNLockAve)
    rel_lock_tol    = Float(gRelLockTol)
    lock_sustain    = Int(gLockSustain)
    fc      = Range(low = 0., value = gFc, exclude_low = True) # (GHz)
    rj      = Float(gRj)                                       # (ps)
    sj_mag  = Float(gSjMag)                                    # (ps)
    sj_freq = Float(gSjFreq)                                   # (MHz)
    plot_out = Instance(VPlotContainer)
    plot_in  = Instance(GridPlotContainer)
    plot_dfe = Instance(GridPlotContainer)
    plot_eye = Instance(GridPlotContainer)
    eye_bits = Int(4000)
    status       = String("Ready.")
    channel_perf = Float(1.)
    cdr_perf     = Float(1.)
    dfe_perf     = Float(1.)
    ident  = String('PyBERT v0.1 - a serial communication link design tool, written in Python\n\n \
    David Banas\n \
    August 24, 2014\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')

    # Dependent variables
    bits     = Property(Array, depends_on=['nbits', 'pattern_len'])
    npts     = Property(Array, depends_on=['nbits', 'nspb'])
    eye_offset = Property(Int, depends_on=['nspb'])
    t        = Property(Array, depends_on=['ui', 'npts', 'nspb'])
    t_ns     = Property(Array, depends_on=['t'])
    fs       = Property(Array, depends_on=['ui', 'nspb'])
    a        = Property(Array, depends_on=['fc', 'fs'])
    b        = array([1.]) # Will be set by 'a' handler, upon change in dependencies.
    h        = Property(Array, depends_on=['npts', 'a'])
    crossing_times_chnl_out = Property(Array, depends_on=['chnl_out'])
    jitter                  = Property(Array, depends_on=['crossing_times_chnl_out'])
    jitter_spectrum         = Property(Array, depends_on=['jitter'])
    jitter_rejection_ratio  = Property(Array, depends_on=['run_result'])
    status_str = Property(String, depends_on=['status', 'channel_perf', 'cdr_perf', 'dfe_perf', 'jitter'])

    # Handler set variables
    chnl_out    = Array()
    run_result  = Array()
    adaptation  = Array()
    ui_ests     = Array()
    clocks      = Array()
    lockeds     = Array()
    t_jitter    = array([0.])

    # Default initialization
    def __init__(self):
        super(PyBERT, self).__init__()

        ui    = self.ui
        nbits = self.nbits
        nspb  = self.nspb
        eye_offset = self.eye_offset

        # Initialize `self.plotdata' with everything that doesn't depend on `chnl_out'.
        plotdata = ArrayPlotData(t_ns            = self.t_ns,
                                 clocks          = self.clocks,
                                 lockeds         = self.lockeds,
                                 ui_ests         = self.ui_ests,
                                 dfe_out         = self.run_result,
                                 t_jitter        = self.t_jitter,
                                 # The following are necessary, since we can't run `update_results()', until the plots have been created.
                                 imagedata       = zeros([100, 2 * gNspb]),
                                 eye_index       = linspace(-gUI, gUI, 2 * gNspb),
                                 zero_xing_pdf   = zeros(2 * gNspb),
                                 bathtub         = zeros(2 * gNspb),
                                )
        self.plotdata = plotdata
        # Then, run the channel and the DFE, which includes the CDR implicitly, to generate the rest.
        my_run_channel(self)
        my_run_dfe(self)

        # Now, create all the various plots we need for our GUI.
        plot1 = Plot(plotdata)
        #plot1.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot1.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot1.plot(("t_ns", "clocks"), type="line", color="green")
        plot1.plot(("t_ns", "lockeds"), type="line", color="red")
        #plot1.title  = "Channel Output, Recovered Clocks, & Locked"
        plot1.title  = "DFE Output, Recovered Clocks, & Locked"
        plot1.index_axis.title = "Time (ns)"
        plot1.tools.append(PanTool(plot1, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom1 = ZoomTool(plot1, tool_mode="range", axis='index', always_on=False)
        plot1.overlays.append(zoom1)

        plot2        = Plot(plotdata)
        plot2.plot(("t_ns", "ui_ests"), type="line", color="blue")
        plot2.title  = "CDR Adaptation"
        plot2.index_axis.title = "Time (ns)"
        plot2.value_axis.title = "UI (ps)"
        plot2.index_range = plot1.index_range # Zoom x-axes in tandem.

        plot3        = Plot(plotdata)
        plot3.plot(('f_MHz', 'jitter_rejection_ratio'), type="line", color="blue")
        plot3.title  = "CDR Jitter Rejection Ratio"
        plot3.index_axis.title = "Frequency (MHz)"
        plot3.value_axis.title = "Ratio (dB)"
        zoom3 = ZoomTool(plot3, tool_mode="range", axis='index', always_on=False)
        plot3.overlays.append(zoom3)

        plot5        = Plot(plotdata)
        plot5.plot(('tie_hist_bins', 'tie_hist_counts'), type="line", color="auto")
        plot5.title  = "Jitter Distribution"
        plot5.index_axis.title = "Time (ps)"
        plot5.value_axis.title = "Count"
        zoom5 = ZoomTool(plot5, tool_mode="range", axis='index', always_on=False)
        plot5.overlays.append(zoom5)

        plot6        = Plot(plotdata)
        plot6.plot(('f_MHz', 'jitter_spectrum'), type="line", color="auto")
        plot6.title  = "Jitter Spectrum"
        plot6.index_axis.title = "Frequency (MHz)"
        plot6.value_axis.title = "|FFT(jitter)| (dBui)"
        zoom6 = ZoomTool(plot6, tool_mode="range", axis='index', always_on=False)
        plot6.overlays.append(zoom6)

        plot7 = Plot(plotdata)
        plot7.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot7.title  = "Channel Output"
        plot7.index_axis.title = "Time (ns)"
        plot7.tools.append(PanTool(plot7, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom7 = ZoomTool(plot7, tool_mode="range", axis='index', always_on=False)
        plot7.overlays.append(zoom7)

        plot4        = Plot(plotdata)
        plot4.plot(("t_jitter", "jitter"), type="line", color="blue")
        plot4.title  = "Channel Output Jitter"
        plot4.index_axis.title = "Time (ns)"
        plot4.value_axis.title = "Jitter (ps)"
        #plot4.value_range.high_setting = ui / 2.
        #plot4.value_range.low_setting  = -ui / 2.
        plot4.index_range = plot7.index_range # Zoom x-axes in tandem.

        plot9 = Plot(plotdata)
        for i in range(gNtaps):
            plot9.plot(("tap_weight_index", "tap%d_weights" % (i + 1)), type="line", color="auto", name="tap%d"%(i+1))
        plot9.title  = "DFE Adaptation"
        plot9.tools.append(PanTool(plot9, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom9 = ZoomTool(plot9, tool_mode="range", axis='index', always_on=False)
        plot9.overlays.append(zoom9)
        plot9.legend.visible = True
        plot9.legend.align = 'ul'

        seg_map = dict(
            red = [
                (0.00, 0.00, 0.00), # black
                (0.00001, 0.00, 0.00), # blue
                (0.15, 0.00, 0.00), # cyan
                (0.30, 0.00, 0.00), # green
                (0.45, 1.00, 1.00), # yellow
                (0.60, 1.00, 1.00), # orange
                (0.75, 1.00, 1.00), # red
                (0.90, 1.00, 1.00), # pink
                (1.00, 1.00, 1.00) # white
            ],
            green = [
                (0.00, 0.00, 0.00), # black
                (0.00001, 0.00, 0.00), # blue
                (0.15, 0.50, 0.50), # cyan
                (0.30, 0.50, 0.50), # green
                (0.45, 1.00, 1.00), # yellow
                (0.60, 0.50, 0.50), # orange
                (0.75, 0.00, 0.00), # red
                (0.90, 0.50, 0.50), # pink
                (1.00, 1.00, 1.00) # white
            ],
            blue = [
                (0.00, 0.00, 0.00), # black
                (1e-18, 0.50, 0.50), # blue
                (0.15, 0.50, 0.50), # cyan
                (0.30, 0.00, 0.00), # green
                (0.45, 0.00, 0.00), # yellow
                (0.60, 0.00, 0.00), # orange
                (0.75, 0.00, 0.00), # red
                (0.90, 0.50, 0.50), # pink
                (1.00, 1.00, 1.00) # white
            ]
        )
        clr_map = ColorMapper.from_segment_map(seg_map)
        self.clr_map = clr_map
        plot10 = Plot(plotdata)
        plot10.img_plot("imagedata", 
            colormap=clr_map,
        )
        plot10.y_direction = 'normal'
        plot10.components[0].y_direction = 'normal'
        plot10.title  = "Eye Diagram"
        plot10.x_axis.title = "Time (ps)"
        plot10.x_axis.orientation = "bottom"
        plot10.y_axis.title = "Slicer Input (V)"
        plot10.x_grid.visible = True
        plot10.y_grid.visible = True
        plot10.x_grid.line_color = 'gray'
        plot10.y_grid.line_color = 'gray'

        plot11 = Plot(plotdata)
        plot11.plot(("eye_index", "zero_xing_pdf"), type="line", color="blue")
        plot11.title  = "Zero Crossing Probability Density Function"
        plot11.index_axis.title = "Time (ps)"

        plot12 = Plot(plotdata)
        plot12.plot(("eye_index", "bathtub"), type="line", color="blue")
        plot12.title  = "Bathtub Curves"
        plot12.index_axis.title = "Time (ps)"

        # And assemble them into the appropriate tabbed containers.
        container_in  = GridPlotContainer(shape=(2,2))
        container_in.add(plot7)
        container_in.add(plot4)
        container_in.add(plot5)
        container_in.add(plot6)
        self.plot_in  = container_in
        #container_dfe = VPlotContainer(plot8, plot9)
        container_dfe = GridPlotContainer(shape=(2,2))
        container_dfe.add(plot2)
        container_dfe.add(plot9)
        container_dfe.add(plot1)
        container_dfe.add(plot3)
        self.plot_dfe = container_dfe
        container_eye  = GridPlotContainer(shape=(2,2))
        container_eye.add(plot10)
        container_eye.add(plot12)
        container_eye.add(plot11)
        self.plot_eye  = container_eye

        # Update the `Results' tab.
        update_results(self)

    # Dependent variable definitions
    @cached_property
    def _get_bits(self):
        return resize(array([0, 1] + [randint(2) for i in range(self.pattern_len - 2)]), self.nbits)
    
    @cached_property
    def _get_t(self):
        t0   = (self.ui * 1.e-12) / self.nspb
        npts = self.npts
        return [i * t0 for i in range(npts)]
    
    @cached_property
    def _get_t_ns(self):
        return 1.e9 * array(self.t)
    
    @cached_property
    def _get_npts(self):
        return self.nbits * self.nspb
    
    @cached_property
    def _get_eye_offset(self):
        return self.nspb / 2
    
    @cached_property
    def _get_fs(self):
        return self.nspb / (self.ui * 1.e-12)
    
    @cached_property
    def _get_a(self):
        fc     = self.fc * 1.e9 # User enters channel cutoff frequency in GHz.
        (b, a) = iirfilter(gNch_taps - 1, fc/(self.fs/2), btype='lowpass')
        self.b = b
        return a

    @cached_property
    def _get_h(self):
        x = array([0., 1.] + [0. for i in range(self.npts-2)])
        a = self.a
        b = self.b
        return lfilter(b, a, x)

    @cached_property
    def _get_crossing_times_chnl_out(self):
        return find_crossing_times(self.t, self.chnl_out, anlg=True)

    @cached_property
    def _get_jitter(self):
        """Calculate channel output jitter."""

        # Grab local copies of class instance variables.
        ideal_xings     = array(self.ideal_xings)
        actual_xings    = array(self.crossing_times_chnl_out)
        ui              = self.ui * 1.e-12
        nbits           = self.nbits
        pattern_len     = self.pattern_len

        # Calculate and correct channel delay.
        # We insert a "10 '0's / 10 '1's" sequence at the beginning of the bit stream, in order to ensure that this works.
        dly = actual_xings[0] - ideal_xings[0]
        actual_xings    -= dly

        # Calculate jitter and its histogram.
        (jitter, t_jitter, isi, dcd, pj, rj) = calc_jitter(ui, nbits, pattern_len, ideal_xings, actual_xings)
        hist, bin_edges      = histogram(jitter, 99, (-ui/2., ui/2.))
        bin_centers          = [mean([bin_edges[i], bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]

        # Store class instance variables.
        self.channel_delay   = dly
        self.t_jitter        = t_jitter
        self.tie_hist_counts = hist
        self.tie_hist_bins   = bin_centers
        self.isi_chnl        = isi
        self.dcd_chnl        = dcd
        self.pj_chnl         = pj
        self.rj_chnl         = rj

        return list(jitter)

    @cached_property
    def _get_jitter_spectrum(self):
        jitter      = self.jitter
        t_jitter    = array(self.t_jitter)
        ui          = self.ui * 1.e-12
        nbits       = self.nbits

        (f, y) = calc_jitter_spectrum(t_jitter, jitter, ui, nbits)

        self.f_MHz  = array(f) * 1.e-6
        return 10. * log10(y)

    @cached_property
    def _get_jitter_rejection_ratio(self):
        """Calculate the jitter rejection ratio (JRR) of the CDR/DFE, by comparing
        the jitter in the signal before and after passage through the DFE."""

        # Copy class instance values into local storage.
        res         = self.run_result
        ui          = self.ui * 1.e-12
        nbits       = self.nbits
        pattern_len = self.pattern_len
        eye_bits    = self.eye_bits
        t           = self.t
        clocks      = array(t)[where(array(self.clocks) == 1)[0]]
        ideal_xings = array(self.ideal_xings)

        # Get the actual crossings of interest.
        xings   = find_crossing_times(t, res, anlg=True)
        dly     = xings[0] - ideal_xings[0]
        xings  -= dly
        ignore_until = (nbits - eye_bits) * ui
        i       = 0
        while(xings[i] < ignore_until):
            i += 1
        xings = xings[i:]

        # Assemble the corresponding "ideal" crossings, based on the recovered clock times.
        half_ui     = ui / 2.
        ideal_xings = []
        for xing in xings:
            while(clocks[i] <= xing):
                i += 1
            ideal_xings.append(clocks[i] - half_ui)

        # Offset both vectors to begin at time 0, as required by calc_jitter().
        ideal_xings = array(ideal_xings) - ignore_until
        xings       = array(xings)       - ignore_until

        # Calculate the jitter and its spectrum.
        (jitter, t_jitter, isi, dcd, pj, rj) = calc_jitter(ui, eye_bits, pattern_len, ideal_xings, xings)
        (f, y)      = calc_jitter_spectrum(t_jitter, jitter, ui, nbits)
        f_MHz       = array(f) * 1.e-6
        assert f_MHz[1]   == self.f_MHz[1], "%e %e" % (f_MHz[1], self.f_MHz[1])
        assert len(f_MHz) == len(self.f_MHz)
        jitter_spectrum = 10. * log10(y)

        return self.jitter_spectrum - jitter_spectrum
        #return jitter_spectrum

    @cached_property
    def _get_status_str(self):
        jit_str  = "         | Jitter (ps):    ISI=%6.3f    DCD=%6.3f    Pj=%6.3f    Rj=%6.3f" % \
                     (self.isi_chnl * 1.e12, self.dcd_chnl * 1.e12, self.pj_chnl * 1.e12, self.rj_chnl * 1.e12)
        perf_str = "%-20s | Perf. (Msmpls/min.):    Channel = %4.1f    CDR = %4.1f    DFE = %4.1f    TOTAL = %4.1f" \
                % (self.status, self.channel_perf * 60.e-6, self.cdr_perf * 60.e-6, self.dfe_perf * 60.e-6, \
                   60.e-6 / (1 / self.channel_perf + 1 / self.dfe_perf))
        return perf_str + jit_str

    # Dynamic behavior definitions.
    def _chnl_out_changed(self):
        self.plotdata.set_data("chnl_out", self.chnl_out)
        self.plotdata.set_data("t_ns", self.t_ns)

    def _clocks_changed(self):
        self.plotdata.set_data("clocks", self.clocks)

    def _ui_ests_changed(self):
        self.plotdata.set_data("ui_ests", self.ui_ests)

    def _lockeds_changed(self):
        self.plotdata.set_data("lockeds", self.lockeds)

    def _jitter_changed(self):
        self.plotdata.set_data("jitter", array(self.jitter) * 1.e12)
        self.plotdata.set_data("t_jitter", array(self.t_jitter) * 1.e9)
        self.plotdata.set_data("tie_hist_bins", array(self.tie_hist_bins) * 1.e12)
        self.plotdata.set_data("tie_hist_counts", self.tie_hist_counts)

    def _jitter_spectrum_changed(self):
        self.plotdata.set_data("jitter_spectrum", self.jitter_spectrum[1:])
        self.plotdata.set_data("f_MHz", self.f_MHz[1:])

    def _jitter_rejection_ratio_changed(self):
        self.plotdata.set_data("jitter_rejection_ratio", self.jitter_rejection_ratio[1:])

    def _run_result_changed(self):
        self.plotdata.set_data("dfe_out", self.run_result)

    def _adaptation_changed(self):
        tap_weights = transpose(array(self.adaptation))
        i = 1
        for tap_weight in tap_weights:
            self.plotdata.set_data("tap%d_weights" % i, tap_weight)
            i += 1
        self.plotdata.set_data("tap_weight_index", range(len(tap_weight)))

if __name__ == '__main__':
    PyBERT().configure_traits(view=traits_view)

