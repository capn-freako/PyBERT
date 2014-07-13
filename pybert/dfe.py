#! /usr/bin/env python

"""
Behavioral model of a decision feedback equalizer (DFE).

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

This Python script provides a behavioral model of a decision feedback
equalizer (DFE). The class defined, here, is intended for integration
into the larger `PyBERT' framework, but is also capable of
running in stand-alone mode for preliminary debugging.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from traits.api      import HasTraits, Array, Range, Float, Int, Enum, Property, \
                            String, List, cached_property, Instance
from traitsui.api    import View, Item, VSplit, Group, VGroup, HGroup, Label, \
                            Action, Handler, DefaultOverride
from chaco.api       import Plot, ArrayPlotData, VPlotContainer, GridPlotContainer
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom, \
                            LineInspector, RangeSelection, RangeSelectionOverlay
from chaco.chaco_plot_editor import ChacoPlotItem
from enable.component_editor import ComponentEditor
from numpy        import arange, real, concatenate, angle, sign, sin, pi, array, \
                         float, zeros, repeat, histogram, mean, where, diff, log10, transpose
from numpy.fft    import fft, ifft
from numpy.random import randint, random, normal
from scipy.signal import lfilter, firwin, iirdesign, iirfilter, freqz
from cdr          import CDR
import re

# Default model parameters - Modify these to customize the default simulation.
gNpts           = 1024    # number of vector points
gNtaps          = 5
gGain           = 0.0
gNave           = 10
gDeltaT         = 0.1     # (ps)
gAlpha          = 0.01
gUI             = 100     # (ps)
gDecisionScaler = 1.0
gNbits          = 1000    # number of bits to run
gNspb           = 100     # samples per bit
gFc             = 50.0     # default channel cut-off frequency (GHz)
gFc_min         = 1.0     # min. channel cut-off frequency (GHz)
gFc_max         = 50.0    # max. channel cut-off frequency (GHz)
gNch_taps       = 2       # number of taps in IIR filter representing channel
gRj             = 0.001     # standard deviation of Gaussian random jitter (ps)
gSjMag          = 0.     # magnitude of periodic jitter (ps)
gSjFreq         = 100.    # frequency of periodic jitter (MHz)

# Runs the DFE adaptation, when user clicks button.
class MyHandler(Handler):

    def do_run_dfe(self, info):
        chnl_out        = info.object.chnl_out
        t               = info.object.t
        delta_t         = info.object.delta_t          # (ps)
        alpha           = info.object.alpha
        ui              = info.object.ui               # (ps)
        nspb            = info.object.nspb
        n_taps          = info.object.n_taps
        gain            = info.object.gain
        n_ave           = info.object.n_ave
        decision_scaler = info.object.decision_scaler
        dfe             = DFE(n_taps, gain, delta_t * 1.e-12, alpha, ui * 1.e-12, decision_scaler, n_ave)

        (res, tap_weights, ui_ests, clocks) = dfe.run(t, chnl_out)

        info.object.run_result = res
        info.object.adaptation = tap_weights
        info.object.ui_ests    = array(ui_ests) * 1.e12 # (ps)
        info.object.clocks     = clocks

run_dfe = Action(name="RunDFE", action="do_run_dfe")
    
# Model Proper - Don't change anything below this line.
class DFE(object):
    """Behavioral model of a decision feedback equalizer (DFE)."""

    def __init__(self, n_taps = gNtaps, gain = gGain, delta_t = gDeltaT,
                       alpha = gAlpha, ui = gUI, decision_scaler = gDecisionScaler, n_ave = gNave):
        """
        Inputs:

          - n_taps           # of taps in adaptive filter

          - gain             adaptive filter tap weight correction gain

          - delta_t          CDR proportional branch constant (ps)

          - alpha            CDR integral branch constant (normalized to delta_t)

          - ui               nominal unit interval (ps)

          - decision_scaler  multiplicative constant applied to the result of
                             the sign function, when making a "1 vs. 0" decision.
                             Sets the target magnitude for the DFE.
        """

        self.tap_weights       = [0.0] * n_taps
        self.tap_values        = [0.0] * n_taps
        self.gain              = gain
        self.ui                = ui
        self.decision_scaler   = decision_scaler
        self.cdr               = CDR(delta_t, alpha, ui)
        self.n_ave             = n_ave
        self.corrections       = zeros(n_taps)

    def step(self, decision, error, update):
        """Step the DFE, according to the new decision and error inputs."""

        # Copy class object variables into local function namespace, for efficiency.
        tap_weights = self.tap_weights
        tap_values  = self.tap_values
        gain        = self.gain
        n_ave       = self.n_ave

        # Calculate this step's corrections and add to running total.
        corrections = [old + new for (old, new) in zip(self.corrections,
                                                       [val * error * gain for val in tap_values])]

        # Update the tap weights with the average corrections, if appropriate.
        if(update):
            tap_weights = [weight + correction / n_ave for (weight, correction) in zip(tap_weights, corrections)]
            corrections = zeros(len(corrections)) # Start the averaging process over, again.

        # Step the filter delay chain and generate the new output.
        tap_values  = [decision] + tap_values[:-1]
        filter_out  = sum(array(tap_weights) * array(tap_values))

        # Copy local values back to their respective class object variables.
        self.tap_weights = tap_weights
        self.tap_values  = tap_values
        self.corrections = corrections

        return filter_out

    def run(self, sample_times, signal):
        """Run the DFE on the input signal."""

        ui                = self.ui
        decision_scaler   = self.decision_scaler
        n_ave             = self.n_ave

        clk_cntr           = 0
        smpl_cntr          = 0
        filter_out         = 0
        nxt_filter_out     = 0
        last_clock_sample  = 0
        next_boundary_time = 0
        next_clock_time    = ui / 2.

        res         = []
        tap_weights = []
        ui_ests     = []
        clocks      = zeros(len(sample_times))
        for (t, x) in zip(sample_times, signal):
            sum_out = x - filter_out
            res.append(sum_out)
            if(t >= next_boundary_time):
                boundary_sample = sum_out
                filter_out = nxt_filter_out
                next_boundary_time += ui
            if(t >= next_clock_time):
                clk_cntr += 1
                clocks[smpl_cntr] = 1
                current_clock_sample = sum_out
                ui = self.cdr.adapt([last_clock_sample, boundary_sample, current_clock_sample])
                decision = sign(x)
                error = sum_out - decision * decision_scaler
                update = (clk_cntr % n_ave) == 0
                nxt_filter_out = self.step(decision, error, update)
                if(update):
                    tap_weights.append(self.tap_weights)
                last_clock_sample  = sum_out
                next_boundary_time = next_clock_time + ui / 2.
                next_clock_time    = next_clock_time + ui
            ui_ests.append(ui)
            smpl_cntr += 1

        self.ui                = ui               

        return (res, tap_weights, ui_ests, clocks)

class DFEDemo(HasTraits):
    # Independent variables
    ui     = Float(gUI)        # (ps)
    gain   = Float(gGain)
    n_ave  = Float(gNave)
    decision_scaler = Float(gDecisionScaler)
    n_taps = Int(gNtaps)
    nbits  = Int(gNbits)
    nspb   = Int(gNspb)
    delta_t = Float(gDeltaT)   # (ps)
    alpha  = Float(gAlpha)
    fc     = Float(gFc)        # (GHz)
    rj     = Float(gRj)        # (ps)
    sj_mag = Float(gSjMag)     # (ps)
    sj_freq = Float(gSjFreq)   # (MHz)
    clocks = Array()
    plot_out = Instance(VPlotContainer)
    plot_in  = Instance(GridPlotContainer)
    plot_dfe = Instance(VPlotContainer)
    plot_eye  = Instance(VPlotContainer)
    ident  = String('PyDFE v0.1 - a DFE design tool, written in Python\n\n \
    David Banas\n \
    June 19, 2014\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')


    # Dependent variables
    bits     = Property(Array, depends_on=['nbits'])
    npts     = Property(Array, depends_on=['nbits', 'nspb'])
    eye_offset = Property(Int, depends_on=['nspb'])
    fs       = Property(Array, depends_on=['ui', 'nspb'])
    a        = Property(Array, depends_on=['fc', 'fs'])
    b        = array([1.]) # Will be set by 'a' handler, upon change in dependencies.
    h        = Property(Array, depends_on=['npts', 'a'])
    chnl_in  = Property(Array, depends_on=['bits', 'fs', 'rj', 'sj_mag', 'sj_freq'])
    chnl_out = Property(Array, depends_on=['chnl_in', 'a'])
    tie      = Property(Array, depends_on=['chnl_out'])
    jitter   = Property(Array, depends_on=['chnl_out'])
    jitter_spectrum = Property(Array, depends_on=['jitter'])
    t        = Property(Array, depends_on=['ui', 'npts', 'nspb'])
    t_ns     = Property(Array, depends_on=['t'])
    tbit_ps  = Property(Array, depends_on=['t', 'nspb'])

    # Handler set variables
    run_result  = Array # Not a Property, because it gets set by a Handler.
    adaptation  = Array
    ui_ests     = Array

    # Default initialization
    def __init__(self):
        super(DFEDemo, self).__init__()

        ui    = self.ui
        nbits = self.nbits
        nspb  = self.nspb
        eye_offset = self.eye_offset

        # This is just to trigger the eventual calculation of `self.tie_hist_*'.
        chnl_out = self.chnl_out
        tie      = self.tie


        plotdata = ArrayPlotData(t_ns            = self.t_ns,
                                 tbit_ps         = self.tbit_ps,
                                 chnl_in         = self.chnl_in,
                                 chnl_out        = self.chnl_out,
                                 clocks          = self.clocks,
                                 ui_ests         = self.ui_ests,
                                 jitter          = array(self.jitter) * 1.e12,
                                 jitter_spectrum = array(self.jitter_spectrum),
                                 f_MHz           = self.f_MHz,
                                 xing_times      = self.crossing_times_ideal_ns,
                                 tie_hist_bins   = array(self.tie_hist_bins) * 1.e12,
                                 tie_hist_counts = self.tie_hist_counts,
                                 dfe_out         = self.run_result,
                                )
        adaptation  = [zeros(self.n_taps) for i in range(100)]
        tap_weights = transpose(array(adaptation))
        for i in range(len(tap_weights)):
            plotdata.set_data("tap%d_weights" % (i + 1), tap_weights[i])
        plotdata.set_data("tap_weight_index", range(len(tap_weights[0])))
        for i in range(nbits - 2):
            plotdata.set_data("bit%d" % i, chnl_out[i * nspb + eye_offset : (i + 2) * nspb + eye_offset])
        self.plotdata = plotdata

        plot1 = Plot(plotdata)
        plot1.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot1.plot(("t_ns", "clocks"), type="line", color="green")
        plot1.title  = "Channel Output & Recovered Clocks"
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
        plot3.plot(("t_ns", "chnl_in"), type="line", color="blue")
        plot3.title  = "Channel Input"
        plot3.index_axis.title = "Time (ns)"
        plot3.tools.append(PanTool(plot3, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom3 = ZoomTool(plot3, tool_mode="range", axis='index', always_on=False)
        plot3.overlays.append(zoom3)

        plot4        = Plot(plotdata)
        plot4.plot(("xing_times", "jitter"), type="line", color="blue")
        plot4.title  = "Channel Output Jitter"
        plot4.index_axis.title = "Time (ns)"
        plot4.value_axis.title = "Jitter (ps)"
        plot4.value_range.high_setting = ui
        plot4.value_range.low_setting  = -ui

        plot5        = Plot(plotdata)
        plot5.plot(('tie_hist_bins', 'tie_hist_counts'), type="line", color="auto")
        plot5.title  = "Time Interval Error Distribution"
        plot5.index_axis.title = "Time (ps)"
        plot5.value_axis.title = "Count"
        zoom5 = ZoomTool(plot5, tool_mode="range", axis='index', always_on=False)
        plot5.overlays.append(zoom5)

        plot6        = Plot(plotdata)
        plot6.plot(('f_MHz', 'jitter_spectrum'), type="line", color="auto")
        plot6.title  = "Jitter Spectrum"
        plot6.index_axis.title = "Frequency (MHz)"
        plot6.value_axis.title = "|FFT(jitter)| (dBui)"
        #plot6.value_scale = 'log'
        zoom6 = ZoomTool(plot6, tool_mode="range", axis='index', always_on=False)
        plot6.overlays.append(zoom6)

        plot7 = Plot(plotdata)
        plot7.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot7.title  = "Channel Output"
        plot7.index_axis.title = "Time (ns)"
        plot7.tools.append(PanTool(plot7, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom7 = ZoomTool(plot7, tool_mode="range", axis='index', always_on=False)
        plot7.overlays.append(zoom7)

        plot8 = Plot(plotdata)
        plot8.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot8.title  = "DFE Output"
        plot8.index_axis.title = "Time (ns)"
        plot8.tools.append(PanTool(plot8, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom8 = ZoomTool(plot8, tool_mode="range", axis='index', always_on=False)
        plot8.overlays.append(zoom8)

        plot9 = Plot(plotdata)
        for i in range(len(tap_weights)):
            plot9.plot(("tap_weight_index", "tap%d_weights" % (i + 1)), type="line", color="auto", name="tap%d"%(i+1))
        plot9.title  = "DFE Tap Weight Adaptation"
        plot9.tools.append(PanTool(plot9, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom9 = ZoomTool(plot9, tool_mode="range", axis='index', always_on=False)
        plot9.overlays.append(zoom9)
        plot9.legend.visible = True
        plot9.legend.align = 'ul'

        plot10 = Plot(plotdata)
        for i in range(nbits - 2):
            plot10.plot(("tbit_ps", "bit%d" % i), type="line", color="blue")
        plot10.title  = "Eye Diagram"

        container_out = VPlotContainer(plot2, plot1)
        self.plot_out = container_out
        container_in  = GridPlotContainer(shape=(2,2))
        container_in.add(plot7)
        container_in.add(plot4)
        container_in.add(plot5)
        container_in.add(plot6)
        self.plot_in  = container_in
        container_dfe = VPlotContainer(plot8, plot9)
        self.plot_dfe = container_dfe
        container_eye  = VPlotContainer(plot10)
        self.plot_eye  = container_eye

    # Dependent variable definitions
    @cached_property
    def _get_bits(self):
        return [randint(2) for i in range(self.nbits)]
    
    @cached_property
    def _get_t(self):
        t0   = (self.ui * 1.e-12) / self.nspb
        npts = self.npts
        return [i * t0 for i in range(npts)]
    
    @cached_property
    def _get_t_ns(self):
        return 1.e9 * array(self.t)
    
    @cached_property
    def _get_tbit_ps(self):
        eye_offset = self.eye_offset
        return 1.e12 * array(self.t[eye_offset : 2 * self.nspb + eye_offset]) - self.ui
    
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
    def _get_chnl_in(self):
        bits    = self.bits
        nspb    = self.nspb
        fs      = self.fs
        rj      = self.rj * 1.e-12
        sj_mag  = self.sj_mag * 1.e-12
        sj_freq = self.sj_freq * 1.e6
        t       = self.t
        ui      = self.ui * 1.e-12

        ts      = 1. / fs

        res                          = repeat(2 * array(bits) - 1, nspb)
        self.crossing_times_ideal    = find_crossing_times(t, res)
        self.crossing_times_ideal_ns = array(self.crossing_times_ideal) * 1.e9

        jitter = [sj_mag * sin(2 * pi * sj_freq * i * ui) + normal(0., rj) for i in range(len(bits) - 1)]
        i = 1
        for jit in jitter:
            if(jit < -ui):
                jit = -ui
            if(jit > ui):
                jit = ui
            if(jit < 0.):
                res[i * nspb + int(jit / ts - 0.5) : i * nspb] = res[i * nspb]
            else:
                res[i * nspb : i * nspb + int(jit / ts + 0.5)] = res[i * nspb - 1]
            i += 1
        self.crossing_times_chnl_in = find_crossing_times(t, res)

        ## Introduce a 1/2 UI phase shift, half way through the sequence, to test CDR adaptation.
        ##res = res[:len(res)/2 - nspb/2] + res[len(res)/2:] + res[len(res)/2 - nspb/2 : len(res)/2]

        return res

    @cached_property
    def _get_chnl_out(self):
        chnl_in = self.chnl_in
        a       = self.a
        b       = self.b
        nbits   = self.nbits
        nspb    = self.nspb
        t       = self.t

        sig_len = nbits * nspb
        res     = lfilter(b, a, chnl_in)
        self.crossing_times_chnl_out = find_crossing_times(t, res, anlg=True)
        return res[:sig_len]

    @cached_property
    def _get_tie(self):
        ui       = self.ui * 1.e-12

        ties = diff(self.crossing_times_chnl_out) - ui

        def normalize(tie):
            while(tie > ui / 2.):
                tie -= ui
            return tie
            
        ties                 = map(normalize, ties)
        hist, bin_edges      = histogram(ties, 99, (-ui/4., ui/4.))
        bin_centers          = [mean([bin_edges[i], bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]
        self.tie_hist_counts = hist
        self.tie_hist_bins   = bin_centers

        return ties

    @cached_property
    def _get_jitter(self):
        ui           = self.ui * 1.e-12
        actual_xings = self.crossing_times_chnl_out
        ideal_xings  = self.crossing_times_ideal

        res = (actual_xings - ideal_xings[:len(actual_xings)]) % ui
        return res - mean(res)

    @cached_property
    def _get_jitter_spectrum(self):
        jitter      = self.jitter
        t_xings     = self.crossing_times_ideal   
        ui          = self.ui * 1.e-12
        run_lengths = map(int, array(t_xings[0] + diff(t_xings)) / ui)
        x           = [jit for run_length, jit in zip(run_lengths, jitter) for i in range(run_length)]
        f0          = 1. / t_xings[-1]
        self.f_MHz  = array([i * f0 for i in range(len(x) / 2)]) * 1.e-6
        res         = fft(x)
        res         = abs(res[:len(res) / 2]) / (len(x) * ui / 2)
        return 10. * log10(res)

    # Dynamic behavior definitions.
    def _t_ns_changed(self):
        self.plotdata.set_data("t_ns", self.t_ns)

    def _tbit_ps_changed(self):
        self.plotdata.set_data("tbit_ps", self.tbit_ps)

    def _chnl_in_changed(self):
        self.plotdata.set_data("chnl_in", self.chnl_in)
        self.plotdata.set_data("xing_times", self.crossing_times_ideal_ns)

    def _chnl_out_changed(self):
        chnl_out   = self.chnl_out
        nspb       = self.nspb
        eye_offset = self.eye_offset

        self.plotdata.set_data("chnl_out", chnl_out)
        for i in range(self.nbits - 2):
            self.plotdata.set_data("bit%d" % i, chnl_out[i * nspb + eye_offset : (i + 2) * nspb + eye_offset])

    def _clocks_changed(self):
        self.plotdata.set_data("clocks", self.clocks)

    def _ui_ests_changed(self):
        self.plotdata.set_data("ui_ests", self.ui_ests)

    def _tie_changed(self):
        self.plotdata.set_data("tie_hist_bins", array(self.tie_hist_bins) * 1.e12)
        self.plotdata.set_data("tie_hist_counts", self.tie_hist_counts)

    def _jitter_changed(self):
        self.plotdata.set_data("jitter", array(self.jitter) * 1.e12)

    def _jitter_spectrum_changed(self):
        self.plotdata.set_data("jitter_spectrum", self.jitter_spectrum)

    def _f_MHz_changed(self):
        self.plotdata.set_data("f_MHz", self.f_MHz)

    def _run_result_changed(self):
        run_result = self.run_result
        nspb       = self.nspb

        self.plotdata.set_data("dfe_out", run_result)
        j = 0
        for i in where(self.clocks)[0][1 : -1]:
            self.plotdata.set_data("bit%d" % j, run_result[i - nspb : i + nspb])
            j += 1

    def _adaptation_changed(self):
        tap_weights = transpose(array(self.adaptation))
        for i in range(len(tap_weights)):
            self.plotdata.set_data("tap%d_weights" % (i + 1), tap_weights[i])
        self.plotdata.set_data("tap_weight_index", range(len(tap_weights[0])))

    # Main window layout definition.
    traits_view = View(
            VGroup(
                HGroup(
                    VGroup(
                        Item(name='ui', label='Unit Interval (ps)', show_label=True, enabled_when='True'), #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                        Item(name='nbits',  label='# of bits',          show_label=True, enabled_when='True'),
                        Item(name='nspb',   label='samples per bit',    show_label=True, enabled_when='True'),
                        label='Simulation Control', show_border=True,
                    ),
                    VGroup(
                        Item(name='fc',     label='Channel fc (GHz)',   show_label=True, enabled_when='True'),
                        Item(name='rj',     label='Random Jitter (ps)', show_label=True, enabled_when='True'),
                        Item(name='sj_mag', label='Periodic Jitter Mag. (ps)', ),
                        Item(name='sj_freq', label='Periodic Jitter Freq. (MHz)', ),
                        label='Channel Definition', show_border=True,
                    ),
                    VGroup(
                        Item(name='gain',   label='DFE Gain'),
                        Item(name='n_taps', label='DFE Taps'),
                        Item(name='decision_scaler', label='Target Level'),
                        Item(name='n_ave', label='Nave.'),
                        label='DFE Parameters', show_border=True,
                    ),
                    VGroup(
                        Item(name='delta_t',label='CDR Delta-t (ps)',   show_label=True, enabled_when='True'),
                        Item(name='alpha',  label='CDR Alpha',          show_label=True, enabled_when='True'),
                        label='CDR Parameters', show_border=True,
                    ),
                ),
                Group(
                    Group(
                        Item('plot_in', editor=ComponentEditor(), show_label=False,),
                        label = 'Jitter'
                    ),
                    Group(
                        Item('plot_dfe', editor=ComponentEditor(), show_label=False,),
                        label = 'DFE'
                    ),
                    Group(
                        Item('plot_out', editor=ComponentEditor(), show_label=False,),
                        label = 'CDR'
                    ),
                    Group(
                        Item('plot_eye', editor=ComponentEditor(), show_label=False,),
                        label = 'Eye'
                    ),
                    Group(
                        Item('ident', style='readonly', show_label=False),
                        label = 'About'
                    ),
                    layout = 'tabbed', springy = True
                ),
            ),
        resizable = True,
        handler = MyHandler(),
        buttons = [run_dfe, "OK"],
        title='DFE Design Tool',
        width=1200, height=800
    )

def find_crossing_times(t, x, anlg=False):
    """
    Finds the zero crossing times of the input signal.

    Inputs:

      - t     Vector of sample times. Intervals do NOT need to be uniform.

      - x     Sampled input vector.

      - anlg  Interpolation flag. When TRUE, use linear interpolation,
              in order to determine zero crossing times more precisely.
    """

    crossing_indeces     = where(diff(sign(x)))[0] + 1
    if(anlg):
        crossing_times   = array([t[i - 1] + (t[i] - t[i - 1]) * x[i - 1] / (x[i - 1] - x[i])
                                   for i in crossing_indeces])
    else:
        crossing_times   = [t[i] for i in crossing_indeces]
    return crossing_times

if __name__ == '__main__':
    #viewer = CDRDemo()
    DFEDemo().configure_traits()

