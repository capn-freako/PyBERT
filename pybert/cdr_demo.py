#! /usr/bin/env python

"""
GUI demo of `CDR' class.

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

This Python script provides a stand alone demonstration of the CDR
class.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from traits.api    import \
    HasTraits, Array, Range, Int, Float, Enum, Property, String, List, cached_property, Instance
from traitsui.api import \
    View, Item, VSplit, Group, VGroup, HGroup, Label, Action, Handler, DefaultOverride
from chaco.api     import \
    Plot, ArrayPlotData, VPlotContainer, GridPlotContainer
from chaco.tools.api import \
    PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom, LineInspector, RangeSelection, RangeSelectionOverlay
from enable.component_editor import \
    ComponentEditor
from numpy        import arange, real, concatenate, angle, sign, sin, pi, \
                         array, float, zeros, ones, repeat, histogram, mean, where, diff, log10
from numpy.fft    import fft
from numpy.random import randint, random, normal
from scipy.signal import lfilter, firwin, iirdesign, iirfilter, freqz
import re

from cdr          import CDR

# Default model parameters - Modify these to customize the default simulation.
gNbits  = 1000    # number of bits to run
gNspb   = 100     # samples per bit
gDeltaT = 0.1     # proportional branch period correction (ps)
gAlpha  = 0.001   # relative significance of integral branch
gNLockAve = 500   # number of UI used to average CDR locked status.
gRelLockTol = .01 # relative lock tolerance of CDR.
gUI     = 100     # nominal unit interval (ps)
gFc     = 5.0     # default channel cut-off frequency (GHz)
gFc_min = 1.0     # min. channel cut-off frequency (GHz)
gFc_max = 50.0    # max. channel cut-off frequency (GHz)
gNtaps  = 2       # number of taps in IIR filter representing channel
gRj     = 0.1     # standard deviation of Gaussian random jitter (ps)
gSjMag  = 10.     # magnitude of periodic jitter (ps)
gSjFreq = 100.    # frequency of periodic jitter (MHz)

# Runs the CDR adaptation, when user clicks button.
class MyHandler(Handler):

    def do_run_cdr(self, info):
        chnl_out = info.object.chnl_out
        delta_t  = info.object.delta_t
        alpha    = info.object.alpha
        n_lock_ave    = info.object.n_lock_ave
        rel_lock_tol  = info.object.rel_lock_tol
        ui       = info.object.ui
        nspb     = info.object.nspb
        cdr      = CDR(delta_t, alpha, ui, n_lock_ave, rel_lock_tol)

        smpl_time           = ui / nspb
        t = next_bndry_time = 0.
        next_clk_time       = ui / 2.
        last_clk_smpl       = 1
        ui_est              = ui
        ui_ests             = []
        locked              = False
        lockeds             = []
        clocks              = zeros(len(chnl_out))
        clk_ind             = 0
        for smpl in chnl_out:
            if(t >= next_bndry_time):
                last_bndry_smpl  = sign(smpl)
                next_bndry_time += ui_est
            if(t >= next_clk_time):
                clocks[clk_ind] = 1
                (ui_est, locked) = cdr.adapt([last_clk_smpl, last_bndry_smpl, sign(smpl)])
                last_clk_smpl   = sign(smpl)
                next_bndry_time = next_clk_time + ui_est / 2.
                next_clk_time  += ui_est
            ui_ests.append(ui_est)
            lockeds.append(locked)
            t       += smpl_time
            clk_ind += 1

        info.object.clocks  = clocks
        info.object.ui_ests = ui_ests
        info.object.lockeds = lockeds

run_cdr = Action(name="RunCDR", action="do_run_cdr")
    
class CDRDemo(HasTraits):
    # Independent variables
    ui     = Float(gUI)        # (ps)
    nbits  = Int(gNbits)
    nspb   = Int(gNspb)
    delta_t = Float(gDeltaT)   # (ps)
    alpha  = Float(gAlpha)
    n_lock_ave = Int(gNLockAve)
    rel_lock_tol = Float(gRelLockTol)
    fc     = Float(gFc)        # (GHz)
    #fc     = Range(gFc_min, gFc_max, gFc_max)        # (GHz)
    rj     = Float(gRj)        # (ps)
    sj_mag = Float(gSjMag)     # (ps)
    sj_freq = Float(gSjFreq)   # (MHz)
    clocks = Array()
    plot_out = Instance(VPlotContainer)
    plot_in  = Instance(GridPlotContainer)
    #plot_in  = Instance(VPlotContainer)
    ident  = String('PyCDR v0.1 - a CDR design tool, written in Python\n\n \
    David Banas\n \
    June 19, 2014\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')


    # Dependent variables
    bits     = Property(Array, depends_on=['nbits'])
    npts     = Property(Array, depends_on=['nbits', 'nspb'])
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
    ui_ests  = Array # Not a Property, because it gets set by a Handler.
    lockeds  = Array # Not a Property, because it gets set by a Handler.

    # Default initialization
    def __init__(self):
        super(CDRDemo, self).__init__()

        ui = self.ui

        # This is just to trigger the eventual calculation of `self.tie_hist_*'.
        chnl_out = self.chnl_out
        tie      = self.tie

        plotdata = ArrayPlotData(t_ns            = self.t_ns,
                                 chnl_in         = self.chnl_in,
                                 chnl_out        = self.chnl_out,
                                 clocks          = self.clocks,
                                 ui_ests         = self.ui_ests,
                                 lockeds         = where(self.lockeds, 0.5 * ones(len(self.lockeds)), -0.5 * ones(len(self.lockeds))),
                                 jitter          = array(self.jitter) * 1.e12,
                                 jitter_spectrum = array(self.jitter_spectrum) / 1.e6,
                                 f_MHz           = self.f_MHz,
                                 xing_times      = self.crossing_times_ideal_ns,
                                 tie_hist_bins   = array(self.tie_hist_bins) * 1.e12,
                                 tie_hist_counts = self.tie_hist_counts,
                                )
        self.plotdata = plotdata

        plot1 = Plot(plotdata)
        plot1.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot1.plot(("t_ns", "clocks"), type="line", color="green")
        plot1.plot(("t_ns", "lockeds"), type="line", color="red")
        plot1.title  = "Channel Output, Recovered Clocks, & Locked"
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

        container_out = VPlotContainer(plot2, plot1)
        self.plot_out = container_out
        container_in  = GridPlotContainer(shape=(2,2))
        container_in.add(plot7)
        container_in.add(plot4)
        container_in.add(plot5)
        container_in.add(plot6)
        #container_in  = VPlotContainer(plot3, plot4, plot5)
        self.plot_in  = container_in

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
    def _get_npts(self):
        return self.nbits * self.nspb
    
    @cached_property
    def _get_fs(self):
        return self.nspb / (self.ui * 1.e-12)
    
    @cached_property
    def _get_a(self):
        fc     = self.fc * 1.e9 # User enters channel cutoff frequency in GHz.
        (b, a) = iirfilter(gNtaps - 1, fc/(self.fs/2), btype='lowpass')
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
        try:
            res = array(self.crossing_times_chnl_out - self.crossing_times_ideal)
        except:
            print "len(self.crossing_times_chnl_out):", len(self.crossing_times_chnl_out)
            raise

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

    def _chnl_in_changed(self):
        self.plotdata.set_data("chnl_in", self.chnl_in)
        self.plotdata.set_data("xing_times", self.crossing_times_ideal_ns)

    def _chnl_out_changed(self):
        self.plotdata.set_data("chnl_out", self.chnl_out)

    def _clocks_changed(self):
        self.plotdata.set_data("clocks", self.clocks)

    def _ui_ests_changed(self):
        self.plotdata.set_data("ui_ests", self.ui_ests)

    def _lockeds_changed(self):
        self.plotdata.set_data("lockeds", where(self.lockeds, 0.5 * ones(len(self.lockeds)), -0.5 * ones(len(self.lockeds))))

    def _tie_changed(self):
        self.plotdata.set_data("tie_hist_bins", array(self.tie_hist_bins) * 1.e12)
        self.plotdata.set_data("tie_hist_counts", self.tie_hist_counts)

    def _jitter_changed(self):
        self.plotdata.set_data("jitter", array(self.jitter) * 1.e12)

    def _jitter_spectrum_changed(self):
        self.plotdata.set_data("jitter_spectrum", self.jitter_spectrum)

    def _f_MHz_changed(self):
        self.plotdata.set_data("f_MHz", self.f_MHz)

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
                        Item(name='delta_t',label='CDR Delta-t (ps)',   show_label=True, enabled_when='True'),
                        Item(name='alpha',  label='CDR Alpha',          show_label=True, enabled_when='True'),
                        Item(name='n_lock_ave',  label='Nave.',          show_label=True, enabled_when='True'),
                        Item(name='rel_lock_tol',  label='Tol.',          show_label=True, enabled_when='True'),
                        label='CDR Parameters', show_border=True,
                    ),
                    VGroup(
                        Item(name='fc',     label='Channel fc (GHz)',   show_label=True, enabled_when='True'),
                        Item(name='rj',     label='Random Jitter (ps)', show_label=True, enabled_when='True'),
                        Item(name='sj_mag', label='Periodic Jitter Mag. (ps)', ),
                        Item(name='sj_freq', label='Periodic Jitter Freq. (MHz)', ),
                        label='Channel Definition', show_border=True,
                    ),
                    Item('ident',  style='readonly',                show_label=False),
                ),
                Group(
                    Group(
                        Item('plot_out',   editor=ComponentEditor(),        show_label=False,),
                        label = 'CDR Adaptation'
                    ),
                    Group(
                        Item('plot_in',   editor=ComponentEditor(),        show_label=False,),
                        label = 'Jitter'
                    ),
                    layout = 'tabbed', springy = True
                ),
            ),
        resizable = True,
        handler = MyHandler(),
        buttons = [run_cdr, "OK"],
        title='CDR Design Tool',
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
    viewer = CDRDemo()
    viewer.configure_traits()

