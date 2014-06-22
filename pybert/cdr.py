#! /usr/bin/env python

"""
Behavioral model of a "bang-bang" clock data recovery (CDR) unit.

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

This Python script provides a behavioral model of a "bang-bang" clock
data recovery (CDR) unit. The class defined, here, is intended for
integration into the larger `PyBERT' framework, but is also capable of
running in stand-alone mode for preliminary debugging.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""

from enthought.traits.api    import \
    HasTraits, Array, Range, Int, Float, Enum, Property, String, List, cached_property, Instance
from enthought.traits.ui.api import \
    View, Item, VSplit, Group, VGroup, HGroup, Label, Action, Handler, DefaultOverride
from enthought.chaco.api     import \
    Plot, ArrayPlotData, VPlotContainer
from enthought.chaco.tools.api import \
    PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom, LineInspector, RangeSelection, RangeSelectionOverlay
from enthought.enable.component_editor import \
    ComponentEditor
from numpy        import arange, real, concatenate, angle, sign, sin, pi, array, float, zeros, repeat
from numpy.fft    import ifft
from numpy.random import random
from scipy.signal import lfilter, firwin, iirdesign, iirfilter, freqz
import re

# Default model parameters - Modify these to customize the default simulation.
gNbits  = 1000    # number of bits to run
gNspb   = 100     # samples per bit
gDeltaT = 0.1     # proportional branch period correction (ps)
gAlpha  = 0.001   # relative significance of integral branch.
gUI     = 100     # nominal unit interval (ps)
gFc     = 5.0     # channel cut-off frequency (GHz)
gNtaps  = 2       # number of taps in IIR filter representing channel.

# Runs the CDR adaptation, when user clicks button.
class MyHandler(Handler):

    def do_run_cdr(self, info):
        chnl_out = info.object.chnl_out
        delta_t  = info.object.delta_t
        alpha    = info.object.alpha
        ui       = info.object.ui
        nspb     = info.object.nspb
        cdr      = CDR(delta_t, alpha, ui)

        smpl_time           = ui / nspb
        t = next_bndry_time = 0.
        next_clk_time       = ui / 2.
        last_clk_smpl       = 1
        ui_est              = ui
        ui_ests             = []
        for smpl in chnl_out:
            if(t >= next_bndry_time):
                last_bndry_smpl  = sign(smpl)
                next_bndry_time += ui_est
            if(t >= next_clk_time):
                ui_est          = cdr.adapt([last_clk_smpl, last_bndry_smpl, sign(smpl)])
                last_clk_smpl   = sign(smpl)
                next_bndry_time = next_clk_time + ui_est / 2.
                next_clk_time  += ui_est
                ui_ests.append(ui_est)
            t += smpl_time

        clocks        = zeros(len(chnl_out))
        next_clk_time = ui / 2.
        for ui_est in ui_ests:
            clocks[int(next_clk_time / smpl_time + 0.5)] = 1
            next_clk_time += ui_est

        info.object.clocks  = clocks
        info.object.ui_ests = repeat(array(ui_ests), nspb)

run_cdr = Action(name="RunCDR", action="do_run_cdr")
    
# Model Proper - Don't change anything below this line.
class CDR(object):
    """A class providing behavioral modeling of a 'bang- bang' clock
       data recovery (CDR) unit."""

    def __init__(self, delta_t, alpha, ui):
        #super(object, self).__init__()
        self.delta_t = delta_t
        self.alpha   = alpha
        self.nom_ui  = ui
        self.ui      = ui
        self.integral_correction = 0.0

    def adapt(self, samples):
        """Adapt period/phase, according to 3 samples."""

        if(samples[0] == samples[2]):   # No transition; no correction.
            proportional_correction = 0.0
        elif(samples[0] == samples[1]): # Early clock; increase period.
            proportional_correction = self.delta_t
        else:                           # Late clock; decrease period.
            proportional_correction = -self.delta_t
        self.integral_correction += self.alpha * proportional_correction
        self.ui = self.nom_ui + self.integral_correction + proportional_correction
        return self.ui

class CDRDemo(HasTraits):
    # Independent variables
    ui     = Float(gUI)
    nbits  = Int(gNbits)
    nspb   = Int(gNspb)
    delta_t = Float(gDeltaT)
    alpha  = Float(gAlpha)
    fc     = Float(gFc)
    clocks = Array()
    plot   = Instance(VPlotContainer)
    ident  = String('PyCDR v0.1 - a CDR design tool, written in Python\n\n \
    David Banas\n \
    June 19, 2014\n\n \
    Copyright (c) 2014 David Banas; All rights reserved World wide.')


    # Dependent variables
    bits     = Property(Array, depends_on=['nbits'])
    npts     = Property(Array, depends_on=['nbits', 'nspb'])
    fs       = Property(Array, depends_on=['ui', 'nspb'])
    a        = Property(Array, depends_on=['fc', 'fs'])
    b        = array([1.]) # Will be set by 'a' handler, upon change in dependencies.
    h        = Property(Array, depends_on=['npts', 'a'])
    chnl_in  = Property(Array, depends_on=['bits', 'nspb'])
    chnl_out = Property(Array, depends_on=['chnl_in', 'a'])
    t        = Property(Array, depends_on=['ui', 'npts', 'nspb'])
    t_ns     = Property(Array, depends_on=['t'])
    ui_ests  = Array # Not a Property, because it gets set by a Handler.

    # Default initialization
    def __init__(self):
        super(CDRDemo, self).__init__()
        plotdata = ArrayPlotData(t_ns=self.t_ns, chnl_out=self.chnl_out, clocks=self.clocks, ui_ests=self.ui_ests)
        plot1 = Plot(plotdata)
        plot1.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot1.plot(("t_ns", "clocks"), type="line", color="green")
        plot1.title = "Channel Output"
        plot1.tools.append(PanTool(plot1))
        # The ZoomTool tool is stateful and allows drawing a zoom
        # box to select a zoom region.
        zoom1 = ZoomTool(plot1, tool_mode="box", always_on=False)
        plot1.overlays.append(zoom1)
        # The DragZoom tool just zooms in and out as the user drags
        # the mouse vertically.
        dragzoom1 = DragZoom(plot1, drag_button="right")
        plot1.tools.append(dragzoom1)
        plot1.active_tool = RangeSelection(plot1, left_button_selects = True)
        plot1.overlays.append(RangeSelectionOverlay(component=plot1))

        plot2 = Plot(plotdata)
        plot2.plot(("t_ns", "ui_ests"), type="line", color="blue")
        plot2.title = "CDR Adaptation"
        container = VPlotContainer(plot2, plot1)
        self.plot  = container
        self.plotdata = plotdata

    # Dependent variable definitions
    @cached_property
    def _get_bits(self):
        return [int(2 * random()) for i in range(self.nbits)]
    
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
        bits = self.bits
        nspb = self.nspb
        res = repeat(2 * array(bits) - 1, nspb)
        # Introduce a 1/2 UI phase shift, half way through the sequence, to test CDR adaptation.
        res = res.tolist() # Makes what we're about to do easier.
        res = res[:len(res)/2 - nspb/2] + res[len(res)/2:] + res[len(res)/2 - nspb/2 : len(res)/2]
        return res

    @cached_property
    def _get_chnl_out(self):
        chnl_in = self.chnl_in
        a       = self.a
        b       = self.b
        return lfilter(b, a, chnl_in)

    # Dynamic behavior definitions.
    def _t_ns_changed(self):
        self.plotdata.set_data("t_ns", self.t_ns)

    def _chnl_out_changed(self):
        self.plotdata.set_data("chnl_out", self.chnl_out)

    def _clocks_changed(self):
        self.plotdata.set_data("clocks", self.clocks)

    def _ui_ests_changed(self):
        self.plotdata.set_data("ui_ests", self.ui_ests)

    # Main window layout definition.
    traits_view = View(
        HGroup(
            VGroup(
                Item(name='ui', label='Unit Interval (ps)', show_label=True, enabled_when='True'), #editor=DefaultOverride(mode='spinner'), width=0.5, style='readonly', format_str="%+06.3f"
                Item(name='nbits',  label='# of bits',          show_label=True, enabled_when='True'),
                Item(name='nspb',   label='samples per bit',    show_label=True, enabled_when='True'),
                Item(name='delta_t',label='CDR Delta-t (ps)',   show_label=True, enabled_when='True'),
                Item(name='alpha',  label='CDR Alpha',          show_label=True, enabled_when='True'),
                Item(name='fc',     label='Channel fc (GHz)',   show_label=True, enabled_when='True'),
                Item('plot',  editor=ComponentEditor(),        show_label=False,),
                Item('ident',  style='readonly',                show_label=False),
            ),
        ),
        resizable = True,
        handler = MyHandler(),
        buttons = [run_cdr, "OK"],
        title='CDR Design Tool',
        width=1000, height=800
    )

if __name__ == '__main__':
    viewer = CDRDemo()
    viewer.configure_traits()

