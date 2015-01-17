#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>
Original Date:   17 June 2014

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

from traits.api      import HasTraits, Array, Range, Float, Int, Property, String, cached_property, Instance, HTML, List, Bool
from chaco.api       import Plot, ArrayPlotData, VPlotContainer, GridPlotContainer, ColorMapper, Legend, OverlayPlotContainer, PlotAxis
from chaco.tools.api import PanTool, ZoomTool, LegendTool, TraitsTool, DragZoom
from numpy           import array, linspace, zeros, histogram, mean, diff, log10, transpose, shape
from numpy.fft       import fft
from numpy.random    import randint
from scipy.signal    import lfilter, iirfilter

from pybert_view  import *
from pybert_cntrl import *
from pybert_util  import *

debug = False


# Default model parameters - Modify these to customize the default simulation.
# - Simulation Control
gUI             = 100     # (ps)
gNbits          = 8000    # number of bits to run
gPatLen         = 127     # repeating bit pattern length
gNspb           = 32      # samples per bit
gFftConv        = False   # True = Use frequency domain signal processing.
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
gThresh         = 4       # threshold for identifying periodic jitter spectral elements (sigma)

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
    fft_conv        = Bool(gFftConv)
    # - Channel Control
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
    pretap          = Float(0.0)
    posttap         = Float(0.0)
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
#    channel_perf    = Float(1.)
#    cdr_perf        = Float(1.)
#    dfe_perf        = Float(1.)
    jitter_perf     = Float(0.)
    total_perf      = Float(0.)
    # - About
    ident  = String('PyBERT v0.4 - a serial communication link design tool, written in Python\n\n \
    David Banas\n \
    December 25, 2014\n\n \
    Copyright (c) 2014 David Banas;\n \
    All rights reserved World wide.')

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

    # Default initialization
    def __init__(self):
        super(PyBERT, self).__init__()

        plotdata = self.plotdata

        # Running the simulation will fill in the 'plotdata' structure.
        my_run_simulation(self, initial_run=True)

        # Now, create all the various plots we need for our GUI.
        # - DFE tab
        plot1 = Plot(plotdata)
        plot1.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot1.plot(("t_ns", "clocks"), type="line", color="green")
        plot1.plot(("t_ns", "lockeds"), type="line", color="red")
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
        plot3.plot(('f_GHz', 'jitter_rejection_ratio'), type="line", color="blue")
        plot3.title  = "CDR Jitter Rejection Ratio"
        plot3.index_axis.title = "Frequency (GHz)"
        plot3.value_axis.title = "Ratio (dB)"
        zoom3 = ZoomTool(plot3, tool_mode="range", axis='index', always_on=False)
        plot3.overlays.append(zoom3)

        plot9 = Plot(plotdata, auto_colors=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
        for i in range(gNtaps):
            plot9.plot(("tap_weight_index", "tap%d_weights" % (i + 1)), type="line", color="auto", name="tap%d"%(i+1))
        plot9.title  = "DFE Adaptation"
        plot9.tools.append(PanTool(plot9, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom9 = ZoomTool(plot9, tool_mode="range", axis='index', always_on=False)
        plot9.overlays.append(zoom9)
        plot9.legend.visible = True
        plot9.legend.align = 'ul'

        container_dfe = GridPlotContainer(shape=(2,2))
        container_dfe.add(plot2)
        container_dfe.add(plot9)
        container_dfe.add(plot1)
        container_dfe.add(plot3)
        self.plots_dfe = container_dfe

        # - Impulse Responses tab
        plot_h_chnl = Plot(plotdata)
        plot_h_chnl.plot(("t_ns_chnl", "chnl_h"), type="line", color="blue")
        plot_h_chnl.title            = "Channel"
        plot_h_chnl.index_axis.title = "Time (ns)"
        plot_h_chnl.y_axis.title     = "Impulse Response (V/ns)"

        plot_h_tx = Plot(plotdata)
#        plot_h_tx.plot(("t_ns_chnl", "tx_h"),     type="line", color="blue", name="Incremental")
        plot_h_tx.plot(("t_ns_chnl", "tx_out_h"), type="line", color="red",  name="Cumulative")
        plot_h_tx.title            = "Channel + Tx Preemphasis"
        plot_h_tx.index_axis.title = "Time (ns)"
        plot_h_tx.y_axis.title     = "Impulse Response (V/ns)"
#        plot_h_tx.legend.visible   = True
#        plot_h_tx.legend.align     = 'ur'

        plot_h_ctle = Plot(plotdata)
#        plot_h_ctle.plot(("t_ns_chnl", "ctle_h"),     type="line", color="blue", name="Incremental")
        plot_h_ctle.plot(("t_ns_chnl", "ctle_out_h"), type="line", color="red",  name="Cumulative")
        plot_h_ctle.title            = "Channel + Tx Preemphasis + CTLE"
        plot_h_ctle.index_axis.title = "Time (ns)"
        plot_h_ctle.y_axis.title     = "Impulse Response (V/ns)"
#        plot_h_ctle.legend.visible   = True
#        plot_h_ctle.legend.align     = 'ur'

        plot_h_dfe = Plot(plotdata)
#        plot_h_dfe.plot(("t_ns_chnl", "dfe_h"),     type="line", color="blue", name="Incremental")
        plot_h_dfe.plot(("t_ns_chnl", "dfe_out_h"), type="line", color="red",  name="Cumulative")
        plot_h_dfe.title            = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_h_dfe.index_axis.title = "Time (ns)"
        plot_h_dfe.y_axis.title     = "Impulse Response (V/ns)"
#        plot_h_dfe.legend.visible   = True
#        plot_h_dfe.legend.align     = 'ur'

        container_h = GridPlotContainer(shape=(2,2))
        container_h.add(plot_h_chnl)
        container_h.add(plot_h_tx)
        container_h.add(plot_h_ctle)
        container_h.add(plot_h_dfe)
        self.plots_h  = container_h

        # - Step Responses tab
        plot_s_chnl = Plot(plotdata)
        plot_s_chnl.plot(("t_ns_chnl", "chnl_s"), type="line", color="blue")
        plot_s_chnl.title            = "Channel"
        plot_s_chnl.index_axis.title = "Time (ns)"
        plot_s_chnl.y_axis.title     = "Step Response (V)"

        plot_s_tx = Plot(plotdata)
        plot_s_tx.plot(("t_ns_chnl", "tx_s"),     type="line", color="blue", name="Incremental")
        plot_s_tx.plot(("t_ns_chnl", "tx_out_s"), type="line", color="red",  name="Cumulative")
        plot_s_tx.title            = "Channel + Tx Preemphasis"
        plot_s_tx.index_axis.title = "Time (ns)"
        plot_s_tx.y_axis.title     = "Step Response (V)"
        plot_s_tx.legend.visible   = True
        plot_s_tx.legend.align     = 'lr'

        plot_s_ctle = Plot(plotdata)
        plot_s_ctle.plot(("t_ns_chnl", "ctle_s"),     type="line", color="blue", name="Incremental")
        plot_s_ctle.plot(("t_ns_chnl", "ctle_out_s"), type="line", color="red",  name="Cumulative")
        plot_s_ctle.title            = "Channel + Tx Preemphasis + CTLE"
        plot_s_ctle.index_axis.title = "Time (ns)"
        plot_s_ctle.y_axis.title     = "Step Response (V)"
        plot_s_ctle.legend.visible   = True
        plot_s_ctle.legend.align     = 'lr'

        plot_s_dfe = Plot(plotdata)
        plot_s_dfe.plot(("t_ns_chnl", "dfe_s"),     type="line", color="blue", name="Incremental")
        plot_s_dfe.plot(("t_ns_chnl", "dfe_out_s"), type="line", color="red",  name="Cumulative")
        plot_s_dfe.title            = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_s_dfe.index_axis.title = "Time (ns)"
        plot_s_dfe.y_axis.title     = "Step Response (V)"
        plot_s_dfe.legend.visible   = True
        plot_s_dfe.legend.align     = 'lr'

        container_s = GridPlotContainer(shape=(2,2))
        container_s.add(plot_s_chnl)
        container_s.add(plot_s_tx)
        container_s.add(plot_s_ctle)
        container_s.add(plot_s_dfe)
        self.plots_s  = container_s

        # - Frequency Responses tab
        plot_H_chnl = Plot(plotdata)
        plot_H_chnl.plot(("f_GHz", "chnl_H"), type="line", color="blue", index_scale='log')
        plot_H_chnl.title            = "Channel"
        plot_H_chnl.index_axis.title = "Frequency (GHz)"
        plot_H_chnl.y_axis.title     = "Frequency Response (dB)"
        plot_H_chnl.index_range.low_setting  = 0.1
        plot_H_chnl.index_range.high_setting = 40.
        plot_H_chnl.value_range.low_setting  = -40.

        plot_H_tx = Plot(plotdata)
        plot_H_tx.plot(("f_GHz", "tx_H"),     type="line", color="blue", name="Incremental", index_scale='log')
        plot_H_tx.plot(("f_GHz", "tx_out_H"), type="line", color="red",  name="Cumulative", index_scale='log')
        plot_H_tx.title            = "Channel + Tx Preemphasis"
        plot_H_tx.index_axis.title = "Frequency (GHz)"
        plot_H_tx.y_axis.title     = "Frequency Response (dB)"
        plot_H_tx.index_range.low_setting  = 0.1
        plot_H_tx.index_range.high_setting = 40.
        plot_H_tx.value_range.low_setting  = -40.
        plot_H_tx.legend.visible   = True
        plot_H_tx.legend.align     = 'll'

        plot_H_ctle = Plot(plotdata)
        plot_H_ctle.plot(("f_GHz", "ctle_H"),     type="line", color="blue", name="Incremental", index_scale='log')
        plot_H_ctle.plot(("f_GHz", "ctle_out_H"), type="line", color="red",  name="Cumulative", index_scale='log')
        plot_H_ctle.title            = "Channel + Tx Preemphasis + CTLE"
        plot_H_ctle.index_axis.title = "Frequency (GHz)"
        plot_H_ctle.y_axis.title     = "Frequency Response (dB)"
        plot_H_ctle.index_range.low_setting  = 0.1
        plot_H_ctle.index_range.high_setting = 40.
        plot_H_ctle.value_range.low_setting  = -40.
        plot_H_ctle.legend.visible   = True
        plot_H_ctle.legend.align     = 'll'

        plot_H_dfe = Plot(plotdata)
        plot_H_dfe.plot(("f_GHz", "dfe_H"),     type="line", color="blue", name="Incremental", index_scale='log')
        plot_H_dfe.plot(("f_GHz", "dfe_out_H"), type="line", color="red",  name="Cumulative", index_scale='log')
        plot_H_dfe.title            = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_H_dfe.index_axis.title = "Frequency (GHz)"
        plot_H_dfe.y_axis.title     = "Frequency Response (dB)"
        plot_H_dfe.index_range.low_setting  = 0.1
        plot_H_dfe.index_range.high_setting = 40.
        plot_H_dfe.value_range.low_setting  = -40.
        plot_H_dfe.legend.visible   = True
        plot_H_dfe.legend.align     = 'll'

        container_H = GridPlotContainer(shape=(2,2))
        container_H.add(plot_H_chnl)
        container_H.add(plot_H_tx)
        container_H.add(plot_H_ctle)
        container_H.add(plot_H_dfe)
        self.plots_H  = container_H

        # - Outputs tab
        plot_out_chnl = Plot(plotdata)
        plot_out_chnl.plot(("t_ns", "chnl_out"), type="line", color="blue")
        plot_out_chnl.title            = "Channel"
        plot_out_chnl.index_axis.title = "Time (ns)"
        plot_out_chnl.y_axis.title     = "Output (V)"
        zoom_out_chnl = ZoomTool(plot_out_chnl, tool_mode="range", axis='index', always_on=False)
        plot_out_chnl.overlays.append(zoom_out_chnl)

        plot_out_tx = Plot(plotdata)
        plot_out_tx.plot(("t_ns", "tx_out"), type="line", color="blue")
        plot_out_tx.title            = "Channel + Tx Preemphasis (Noise added here.)"
        plot_out_tx.index_axis.title = "Time (ns)"
        plot_out_tx.y_axis.title     = "Output (V)"
        plot_out_tx.index_range = plot_out_chnl.index_range # Zoom x-axes in tandem.

        plot_out_ctle = Plot(plotdata)
        plot_out_ctle.plot(("t_ns", "ctle_out"), type="line", color="blue")
        plot_out_ctle.title            = "Channel + Tx Preemphasis + CTLE"
        plot_out_ctle.index_axis.title = "Time (ns)"
        plot_out_ctle.y_axis.title     = "Output (V)"
        plot_out_ctle.index_range = plot_out_chnl.index_range # Zoom x-axes in tandem.

        plot_out_dfe = Plot(plotdata)
        plot_out_dfe.plot(("t_ns", "dfe_out"), type="line", color="blue")
        plot_out_dfe.title            = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_out_dfe.index_axis.title = "Time (ns)"
        plot_out_dfe.y_axis.title     = "Output (V)"
        plot_out_dfe.index_range = plot_out_chnl.index_range # Zoom x-axes in tandem.

        container_out = GridPlotContainer(shape=(2,2))
        container_out.add(plot_out_chnl)
        container_out.add(plot_out_tx)
        container_out.add(plot_out_ctle)
        container_out.add(plot_out_dfe)
        self.plots_out  = container_out

        # - Eye Diagrams tab
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

        plot_eye_chnl = Plot(plotdata)
        plot_eye_chnl.img_plot("eye_chnl", colormap=clr_map,)
        plot_eye_chnl.y_direction = 'normal'
        plot_eye_chnl.components[0].y_direction = 'normal'
        plot_eye_chnl.title  = "Channel"
        plot_eye_chnl.x_axis.title = "Time (ps)"
        plot_eye_chnl.x_axis.orientation = "bottom"
        plot_eye_chnl.y_axis.title = "Signal Level (V)"
        plot_eye_chnl.x_grid.visible = True
        plot_eye_chnl.y_grid.visible = True
        plot_eye_chnl.x_grid.line_color = 'gray'
        plot_eye_chnl.y_grid.line_color = 'gray'

        plot_eye_tx = Plot(plotdata)
        plot_eye_tx.img_plot("eye_tx", colormap=clr_map,)
        plot_eye_tx.y_direction = 'normal'
        plot_eye_tx.components[0].y_direction = 'normal'
        plot_eye_tx.title  = "Channel + Tx Preemphasis (Noise added here.)"
        plot_eye_tx.x_axis.title = "Time (ps)"
        plot_eye_tx.x_axis.orientation = "bottom"
        plot_eye_tx.y_axis.title = "Signal Level (V)"
        plot_eye_tx.x_grid.visible = True
        plot_eye_tx.y_grid.visible = True
        plot_eye_tx.x_grid.line_color = 'gray'
        plot_eye_tx.y_grid.line_color = 'gray'

        plot_eye_ctle = Plot(plotdata)
        plot_eye_ctle.img_plot("eye_ctle", colormap=clr_map,)
        plot_eye_ctle.y_direction = 'normal'
        plot_eye_ctle.components[0].y_direction = 'normal'
        plot_eye_ctle.title  = "Channel + Tx Preemphasis + CTLE"
        plot_eye_ctle.x_axis.title = "Time (ps)"
        plot_eye_ctle.x_axis.orientation = "bottom"
        plot_eye_ctle.y_axis.title = "Signal Level (V)"
        plot_eye_ctle.x_grid.visible = True
        plot_eye_ctle.y_grid.visible = True
        plot_eye_ctle.x_grid.line_color = 'gray'
        plot_eye_ctle.y_grid.line_color = 'gray'

        plot_eye_dfe = Plot(plotdata)
        plot_eye_dfe.img_plot("eye_dfe", colormap=clr_map,)
        plot_eye_dfe.y_direction = 'normal'
        plot_eye_dfe.components[0].y_direction = 'normal'
        plot_eye_dfe.title  = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_eye_dfe.x_axis.title = "Time (ps)"
        plot_eye_dfe.x_axis.orientation = "bottom"
        plot_eye_dfe.y_axis.title = "Signal Level (V)"
        plot_eye_dfe.x_grid.visible = True
        plot_eye_dfe.y_grid.visible = True
        plot_eye_dfe.x_grid.line_color = 'gray'
        plot_eye_dfe.y_grid.line_color = 'gray'

        container_eye = GridPlotContainer(shape=(2,2))
        container_eye.add(plot_eye_chnl)
        container_eye.add(plot_eye_tx)
        container_eye.add(plot_eye_ctle)
        container_eye.add(plot_eye_dfe)
        self.plots_eye  = container_eye

        # - Jitter Distributions tab
        plot_jitter_dist_chnl        = Plot(plotdata)
        plot_jitter_dist_chnl.plot(('jitter_bins', 'jitter_chnl'),     type="line", color="blue", name="Measured")
        plot_jitter_dist_chnl.plot(('jitter_bins', 'jitter_ext_chnl'), type="line", color="red",  name="Extrapolated")
        plot_jitter_dist_chnl.title  = "Channel"
        plot_jitter_dist_chnl.index_axis.title = "Time (ps)"
        plot_jitter_dist_chnl.value_axis.title = "Count"
        plot_jitter_dist_chnl.legend.visible   = True
        plot_jitter_dist_chnl.legend.align     = 'ur'

        plot_jitter_dist_tx        = Plot(plotdata)
        plot_jitter_dist_tx.plot(('jitter_bins', 'jitter_tx'),     type="line", color="blue", name="Measured")
        plot_jitter_dist_tx.plot(('jitter_bins', 'jitter_ext_tx'), type="line", color="red",  name="Extrapolated")
        plot_jitter_dist_tx.title  = "Channel + Tx Preemphasis (Noise added here.)"
        plot_jitter_dist_tx.index_axis.title = "Time (ps)"
        plot_jitter_dist_tx.value_axis.title = "Count"
        plot_jitter_dist_tx.legend.visible   = True
        plot_jitter_dist_tx.legend.align     = 'ur'

        plot_jitter_dist_ctle        = Plot(plotdata)
        plot_jitter_dist_ctle.plot(('jitter_bins', 'jitter_ctle'),     type="line", color="blue", name="Measured")
        plot_jitter_dist_ctle.plot(('jitter_bins', 'jitter_ext_ctle'), type="line", color="red",  name="Extrapolated")
        plot_jitter_dist_ctle.title  = "Channel + Tx Preemphasis + CTLE"
        plot_jitter_dist_ctle.index_axis.title = "Time (ps)"
        plot_jitter_dist_ctle.value_axis.title = "Count"
        plot_jitter_dist_ctle.legend.visible   = True
        plot_jitter_dist_ctle.legend.align     = 'ur'

        plot_jitter_dist_dfe        = Plot(plotdata)
        plot_jitter_dist_dfe.plot(('jitter_bins', 'jitter_dfe'),     type="line", color="blue", name="Measured")
        plot_jitter_dist_dfe.plot(('jitter_bins', 'jitter_ext_dfe'), type="line", color="red",  name="Extrapolated")
        plot_jitter_dist_dfe.title  = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_jitter_dist_dfe.index_axis.title = "Time (ps)"
        plot_jitter_dist_dfe.value_axis.title = "Count"
        plot_jitter_dist_dfe.legend.visible   = True
        plot_jitter_dist_dfe.legend.align     = 'ur'

        container_jitter_dist = GridPlotContainer(shape=(2,2))
        container_jitter_dist.add(plot_jitter_dist_chnl)
        container_jitter_dist.add(plot_jitter_dist_tx)
        container_jitter_dist.add(plot_jitter_dist_ctle)
        container_jitter_dist.add(plot_jitter_dist_dfe)
        self.plots_jitter_dist  = container_jitter_dist

        # - Jitter Spectrums tab
        plot_jitter_spec_chnl        = Plot(plotdata)
        plot_jitter_spec_chnl.plot(('f_MHz', 'jitter_spectrum_chnl'),     type="line", color="blue",    name="Total")
        plot_jitter_spec_chnl.plot(('f_MHz', 'jitter_ind_spectrum_chnl'), type="line", color="red",     name="Data Independent")
        plot_jitter_spec_chnl.plot(('f_MHz', 'thresh_chnl'),              type="line", color="magenta", name="Pj Threshold")
        plot_jitter_spec_chnl.title  = "Channel"
        plot_jitter_spec_chnl.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_chnl.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_chnl.tools.append(PanTool(plot_jitter_spec_chnl, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom_jitter_spec_chnl = ZoomTool(plot_jitter_spec_chnl, tool_mode="range", axis='index', always_on=False)
        plot_jitter_spec_chnl.overlays.append(zoom_jitter_spec_chnl)
        plot_jitter_spec_chnl.legend.visible = True
        plot_jitter_spec_chnl.legend.align = 'lr'

        plot_jitter_spec_tx        = Plot(plotdata)
        plot_jitter_spec_tx.plot(('f_MHz', 'jitter_spectrum_tx'),     type="line", color="blue",    name="Total")
        plot_jitter_spec_tx.plot(('f_MHz', 'jitter_ind_spectrum_tx'), type="line", color="red",     name="Data Independent")
        plot_jitter_spec_tx.plot(('f_MHz', 'thresh_tx'),              type="line", color="magenta", name="Pj Threshold")
        plot_jitter_spec_tx.title  = "Channel + Tx Preemphasis (Noise added here.)"
        plot_jitter_spec_tx.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_tx.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_tx.value_range.low_setting  = -40.
        plot_jitter_spec_tx.tools.append(PanTool(plot_jitter_spec_tx, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom_jitter_spec_tx = ZoomTool(plot_jitter_spec_tx, tool_mode="range", axis='index', always_on=False)
        plot_jitter_spec_tx.overlays.append(zoom_jitter_spec_tx)
        plot_jitter_spec_tx.legend.visible = True
        plot_jitter_spec_tx.legend.align = 'lr'

        plot_jitter_spec_chnl.value_range = plot_jitter_spec_tx.value_range 

        plot_jitter_spec_ctle        = Plot(plotdata)
        plot_jitter_spec_ctle.plot(('f_MHz', 'jitter_spectrum_ctle'),     type="line", color="blue",    name="Total")
        plot_jitter_spec_ctle.plot(('f_MHz', 'jitter_ind_spectrum_ctle'), type="line", color="red",     name="Data Independent")
        plot_jitter_spec_ctle.plot(('f_MHz', 'thresh_ctle'),              type="line", color="magenta", name="Pj Threshold")
        plot_jitter_spec_ctle.title  = "Channel + Tx Preemphasis + CTLE"
        plot_jitter_spec_ctle.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_ctle.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_ctle.tools.append(PanTool(plot_jitter_spec_ctle, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom_jitter_spec_ctle = ZoomTool(plot_jitter_spec_ctle, tool_mode="range", axis='index', always_on=False)
        plot_jitter_spec_ctle.overlays.append(zoom_jitter_spec_ctle)
        plot_jitter_spec_ctle.legend.visible = True
        plot_jitter_spec_ctle.legend.align = 'lr'
        plot_jitter_spec_ctle.value_range = plot_jitter_spec_tx.value_range 

        plot_jitter_spec_dfe        = Plot(plotdata)
        plot_jitter_spec_dfe.plot(('f_MHz_dfe', 'jitter_spectrum_dfe'),     type="line", color="blue",    name="Total")
        plot_jitter_spec_dfe.plot(('f_MHz_dfe', 'jitter_ind_spectrum_dfe'), type="line", color="red",     name="Data Independent")
        plot_jitter_spec_dfe.plot(('f_MHz_dfe', 'thresh_dfe'),              type="line", color="magenta", name="Pj Threshold")
        plot_jitter_spec_dfe.title  = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_jitter_spec_dfe.index_axis.title = "Frequency (MHz)"
        plot_jitter_spec_dfe.value_axis.title = "|FFT(TIE)| (dBui)"
        plot_jitter_spec_dfe.tools.append(PanTool(plot_jitter_spec_dfe, constrain=True, constrain_key=None, constrain_direction='x'))
        zoom_jitter_spec_dfe = ZoomTool(plot_jitter_spec_dfe, tool_mode="range", axis='index', always_on=False)
        plot_jitter_spec_dfe.overlays.append(zoom_jitter_spec_dfe)
        plot_jitter_spec_dfe.legend.visible = True
        plot_jitter_spec_dfe.legend.align = 'lr'
        plot_jitter_spec_dfe.value_range = plot_jitter_spec_tx.value_range 

        container_jitter_spec = GridPlotContainer(shape=(2,2))
        container_jitter_spec.add(plot_jitter_spec_chnl)
        container_jitter_spec.add(plot_jitter_spec_tx)
        container_jitter_spec.add(plot_jitter_spec_ctle)
        container_jitter_spec.add(plot_jitter_spec_dfe)
        self.plots_jitter_spec  = container_jitter_spec

        # - Bathtub Curves tab
        plot_bathtub_chnl = Plot(plotdata)
        plot_bathtub_chnl.plot(("jitter_bins", "bathtub_chnl"), type="line", color="blue")
        plot_bathtub_chnl.value_range.high_setting =   0
        plot_bathtub_chnl.value_range.low_setting  = -18
        plot_bathtub_chnl.value_axis.tick_interval =   3
        plot_bathtub_chnl.title             = "Channel"
        plot_bathtub_chnl.index_axis.title  = "Time (ps)"
        plot_bathtub_chnl.value_axis.title  = "Log10(P(Transition occurs inside.))"

        plot_bathtub_tx = Plot(plotdata)
        plot_bathtub_tx.plot(("jitter_bins", "bathtub_tx"), type="line", color="blue")
        plot_bathtub_tx.value_range.high_setting =   0
        plot_bathtub_tx.value_range.low_setting  = -18
        plot_bathtub_tx.value_axis.tick_interval =   3
        plot_bathtub_tx.title             = "Channel + Tx Preemphasis"
        plot_bathtub_tx.index_axis.title  = "Time (ps)"
        plot_bathtub_tx.value_axis.title  = "Log10(P(Transition occurs inside.))"

        plot_bathtub_ctle = Plot(plotdata)
        plot_bathtub_ctle.plot(("jitter_bins", "bathtub_ctle"), type="line", color="blue")
        plot_bathtub_ctle.value_range.high_setting =   0
        plot_bathtub_ctle.value_range.low_setting  = -18
        plot_bathtub_ctle.value_axis.tick_interval =   3
        plot_bathtub_ctle.title             = "Channel + Tx Preemphasis + CTLE"
        plot_bathtub_ctle.index_axis.title  = "Time (ps)"
        plot_bathtub_ctle.value_axis.title  = "Log10(P(Transition occurs inside.))"

        plot_bathtub_dfe = Plot(plotdata)
        plot_bathtub_dfe.plot(("jitter_bins", "bathtub_dfe"), type="line", color="blue")
        plot_bathtub_dfe.value_range.high_setting =   0
        plot_bathtub_dfe.value_range.low_setting  = -18
        plot_bathtub_dfe.value_axis.tick_interval =   3
        plot_bathtub_dfe.title             = "Channel + Tx Preemphasis + CTLE + DFE"
        plot_bathtub_dfe.index_axis.title  = "Time (ps)"
        plot_bathtub_dfe.value_axis.title  = "Log10(P(Transition occurs inside.))"

        container_bathtub = GridPlotContainer(shape=(2,2))
        container_bathtub.add(plot_bathtub_chnl)
        container_bathtub.add(plot_bathtub_tx)
        container_bathtub.add(plot_bathtub_ctle)
        container_bathtub.add(plot_bathtub_dfe)
        self.plots_bathtub  = container_bathtub

        # These various plot customizing functions are left, for future reference.
        # plot19.index_range = plot5.index_range # Zoom x-axes in tandem.
        # plot4.value_range.high_setting = ui / 2.
        # plot4.value_range.low_setting  = -ui / 2.

        update_eyes(self)

    # Dependent variable definitions
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
            pj_rej_total  = pj_chnl  / pj_dfe
        if(rj_dfe):
            rj_rej_total  = rj_chnl  / rj_dfe

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
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (pj_chnl, pj_tx, 10. * log10(pj_rej_tx))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (rj_chnl, rj_tx, 10. * log10(rj_rej_tx))
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
                      (pj_chnl, pj_dfe, 10. * log10(pj_rej_total))
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % \
                      (rj_chnl, rj_dfe, 10. * log10(rj_rej_total))
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
        return perf_str + dly_str + jit_str

if __name__ == '__main__':
    PyBERT().configure_traits(view=traits_view)

