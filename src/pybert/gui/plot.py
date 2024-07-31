"""Plot definitions for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 21, 2015 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""

from chaco.api import ColorMapper, GridPlotContainer, Plot
from chaco.tools.api import PanTool, ZoomTool

from pybert.models.bert import update_eyes

PLOT_SPACING = 1
PLOT_PADDING = 75
PLOT_PADDING_BOT = 50


# pylint: disable=too-many-locals,too-many-statements
def make_plots(self, n_dfe_taps):
    """Create the plots used by the PyBERT GUI."""

    post_chnl_str = "Channel"
    post_tx_str = "+ Tx De-emphasis & Noise"
    post_ctle_str = "+ CTLE (& IBIS-AMI DFE if apropos)"
    post_dfe_str = "+ PyBERT Native DFE if enabled"

    plotdata = self.plotdata

    # - DFE tab
    plot2 = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING_BOT)
    plot2.plot(("t_ns", "ui_ests"), type="line", color="blue")
    plot2.title = "CDR Adaptation"
    plot2.index_axis.title = "Time (ns)"
    plot2.value_axis.title = "UI (ps)"

    plot9 = Plot(
        plotdata,
        auto_colors=["magenta", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "brown", "black"],
        padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING_BOT,
    )
    line_styles = ["line", "dash"]
    plot9.legend.labels = []
    for i in range(n_dfe_taps):
        name = f"tap{int(i + 1)}"
        plot9.plot(
            ("tap_weight_index", name + "_weights"),
            type="line",
            color="auto",
            style=line_styles[i // 10],
            name=name,
        )
        plot9.legend.labels.append(name)
    plot9.title = "DFE Adaptation"
    plot9.index_axis.title = "Sample Number"
    plot9.tools.append(PanTool(plot9, constrain=True, constrain_key=None, constrain_direction="x"))
    zoom9 = ZoomTool(plot9, tool_mode="range", axis="index", always_on=False)
    plot9.overlays.append(zoom9)
    plot9.legend.visible = True
    plot9.legend.align = "ul"

    plot_clk_per_hist = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING_BOT)
    plot_clk_per_hist.plot(("clk_per_hist_bins", "clk_per_hist_vals"), type="line", color="blue")
    plot_clk_per_hist.title = "CDR Clock Period Histogram"
    plot_clk_per_hist.index_axis.title = "Clock Period (ps)"
    plot_clk_per_hist.value_axis.title = "Bin Count"

    plot_clk_per_spec = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING_BOT)
    plot_clk_per_spec.plot(("clk_freqs", "clk_spec"), type="line", color="blue")
    plot_clk_per_spec.title = "CDR Clock Period Spectrum"
    plot_clk_per_spec.index_axis.title = "Frequency (symbol rate)"
    plot_clk_per_spec.value_axis.title = "|H(f)| (dB norm.)"
    plot_clk_per_spec.index_range.high_setting = 0.5
    plot_clk_per_spec.value_range.low_setting = -10
    zoom_clk_per_spec = ZoomTool(plot_clk_per_spec, tool_mode="range", axis="index", always_on=False)
    plot_clk_per_spec.overlays.append(zoom_clk_per_spec)

    container_dfe = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_dfe.add(plot2)
    container_dfe.add(plot9)
    container_dfe.add(plot_clk_per_hist)
    container_dfe.add(plot_clk_per_spec)
    self.plots_dfe = container_dfe
    self._dfe_plot = plot9  # pylint: disable=protected-access

    # - EQ Tune tab
    plot_h_tune = Plot(plotdata, padding_bottom=PLOT_PADDING)
    plot_h_tune.plot(("t_ns_opt", "clocks_tune"), name="Clocks", type="line", color="gray")
    plot_h_tune.plot(("t_ns_opt", "ctle_out_h_tune"), name="Equalized Pulse Response", type="line", color="blue")
    plot_h_tune.plot(("t_ns_opt", "p_chnl"), name="Channel Pulse Response", type="line", color="magenta")
    plot_h_tune.plot(("curs_ix", "curs_amp"), name="Main Cursor", type="segment", color="red")
    plot_h_tune.title = "Channel + Tx Preemphasis + CTLE (+ AMI DFE) + Ideal DFE"
    plot_h_tune.legend.labels = ["Channel Pulse Response",
                                 "Equalized Pulse Response",
                                 "Clocks",
                                 "Main Cursor"]
    plot_h_tune.legend.visible = True
    plot_h_tune.legend_alignment = "ur"
    plot_h_tune.index_axis.title = "Time (ns)"
    plot_h_tune.y_axis.title = "Pulse Response (V)"
    zoom_tune = ZoomTool(plot_h_tune, tool_mode="box", always_on=False)
    plot_h_tune.overlays.append(zoom_tune)

    container_tune = GridPlotContainer(shape=(1, 1))
    container_tune.add(plot_h_tune)
    self.plot_h_tune = container_tune

    # - Impulse Responses tab
    plot_h_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_h_chnl.plot(("t_ns_chnl", "chnl_h"), type="line", color="blue", name="Incremental")
    plot_h_chnl.title = post_chnl_str
    plot_h_chnl.index_axis.title = "Time (ns)"
    plot_h_chnl.y_axis.title = "Impulse Response (V/samp.)"
    plot_h_chnl.legend.visible = True
    plot_h_chnl.legend.align = "ur"
    zoom_h = ZoomTool(plot_h_chnl, tool_mode="range", axis="index", always_on=False)
    plot_h_chnl.overlays.append(zoom_h)

    plot_h_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_h_tx.plot(("t_ns_chnl", "tx_out_h"), type="line", color="red", name="Cumulative")
    plot_h_tx.title = post_tx_str
    plot_h_tx.index_axis.title = "Time (ns)"
    plot_h_tx.y_axis.title = "Impulse Response (V/samp.)"
    plot_h_tx.legend.visible = True
    plot_h_tx.legend.align = "ur"
    plot_h_tx.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

    plot_h_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_h_ctle.plot(("t_ns_chnl", "ctle_out_h"), type="line", color="red", name="Cumulative")
    plot_h_ctle.title = post_ctle_str
    plot_h_ctle.index_axis.title = "Time (ns)"
    plot_h_ctle.y_axis.title = "Impulse Response (V/samp.)"
    plot_h_ctle.legend.visible = True
    plot_h_ctle.legend.align = "ur"
    plot_h_ctle.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

    plot_h_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_h_dfe.plot(("t_ns_chnl", "dfe_out_h"), type="line", color="red", name="Cumulative")
    plot_h_dfe.title = post_dfe_str
    plot_h_dfe.index_axis.title = "Time (ns)"
    plot_h_dfe.y_axis.title = "Impulse Response (V/samp.)"
    plot_h_dfe.legend.visible = True
    plot_h_dfe.legend.align = "ur"
    plot_h_dfe.index_range = plot_h_chnl.index_range  # Zoom x-axes in tandem.

    container_h = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_h.add(plot_h_chnl)
    container_h.add(plot_h_tx)
    container_h.add(plot_h_ctle)
    container_h.add(plot_h_dfe)
    self.plots_h = container_h

    # - Step Responses tab
    plot_s_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_s_chnl.plot(("t_ns_chnl", "chnl_s"), type="line", color="blue", name="Incremental")
    plot_s_chnl.title = post_chnl_str
    plot_s_chnl.index_axis.title = "Time (ns)"
    plot_s_chnl.y_axis.title = "Step Response (V)"
    plot_s_chnl.legend.visible = True
    plot_s_chnl.legend.align = "lr"
    zoom_s = ZoomTool(plot_s_chnl, tool_mode="range", axis="index", always_on=False)
    plot_s_chnl.overlays.append(zoom_s)

    plot_s_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_s_tx.plot(("t_ns_chnl", "tx_s"), type="line", color="blue", name="Incremental")
    plot_s_tx.plot(("t_ns_chnl", "tx_out_s"), type="line", color="red", name="Cumulative")
    plot_s_tx.title = post_tx_str
    plot_s_tx.index_axis.title = "Time (ns)"
    plot_s_tx.y_axis.title = "Step Response (V)"
    plot_s_tx.legend.visible = True
    plot_s_tx.legend.align = "lr"
    plot_s_tx.legend.labels = ["Incremental", "Cumulative"]
    plot_s_tx.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

    plot_s_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_s_ctle.plot(("t_ns_chnl", "ctle_s"), type="line", color="blue", name="Incremental")
    plot_s_ctle.plot(("t_ns_chnl", "ctle_out_s"), type="line", color="red", name="Cumulative")
    plot_s_ctle.title = post_ctle_str
    plot_s_ctle.index_axis.title = "Time (ns)"
    plot_s_ctle.y_axis.title = "Step Response (V)"
    plot_s_ctle.legend.visible = True
    plot_s_ctle.legend.align = "lr"
    plot_s_ctle.legend.labels = ["Incremental", "Cumulative"]
    plot_s_ctle.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

    plot_s_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_s_dfe.plot(("t_ns_chnl", "dfe_s"), type="line", color="blue", name="Incremental")
    plot_s_dfe.plot(("t_ns_chnl", "dfe_out_s"), type="line", color="red", name="Cumulative")
    plot_s_dfe.title = post_dfe_str
    plot_s_dfe.index_axis.title = "Time (ns)"
    plot_s_dfe.y_axis.title = "Step Response (V)"
    plot_s_dfe.legend.visible = True
    plot_s_dfe.legend.align = "lr"
    plot_s_dfe.legend.labels = ["Incremental", "Cumulative"]
    plot_s_dfe.index_range = plot_s_chnl.index_range  # Zoom x-axes in tandem.

    container_s = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_s.add(plot_s_chnl)
    container_s.add(plot_s_tx)
    container_s.add(plot_s_ctle)
    container_s.add(plot_s_dfe)
    self.plots_s = container_s

    # - Pulse Responses tab
    plot_p_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_p_chnl.plot(("t_ns_chnl", "chnl_p"), type="line", color="blue", name="Incremental")
    plot_p_chnl.title = post_chnl_str
    plot_p_chnl.index_axis.title = "Time (ns)"
    plot_p_chnl.y_axis.title = "Pulse Response (V)"
    plot_p_chnl.legend.visible = True
    plot_p_chnl.legend.align = "ur"
    zoom_p = ZoomTool(plot_p_chnl, tool_mode="range", axis="index", always_on=False)
    plot_p_chnl.overlays.append(zoom_p)

    plot_p_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_p_tx.plot(("t_ns_chnl", "tx_out_p"), type="line", color="red", name="Cumulative")
    plot_p_tx.title = post_tx_str
    plot_p_tx.index_axis.title = "Time (ns)"
    plot_p_tx.y_axis.title = "Pulse Response (V)"
    plot_p_tx.legend.visible = True
    plot_p_tx.legend.align = "ur"
    plot_p_tx.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

    plot_p_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_p_ctle.plot(("t_ns_chnl", "ctle_out_p"), type="line", color="red", name="Cumulative")
    plot_p_ctle.title = post_ctle_str
    plot_p_ctle.index_axis.title = "Time (ns)"
    plot_p_ctle.y_axis.title = "Pulse Response (V)"
    plot_p_ctle.legend.visible = True
    plot_p_ctle.legend.align = "ur"
    plot_p_ctle.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

    plot_p_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_p_dfe.plot(("t_ns_chnl", "dfe_out_p"), type="line", color="red", name="Cumulative")
    plot_p_dfe.title = post_dfe_str
    plot_p_dfe.index_axis.title = "Time (ns)"
    plot_p_dfe.y_axis.title = "Pulse Response (V)"
    plot_p_dfe.legend.visible = True
    plot_p_dfe.legend.align = "ur"
    plot_p_dfe.index_range = plot_p_chnl.index_range  # Zoom x-axes in tandem.

    container_p = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_p.add(plot_p_chnl)
    container_p.add(plot_p_tx)
    container_p.add(plot_p_ctle)
    container_p.add(plot_p_dfe)
    self.plots_p = container_p

    # - Frequency Responses tab
    plot_H_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_H_chnl.plot(("f_GHz", "chnl_H_raw"), type="line", color="black", name="Perfect Term.", index_scale="log")
    plot_H_chnl.plot(("f_GHz", "chnl_H"), type="line", color="blue", name="Actual Term.", index_scale="log")
    plot_H_chnl.plot(("f_GHz", "chnl_trimmed_H"), type="line", color="red", name="Trimmed Impulse", index_scale="log")
    plot_H_chnl.title = post_chnl_str
    plot_H_chnl.index_axis.title = "Frequency (GHz)"
    plot_H_chnl.y_axis.title = "Frequency Response (dB)"
    plot_H_chnl.index_range.low_setting = 0.01
    plot_H_chnl.value_range.low_setting = -40.0
    plot_H_chnl.legend.visible = True
    plot_H_chnl.legend.align = "ll"

    plot_H_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_H_tx.plot(("f_GHz", "tx_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_tx.plot(("f_GHz", "tx_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_tx.title = post_tx_str
    plot_H_tx.index_axis.title = "Frequency (GHz)"
    plot_H_tx.y_axis.title = "Frequency Response (dB)"
    plot_H_tx.index_range.low_setting = 0.01
    plot_H_tx.value_range.low_setting = -40.0
    plot_H_tx.legend.visible = True
    plot_H_tx.legend.labels = ["Incremental", "Cumulative"]
    plot_H_tx.legend.align = "ll"

    plot_H_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_H_ctle.plot(("f_GHz", "ctle_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_ctle.plot(("f_GHz", "ctle_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_ctle.title = post_ctle_str
    plot_H_ctle.index_axis.title = "Frequency (GHz)"
    plot_H_ctle.y_axis.title = "Frequency Response (dB)"
    plot_H_ctle.index_range.low_setting = 0.01
    plot_H_ctle.value_range.low_setting = -40.0
    plot_H_ctle.legend.visible = True
    plot_H_ctle.legend.align = "ll"
    plot_H_ctle.legend.labels = ["Incremental", "Cumulative"]

    plot_H_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_H_dfe.plot(("f_GHz", "dfe_H"), type="line", color="blue", name="Incremental", index_scale="log")
    plot_H_dfe.plot(("f_GHz", "dfe_out_H"), type="line", color="red", name="Cumulative", index_scale="log")
    plot_H_dfe.title = post_dfe_str
    plot_H_dfe.index_axis.title = "Frequency (GHz)"
    plot_H_dfe.y_axis.title = "Frequency Response (dB)"
    plot_H_dfe.index_range.low_setting = 0.01
    plot_H_dfe.value_range.low_setting = -40.0
    plot_H_dfe.legend.visible = True
    plot_H_dfe.legend.align = "ll"
    plot_H_dfe.legend.labels = ["Incremental", "Cumulative"]

    container_H = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_H.add(plot_H_chnl)
    container_H.add(plot_H_tx)
    container_H.add(plot_H_ctle)
    container_H.add(plot_H_dfe)
    self.plots_H = container_H

    # - Outputs tab
    plot_out_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_out_chnl.plot(("t_ns", "ideal_signal"), type="line", color="lightgrey")
    plot_out_chnl.plot(("t_ns", "chnl_out"), type="line", color="blue")
    plot_out_chnl.title = post_chnl_str
    plot_out_chnl.index_axis.title = "Time (ns)"
    plot_out_chnl.y_axis.title = "Output (V)"
    plot_out_chnl.tools.append(PanTool(plot_out_chnl, constrain=True, constrain_key=None, constrain_direction="x"))
    zoom_out_chnl = ZoomTool(plot_out_chnl, tool_mode="range", axis="index", always_on=False)
    plot_out_chnl.overlays.append(zoom_out_chnl)

    plot_out_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_out_tx.plot(("t_ns", "rx_in"), type="line", color="blue")
    plot_out_tx.title = post_tx_str
    plot_out_tx.index_axis.title = "Time (ns)"
    plot_out_tx.y_axis.title = "Output (V)"
    plot_out_tx.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

    plot_out_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_out_ctle.plot(("t_ns", "ctle_out"), type="line", color="blue")
    plot_out_ctle.title = post_ctle_str
    plot_out_ctle.index_axis.title = "Time (ns)"
    plot_out_ctle.y_axis.title = "Output (V)"
    plot_out_ctle.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

    plot_out_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_out_dfe.plot(("t_ns", "dfe_out"), type="line", color="blue")
    plot_out_dfe.title = post_dfe_str
    plot_out_dfe.index_axis.title = "Time (ns)"
    plot_out_dfe.y_axis.title = "Output (V)"
    plot_out_dfe.index_range = plot_out_chnl.index_range  # Zoom x-axes in tandem.

    container_out = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_out.add(plot_out_chnl)
    container_out.add(plot_out_tx)
    container_out.add(plot_out_ctle)
    container_out.add(plot_out_dfe)
    self.plots_out = container_out

    # - Eye Diagrams tab
    seg_map = {
        "red": [
            (0.00, 0.00, 0.00),  # black
            (0.00001, 0.00, 0.00),  # blue
            (0.15, 0.00, 0.00),  # cyan
            (0.30, 0.00, 0.00),  # green
            (0.45, 1.00, 1.00),  # yellow
            (0.60, 1.00, 1.00),  # orange
            (0.75, 1.00, 1.00),  # red
            (0.90, 1.00, 1.00),  # pink
            (1.00, 1.00, 1.00),  # white
        ],
        "green": [
            (0.00, 0.00, 0.00),  # black
            (0.00001, 0.00, 0.00),  # blue
            (0.15, 0.50, 0.50),  # cyan
            (0.30, 0.50, 0.50),  # green
            (0.45, 1.00, 1.00),  # yellow
            (0.60, 0.50, 0.50),  # orange
            (0.75, 0.00, 0.00),  # red
            (0.90, 0.50, 0.50),  # pink
            (1.00, 1.00, 1.00),  # white
        ],
        "blue": [
            (0.00, 0.00, 0.00),  # black
            (1e-18, 0.50, 0.50),  # blue
            (0.15, 0.50, 0.50),  # cyan
            (0.30, 0.00, 0.00),  # green
            (0.45, 0.00, 0.00),  # yellow
            (0.60, 0.00, 0.00),  # orange
            (0.75, 0.00, 0.00),  # red
            (0.90, 0.50, 0.50),  # pink
            (1.00, 1.00, 1.00),  # white
        ]
    }
    clr_map = ColorMapper.from_segment_map(seg_map)
    self.clr_map = clr_map

    plot_eye_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_eye_chnl.img_plot("eye_chnl", colormap=clr_map)
    plot_eye_chnl.y_direction = "normal"
    plot_eye_chnl.components[0].y_direction = "normal"
    plot_eye_chnl.title = post_chnl_str
    plot_eye_chnl.x_axis.title = "Time (ps)"
    plot_eye_chnl.x_axis.orientation = "bottom"
    plot_eye_chnl.y_axis.title = "Signal Level (V)"
    plot_eye_chnl.x_grid.visible = True
    plot_eye_chnl.y_grid.visible = True
    plot_eye_chnl.x_grid.line_color = "gray"
    plot_eye_chnl.y_grid.line_color = "gray"

    plot_eye_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_eye_tx.img_plot("eye_tx", colormap=clr_map)
    plot_eye_tx.y_direction = "normal"
    plot_eye_tx.components[0].y_direction = "normal"
    plot_eye_tx.title = post_tx_str
    plot_eye_tx.x_axis.title = "Time (ps)"
    plot_eye_tx.x_axis.orientation = "bottom"
    plot_eye_tx.y_axis.title = "Signal Level (V)"
    plot_eye_tx.x_grid.visible = True
    plot_eye_tx.y_grid.visible = True
    plot_eye_tx.x_grid.line_color = "gray"
    plot_eye_tx.y_grid.line_color = "gray"

    plot_eye_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_eye_ctle.img_plot("eye_ctle", colormap=clr_map)
    plot_eye_ctle.y_direction = "normal"
    plot_eye_ctle.components[0].y_direction = "normal"
    plot_eye_ctle.title = post_ctle_str
    plot_eye_ctle.x_axis.title = "Time (ps)"
    plot_eye_ctle.x_axis.orientation = "bottom"
    plot_eye_ctle.y_axis.title = "Signal Level (V)"
    plot_eye_ctle.x_grid.visible = True
    plot_eye_ctle.y_grid.visible = True
    plot_eye_ctle.x_grid.line_color = "gray"
    plot_eye_ctle.y_grid.line_color = "gray"

    plot_eye_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_eye_dfe.img_plot("eye_dfe", colormap=clr_map)
    plot_eye_dfe.y_direction = "normal"
    plot_eye_dfe.components[0].y_direction = "normal"
    plot_eye_dfe.title = post_dfe_str
    plot_eye_dfe.x_axis.title = "Time (ps)"
    plot_eye_dfe.x_axis.orientation = "bottom"
    plot_eye_dfe.y_axis.title = "Signal Level (V)"
    plot_eye_dfe.x_grid.visible = True
    plot_eye_dfe.y_grid.visible = True
    plot_eye_dfe.x_grid.line_color = "gray"
    plot_eye_dfe.y_grid.line_color = "gray"

    container_eye = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_eye.add(plot_eye_chnl)
    container_eye.add(plot_eye_tx)
    container_eye.add(plot_eye_ctle)
    container_eye.add(plot_eye_dfe)
    self.plots_eye = container_eye

    # - Jitter Distributions tab
    plot_jitter_dist_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_dist_chnl.plot(("jitter_bins", "jitter_chnl"), type="line", color="blue", name="Total")
    plot_jitter_dist_chnl.plot(("jitter_bins", "jitter_ext_chnl"), type="line", color="red", name="Data-Ind.")
    plot_jitter_dist_chnl.title = post_chnl_str
    plot_jitter_dist_chnl.index_axis.title = "Time (ps)"
    plot_jitter_dist_chnl.value_axis.title = "PDF"
    plot_jitter_dist_chnl.legend.visible = True
    plot_jitter_dist_chnl.legend.align = "ur"

    plot_jitter_dist_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_dist_tx.plot(("jitter_bins", "jitter_tx"), type="line", color="blue", name="Total")
    plot_jitter_dist_tx.plot(("jitter_bins", "jitter_ext_tx"), type="line", color="red", name="Data-Ind.")
    plot_jitter_dist_tx.title = post_tx_str
    plot_jitter_dist_tx.index_axis.title = "Time (ps)"
    plot_jitter_dist_tx.value_axis.title = "PDF"
    plot_jitter_dist_tx.legend.visible = True
    plot_jitter_dist_tx.legend.align = "ur"

    plot_jitter_dist_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_dist_ctle.plot(("jitter_bins", "jitter_ctle"), type="line", color="blue", name="Total")
    plot_jitter_dist_ctle.plot(("jitter_bins", "jitter_ext_ctle"), type="line", color="red", name="Data-Ind.")
    plot_jitter_dist_ctle.title = post_ctle_str
    plot_jitter_dist_ctle.index_axis.title = "Time (ps)"
    plot_jitter_dist_ctle.value_axis.title = "PDF"
    plot_jitter_dist_ctle.legend.visible = True
    plot_jitter_dist_ctle.legend.align = "ur"

    plot_jitter_dist_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_dist_dfe.plot(("jitter_bins", "jitter_dfe"), type="line", color="blue", name="Total")
    plot_jitter_dist_dfe.plot(("jitter_bins", "jitter_ext_dfe"), type="line", color="red", name="Data-Ind.")
    plot_jitter_dist_dfe.title = post_dfe_str
    plot_jitter_dist_dfe.index_axis.title = "Time (ps)"
    plot_jitter_dist_dfe.value_axis.title = "PDF"
    plot_jitter_dist_dfe.legend.visible = True
    plot_jitter_dist_dfe.legend.align = "ur"

    container_jitter_dist = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_jitter_dist.add(plot_jitter_dist_chnl)
    container_jitter_dist.add(plot_jitter_dist_tx)
    container_jitter_dist.add(plot_jitter_dist_ctle)
    container_jitter_dist.add(plot_jitter_dist_dfe)
    self.plots_jitter_dist = container_jitter_dist

    # - Jitter Spectrums tab
    plot_jitter_spec_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_spec_chnl.plot(("f_MHz", "jitter_spectrum_chnl"),     type="line", color="blue",    name="Total")
    plot_jitter_spec_chnl.plot(("f_MHz", "jitter_ind_spectrum_chnl"), type="line", color="red",     name="Data Independent")
    plot_jitter_spec_chnl.plot(("f_MHz", "thresh_chnl"),              type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_chnl.title = post_chnl_str
    plot_jitter_spec_chnl.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_chnl.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_chnl.tools.append(
        PanTool(plot_jitter_spec_chnl, constrain=True, constrain_key=None, constrain_direction="x")
    )
    zoom_jitter_spec_chnl = ZoomTool(plot_jitter_spec_chnl, tool_mode="range", axis="index", always_on=False)
    plot_jitter_spec_chnl.overlays.append(zoom_jitter_spec_chnl)
    plot_jitter_spec_chnl.legend.visible = True
    plot_jitter_spec_chnl.legend.align = "lr"

    plot_jitter_spec_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_spec_tx.plot(("f_MHz", "jitter_spectrum_tx"),     type="line", color="blue",    name="Total")
    plot_jitter_spec_tx.plot(("f_MHz", "jitter_ind_spectrum_tx"), type="line", color="red",     name="Data Independent")
    plot_jitter_spec_tx.plot(("f_MHz", "thresh_tx"),              type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_tx.title = post_tx_str
    plot_jitter_spec_tx.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_tx.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_tx.value_range.low_setting = -40.0
    plot_jitter_spec_tx.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_tx.legend.visible = True
    plot_jitter_spec_tx.legend.align = "lr"

    plot_jitter_spec_chnl.value_range = plot_jitter_spec_tx.value_range

    plot_jitter_spec_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_spec_ctle.plot(("f_MHz", "jitter_spectrum_ctle"),     type="line", color="blue",    name="Total")
    plot_jitter_spec_ctle.plot(("f_MHz", "jitter_ind_spectrum_ctle"), type="line", color="red",     name="Data Independent")
    plot_jitter_spec_ctle.plot(("f_MHz", "thresh_ctle"),              type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_ctle.title = post_ctle_str
    plot_jitter_spec_ctle.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_ctle.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_ctle.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_ctle.legend.visible = True
    plot_jitter_spec_ctle.legend.align = "lr"
    plot_jitter_spec_ctle.value_range = plot_jitter_spec_tx.value_range

    plot_jitter_spec_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_jitter_spec_dfe.plot(("f_MHz_dfe", "jitter_spectrum_dfe"),     type="line", color="blue",    name="Total")
    plot_jitter_spec_dfe.plot(("f_MHz_dfe", "jitter_ind_spectrum_dfe"), type="line", color="red",     name="Data Independent")
    plot_jitter_spec_dfe.plot(("f_MHz_dfe", "thresh_dfe"),              type="line", color="magenta", name="Pj Threshold")
    plot_jitter_spec_dfe.title = post_dfe_str
    plot_jitter_spec_dfe.index_axis.title = "Frequency (MHz)"
    plot_jitter_spec_dfe.value_axis.title = "|FFT(TIE)| (dBui)"
    plot_jitter_spec_dfe.index_range = plot_jitter_spec_chnl.index_range  # Zoom x-axes in tandem.
    plot_jitter_spec_dfe.legend.visible = True
    plot_jitter_spec_dfe.legend.align = "lr"
    plot_jitter_spec_dfe.value_range = plot_jitter_spec_tx.value_range

    container_jitter_spec = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_jitter_spec.add(plot_jitter_spec_chnl)
    container_jitter_spec.add(plot_jitter_spec_tx)
    container_jitter_spec.add(plot_jitter_spec_ctle)
    container_jitter_spec.add(plot_jitter_spec_dfe)
    self.plots_jitter_spec = container_jitter_spec

    # - Bathtub Curves tab
    # ToDo: Make extrapolated portion of curve dashed.
    plot_bathtub_chnl = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_bathtub_chnl.plot(("jitter_bins", "bathtub_chnl"), type="line", color="blue")
    plot_bathtub_chnl.value_range.high_setting = 0
    plot_bathtub_chnl.value_range.low_setting = -12
    plot_bathtub_chnl.value_axis.tick_interval = 3
    plot_bathtub_chnl.title = post_chnl_str
    plot_bathtub_chnl.index_axis.title = "Time (ps)"
    plot_bathtub_chnl.value_axis.title = "Log10(P(Transition occurs inside.))"

    plot_bathtub_tx = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_bathtub_tx.plot(("jitter_bins", "bathtub_tx"), type="line", color="blue")
    plot_bathtub_tx.value_range.high_setting = 0
    plot_bathtub_tx.value_range.low_setting = -12
    plot_bathtub_tx.value_axis.tick_interval = 3
    plot_bathtub_tx.title = post_tx_str
    plot_bathtub_tx.index_axis.title = "Time (ps)"
    plot_bathtub_tx.value_axis.title = "Log10(P(Transition occurs inside.))"

    plot_bathtub_ctle = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_bathtub_ctle.plot(("jitter_bins", "bathtub_ctle"), type="line", color="blue")
    plot_bathtub_ctle.value_range.high_setting = 0
    plot_bathtub_ctle.value_range.low_setting = -12
    plot_bathtub_ctle.value_axis.tick_interval = 3
    plot_bathtub_ctle.title = post_ctle_str
    plot_bathtub_ctle.index_axis.title = "Time (ps)"
    plot_bathtub_ctle.value_axis.title = "Log10(P(Transition occurs inside.))"

    plot_bathtub_dfe = Plot(plotdata, padding_left=PLOT_PADDING, padding_bottom=PLOT_PADDING)
    plot_bathtub_dfe.plot(("jitter_bins", "bathtub_dfe"), type="line", color="blue")
    plot_bathtub_dfe.value_range.high_setting = 0
    plot_bathtub_dfe.value_range.low_setting = -12
    plot_bathtub_dfe.value_axis.tick_interval = 3
    plot_bathtub_dfe.title = post_dfe_str
    plot_bathtub_dfe.index_axis.title = "Time (ps)"
    plot_bathtub_dfe.value_axis.title = "Log10(P(Transition occurs inside.))"

    container_bathtub = GridPlotContainer(shape=(2, 2), spacing=(PLOT_SPACING, PLOT_SPACING))
    container_bathtub.add(plot_bathtub_chnl)
    container_bathtub.add(plot_bathtub_tx)
    container_bathtub.add(plot_bathtub_ctle)
    container_bathtub.add(plot_bathtub_dfe)
    self.plots_bathtub = container_bathtub

    update_eyes(self)
