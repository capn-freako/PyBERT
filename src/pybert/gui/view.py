"""Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from enable.component_editor import ComponentEditor
from pyface.image_resource import ImageResource
from traitsui.api import (  # CloseAction,
    Action,
    CheckListEditor,
    FileEditor,
    Group,
    HGroup,
    Item,
    Menu,
    MenuBar,
    NoButtons,
    ObjectColumn,
    Separator,
    TableEditor,
    TextEditor,
    VGroup,
    View,
    spring,
)

from pybert.gui.handler import MyHandler

HIGH_RES = True

# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
            HGroup(
                VGroup(  # Simulation Control
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(
                                    name="bit_rate",
                                    label="Bit Rate",
                                    tooltip="bit rate",
                                    show_label=True,
                                    enabled_when="True",
                                    editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                                ),
                                Item(label="Gbps"),
                            ),
                            Item(
                                name="nspui",
                                label="Samps. per UI",
                                tooltip="# of samples per unit interval",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="mod_type",
                                label="Modulation",
                                tooltip="line signalling/modulation scheme",
                                editor=CheckListEditor(values=[(0, "NRZ"), (1, "Duo-binary"), (2, "PAM-4")]),
                            ),
                            label="Rate && Modulation",
                            show_border=True,
                        ),
                        VGroup(
                            HGroup(
                                Item(
                                    name="pattern",
                                    label="Pattern",
                                    tooltip="pattern to use to construct bit stream",
                                ),
                                spring,
                                Item(
                                    name="seed",
                                    label="Seed",
                                    tooltip="LFSR seed. 0 means new random seed for each run.",
                                ),
                            ),
                            Item(
                                name="nbits",
                                label="Nbits",
                                tooltip="# of bits to run",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="eye_bits",
                                label="EyeBits",
                                tooltip="# of bits to use to form eye diagrams",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            label="Test Pattern",
                            show_border=True,
                        ),
                        VGroup(
                            HGroup(
                                Item(name="vod", label="Vod", tooltip="Tx output voltage into matched load"),
                                Item(label="V"),
                            ),
                            HGroup(
                                Item(name="rn", label="Rn", tooltip="standard deviation of random noise"),
                                Item(label="V"),
                            ),
                            HGroup(
                                Item(name="pn_mag", label="Pn", tooltip="peak magnitude of periodic noise"),
                                Item(label="V"),
                            ),
                            HGroup(
                                Item(name="pn_freq", label="f(Pn)", tooltip="frequency of periodic noise"),
                                Item(label="MHz"),
                            ),
                            label="Tx Level && Noise",
                            show_border=True,
                        ),
                    ),
                    HGroup(
                        Item(name="debug", label="Debug", tooltip="Enable to log extra information to console."),
                        label="Miscellaneous Options",
                        show_border=True,
                    ),
                    label="Simulation Control",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item(
                            name="impulse_length",
                            label="Impulse Response Length",
                            tooltip="Manual impulse response length override",
                        ),
                        Item(label="ns"),
                        spring,
                    ),
                    HGroup(
                        Item(
                            name="thresh",
                            label="Pj Threshold",
                            tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)",
                        ),
                        Item(label="sigma"),
                        spring,
                    ),
                    HGroup(
                        Item(
                            name="f_max",
                            label="fMax",
                            tooltip="Maximum frequency used for plotting, modeling, and signal processing. (GHz)",
                        ),
                        Item(label="GHz"),
                        spring,
                    ),
                    HGroup(
                        Item(
                            name="f_step",
                            label="fStep",
                            tooltip="Frequency step used for plotting, modeling, and signal processing. (MHz)",
                        ),
                        Item(label="MHz"),
                        spring,
                    ),
                    label="Analysis Parameters",
                    show_border=True,
                ),
            ),
            HGroup(  # "Channel"
                VGroup(  # "Tx"
                    VGroup(
                        HGroup(
                            Item(
                                name="tx_ibis_file",
                                label="File",
                                springy=True,
                                editor=FileEditor(dialog_style="open", filter=["*.ibs"]),
                            ),
                            Item(name="tx_ibis_valid", label="Valid", style="simple", enabled_when="False"),
                        ),
                        HGroup(
                            Item(name="tx_use_ibis", label="Use IBIS"),
                            Item(name="btn_sel_tx", show_label=False),
                            Item(name="btn_view_tx", show_label=False),
                            enabled_when="tx_ibis_valid",
                        ),
                        Item(name="tx_use_ts4", label="Use on-die S-parameters.",
                             enabled_when="tx_use_ibis and tx_ibis_valid and tx_has_ts4",),
                        label="IBIS",
                        show_border=True,
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                name="rs",
                                label="Tx_Rs",
                                tooltip="Tx differential source impedance",
                            ),
                            Item(label="Ohms"),
                            spring,
                        ),
                        HGroup(
                            Item(
                                name="cout",
                                label="Tx_Cout",
                                tooltip="Tx parasitic output capacitance (each pin)",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                            ),
                            Item(label="pF"),
                            spring,
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="tx_use_ibis == False",
                    ),
                    label="Tx",
                    show_border=True,
                ),
                VGroup(  # Interconnect
                    VGroup(  # From File
                        VGroup(
                            Item(
                                name="ch_file",
                                label="File",
                                editor=FileEditor(dialog_style="open"),
                            ),
                            HGroup(
                                Item(
                                    name="use_ch_file",
                                    label="Use file",
                                ),
                                Item(
                                    name="renumber",
                                    label="Fix port numbering",
                                ),
                                spring,
                            ),
                        ),
                        label="From File",
                        show_border=True,
                    ),
                    HGroup(  # Native (i.e. - Howard Johnson's) interconnect model.
                        VGroup(
                            HGroup(
                                Item(
                                    name="l_ch",
                                    label="Length",
                                    tooltip="interconnect length",
                                ),
                                Item(label="m"),
                                spring,
                            ),
                            HGroup(
                                Item(
                                    name="Theta0",
                                    label="Loss Tan.",
                                    tooltip="dielectric loss tangent",
                                ),
                                spring,
                            ),
                            HGroup(
                                Item(
                                    name="Z0",
                                    label="Z0",
                                    tooltip="characteristic differential impedance",
                                ),
                                Item(label="Ohms"),
                                spring,
                            ),
                            HGroup(
                                Item(
                                    name="v0",
                                    label="v_rel",
                                    tooltip="normalized propagation velocity",
                                ),
                                Item(label="c"),
                                spring,
                            ),
                        ),
                        spring,
                        VGroup(
                            HGroup(
                                spring,
                                Item(
                                    name="Rdc",
                                    label="Rdc",
                                    tooltip="d.c. resistance",
                                ),
                                Item(label="Ohms"),
                            ),
                            HGroup(
                                spring,
                                Item(
                                    name="w0",
                                    label="w0",
                                    tooltip="transition frequency",
                                ),
                                Item(label="rads./s"),
                            ),
                            HGroup(
                                spring,
                                Item(
                                    name="R0",
                                    label="R0",
                                    tooltip="skin effect resistance",
                                ),
                                Item(label="Ohms"),
                            ),
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="use_ch_file == False",
                    ),
                    HGroup(  # Misc.
                        Item(
                            name="use_window",
                            label="Apply window",
                            tooltip="Apply raised cosine window to frequency response before FFT()'ing.",
                        ),
                        label="Misc.",
                        show_border=True,
                    ),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(  # Rx
                    VGroup(
                        HGroup(
                            Item(
                                name="rx_ibis_file",
                                label="File",
                                springy=True,
                                editor=FileEditor(dialog_style="open", filter=["*.ibs"]),
                            ),
                            Item(name="rx_ibis_valid", label="Valid", style="simple", enabled_when="False"),
                        ),
                        HGroup(
                            Item(name="rx_use_ibis", label="Use IBIS"),
                            Item(name="btn_sel_rx", show_label=False),
                            Item(name="btn_view_rx", show_label=False),
                            enabled_when="rx_ibis_valid",
                        ),
                        Item(name="rx_use_ts4", label="Use on-die S-parameters.",
                             enabled_when="rx_use_ibis and rx_ibis_valid and rx_has_ts4",),
                        label="IBIS",
                        show_border=True,
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                name="rin",
                                label="Rx_Rin",
                                tooltip="Rx differential input impedance",
                            ),
                            Item(label="Ohms"),
                            spring,
                        ),
                        HGroup(
                            Item(
                                name="cin",
                                label="Rx_Cin",
                                tooltip="Rx parasitic input capacitance (each pin)",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                            ),
                            Item(label="pF"),
                            spring,
                        ),
                        HGroup(
                            Item(
                                name="cac",
                                label="Rx_Cac",
                                tooltip="Rx a.c. coupling capacitance (each pin)",
                            ),
                            Item(label="uF"),
                            spring,
                        ),
                        HGroup(
                            Item(
                                name="rx_use_viterbi", label="Use Viterbi",
                                tooltip="Apply MLSD to recovered symbols, using Viterbi algorithm.",
                            ),
                            Item(
                                name="rx_viterbi_symbols", label="# Symbols",
                                tooltip="Number of symbols to include in MLSD trellis.",
                            ),
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="rx_use_ibis == False",
                    ),
                    label="Rx",
                    show_border=True,
                ),
                label="Channel",
                show_border=True,
            ),
            label="Config.",
            id="config",
        ),
        # "Equalization" tab.
        HGroup(  # Channel Parameters
            VGroup(
                VGroup(
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(name="tx_ami_file", label="AMI File:", style="readonly", springy=True),
                                Item(name="tx_ami_valid", label="Valid", style="simple", enabled_when="False"),
                            ),
                            HGroup(
                                Item(name="tx_dll_file", label="DLL File:", style="readonly", springy=True),
                                Item(name="tx_dll_valid", label="Valid", style="simple", enabled_when="False"),
                            ),
                        ),
                        VGroup(
                            Item(
                                name="tx_use_ami",
                                label="Use AMI",
                                tooltip="You must select both files, first.",
                                enabled_when="tx_ami_valid == True and tx_dll_valid == True",
                            ),
                            Item(
                                name="tx_use_getwave",
                                label="Use GetWave",
                                tooltip="Use the model's GetWave() function.",
                                enabled_when="tx_use_ami and tx_has_getwave",
                            ),
                            Item(
                                "btn_cfg_tx",
                                show_label=False,
                                tooltip="Configure Tx AMI parameters.",
                                enabled_when="tx_ami_valid == True",
                            ),
                        ),
                    ),
                    label="IBIS-AMI",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="tx_taps",
                        editor=TableEditor(
                            columns=[
                                ObjectColumn(name="name", editable=False),
                                ObjectColumn(name="enabled", style="simple"),
                                ObjectColumn(name="value", format="%+05.3f", horizontal_alignment="center"),
                            ],
                            configurable=False,
                            reorderable=False,
                            sortable=False,
                            selection_mode="cell",
                            rows=4,
                        ),
                        show_label=False,
                    ),
                    label="Native",
                    show_border=True,
                    enabled_when="tx_use_ami == False",
                ),
                label="Tx Equalization",
                show_border=True,
            ),
            VGroup(
                VGroup(
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(name="rx_ami_file", label="AMI File:", style="readonly", springy=True),
                                Item(name="rx_ami_valid", label="Valid", style="simple", enabled_when="False"),
                            ),
                            HGroup(
                                Item(name="rx_dll_file", label="DLL File:", style="readonly", springy=True),
                                Item(name="rx_dll_valid", label="Valid", style="simple", enabled_when="False"),
                            ),
                        ),
                        VGroup(
                            Item(
                                name="rx_use_ami",
                                label="Use AMI",
                                tooltip="You must select both files, first.",
                                enabled_when="rx_ami_valid == True and rx_dll_valid == True",
                            ),
                            Item(
                                name="rx_use_getwave",
                                label="Use GetWave",
                                tooltip="Use the model's GetWave() function.",
                                enabled_when="rx_use_ami and rx_has_getwave",
                            ),
                            Item(
                                name="rx_use_clocks",
                                label="Use Clocks",
                                tooltip="Use the clock times returned by the model's GetWave() function.",
                                enabled_when="rx_use_getwave",
                            ),
                            Item(
                                "btn_cfg_rx",
                                show_label=False,
                                tooltip="Configure Rx AMI parameters.",
                                enabled_when="rx_ami_valid == True",
                            ),
                        ),
                    ),
                    label="IBIS-AMI",
                    show_border=True,
                ),
                VGroup(
                    VGroup(  # CTLE
                        Item(name="ctle_enable", label="Enable", tooltip="CTLE enable",),
                        HGroup(  # File
                            Item(
                                name="use_ctle_file",
                                label="Use",
                                tooltip="Select CTLE impulse/step response from file.",
                                enabled_when="ctle_file",
                            ),
                            Item(
                                name="ctle_file",
                                label="Filename",
                                enabled_when="use_ctle_file == True",
                                editor=FileEditor(dialog_style="open"),
                            ),
                            label="File",
                            show_border=True,
                        ),
                        VGroup(  # Model
                            HGroup(
                                Item(
                                    name="peak_freq",
                                    label="CTLE fp",
                                    tooltip="CTLE peaking frequency (GHz)",
                                    enabled_when="use_ctle_file == False",
                                ),
                                Item(label="GHz"),
                                spring,
                                Item(
                                    name="rx_bw",
                                    label="Bandwidth",
                                    tooltip="unequalized signal path bandwidth (GHz).",
                                    enabled_when="use_ctle_file == False",
                                ),
                                Item(label="GHz"),
                            ),
                            HGroup(
                                Item(
                                    name="peak_mag",
                                    label="CTLE boost",
                                    tooltip="CTLE peaking magnitude (dB)",
                                    format_str="%4.1f",
                                    enabled_when="use_ctle_file == False",
                                ),
                                Item(label="dB"),
                                spring,
                            ),
                            label="Model",
                            show_border=True,
                            enabled_when="use_ctle_file == False",
                        ),
                        label="CTLE",
                        show_border=True,
                        enabled_when="rx_use_ami == False",
                    ),
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(
                                    name="delta_t",
                                    label="Delta-t",
                                    tooltip="magnitude of CDR proportional branch",
                                ),
                                Item(label="ps"),
                            ),
                            Item(name="alpha", label="Alpha", tooltip="relative magnitude of CDR integral branch"),
                            Item(
                                name="n_lock_ave",
                                label="Lock Nave.",
                                tooltip="# of UI estimates to average, when determining lock",
                            ),
                            Item(
                                name="rel_lock_tol",
                                label="Lock Tol.",
                                tooltip="relative tolerance for determining lock",
                            ),
                            Item(
                                name="lock_sustain",
                                label="Lock Sus.",
                                tooltip="length of lock determining hysteresis vector",
                            ),
                            label="CDR",
                            show_border=True,
                        ),
                        VGroup(
                            Item(label="Use Optimizer tab to configure."),
                            VGroup(
                                Item(name="gain", label="Gain", tooltip="error feedback gain"),
                                Item(name="n_ave", label="Nave.", tooltip="# of CDR adaptations per DFE adaptation"),
                                HGroup(
                                    Item(name="decision_scaler", label="Level", format_str="%0.3f",
                                         tooltip="target output magnitude"),
                                    Item(label="V"),
                                    Item(name="use_agc", label="Use AGC", tooltip="Continuously adjust `Level` automatically."),
                                ),
                                HGroup(
                                    Item(
                                        name="sum_bw",
                                        label="Bandwidth",
                                        tooltip="summing node bandwidth",
                                        enabled_when="sum_ideal == False",
                                    ),
                                    Item(label="GHz"),
                                    Item(
                                        name="sum_ideal",
                                        label="Ideal",
                                        tooltip="Use ideal DFE. (performance boost)",
                                    ),
                                ),
                            ),
                            label="DFE",
                            show_border=True,
                        ),
                    ),
                    label="Native",
                    show_border=True,
                    enabled_when="rx_use_ami == False",
                ),
                label="Rx Equalization",
                show_border=True,
            ),
            label="Equalization",
            id="equalization",
        ),
        # "Optimizer" tab.
        VGroup(
            HGroup(  # EQ Config.
                Group(  # Tx FFE Config.
                    Item(
                        name="tx_tap_tuners",
                        editor=TableEditor(
                            columns=[
                                ObjectColumn(name="name", editable=False),
                                ObjectColumn(name="enabled"),
                                ObjectColumn(name="min_val"),
                                ObjectColumn(name="max_val"),
                                ObjectColumn(name="step"),
                                ObjectColumn(name="value", format="%+05.3f"),
                            ],
                            configurable=False,
                            reorderable=False,
                            sortable=False,
                            selection_mode="cell",
                            auto_size=False,
                            rows=6,
                            orientation="horizontal",
                            is_grid_cell=True,
                        ),
                        show_label=False,
                    ),
                    label="Tx FFE",
                    show_border=True,
                ),
                VGroup(  # Rx CTLE
                    Item(name="ctle_enable_tune", label="Enable", tooltip="CTLE enable",),
                    VGroup(
                        HGroup(
                            Item(name="peak_freq_tune", label="fp", tooltip="CTLE peaking frequency (GHz)",
                                 enabled_when="ctle_enable_tune",),
                            Item(label="GHz"),
                        ),
                        HGroup(
                            Item(name="rx_bw_tune", label="BW", tooltip="unequalized signal path bandwidth (GHz).",
                                 enabled_when="ctle_enable_tune",),
                            Item(label="GHz"),
                        ),
                        HGroup(
                            Item(name="min_mag_tune", label="Min.", tooltip="CTLE peaking magnitude minimum (dB)",
                                 format_str="%4.1f", enabled_when="ctle_enable_tune",),
                            Item(label="dB"),
                        ),
                        HGroup(
                            Item(name="max_mag_tune", label="Max.", tooltip="CTLE peaking magnitude maximum (dB)",
                                 format_str="%4.1f", enabled_when="ctle_enable_tune",),
                            Item(label="dB"),
                        ),
                        HGroup(
                            Item(name="step_mag_tune", label="Step", tooltip="CTLE peaking magnitude step (dB)",
                                 format_str="%4.1f", enabled_when="ctle_enable_tune",),
                            Item(label="dB"),
                        ),
                        label="Configuration",
                        show_border=True,
                        enabled_when="ctle_enable_tune",
                    ),
                    HGroup(
                        Item(name="peak_mag_tune", label="Boost", tooltip="CTLE peaking magnitude result (dB)",
                             format_str="%4.1f", style="readonly"),
                        Item(label="dB"),
                        label="Result",
                        show_border=True,
                        enabled_when="ctle_enable_tune",
                    ),
                    label="Rx CTLE",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item(name="btn_disable", show_label=False, tooltip="Disable all DFE taps."),
                        Item(name="btn_enable",  show_label=False, tooltip="Enable all DFE taps."),
                    ),
                    Item(
                        name="dfe_tap_tuners",
                        editor=TableEditor(
                            columns=[
                                ObjectColumn(name="name", editable=False),
                                ObjectColumn(name="enabled"),
                                ObjectColumn(name="min_val"),
                                ObjectColumn(name="max_val"),
                                ObjectColumn(name="value", format="%+05.3f", style="readonly"),
                            ],
                            configurable=False,
                            reorderable=False,
                            sortable=False,
                            selection_mode="cell",
                            auto_size=True,
                            rows=6,
                            orientation="horizontal",
                            is_grid_cell=True,
                        ),
                        show_label=False,
                    ),
                    label="Rx DFE",
                    show_border=True,
                ),
            ),
            Item(
                label="To zoom: Click in the plot, hit `z` (Cursor will change to crosshair.), and click/drag to select region of interest. Hit <ESC> to exit zoom.",
            ),
            Item("plot_h_tune", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False, springy=True),
            label="Optimizer",
            id="eq_tune",
        ),
        Group(  # Responses
            Group(
                Item("plots_h", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Impulses",
                id="plots_h",
            ),
            Group(
                Item("plots_s", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Steps",
                id="plots_s",
            ),
            Group(
                Item("plots_p", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Pulses",
                id="plots_p",
            ),
            Group(
                Item("plots_H", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Freq. Resp.",
                id="plots_H",
            ),
            layout="tabbed",
            label="Responses",
            id="responses",
        ),
        Group(  # Results
            Group(
                Item("plots_dfe", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="DFE",
                id="plots_dfe",
            ),
            Group(
                Item("plots_out", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Outputs",
                id="plots_out",
            ),
            Group(
                Item("plots_eye", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Eyes",
                id="plots_eye",
            ),
            Group(
                Item("plots_bathtub", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Bathtubs",
                id="plots_bathtub",
            ),
            layout="tabbed",
            label="Results",
            id="results",
        ),
        Group(  # Jitter
            Group(
                Item("plots_jitter_dist", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Jitter Dist.",
                id="plots_jitter_dist",
            ),
            Group(
                Item("plots_jitter_spec", editor=ComponentEditor(high_resolution=HIGH_RES), show_label=False),
                label="Jitter Spec.",
                id="plots_jitter_spec",
            ),
            Group(Item("jitter_info", style="readonly", show_label=False), label="Jitter Info"),
            layout="tabbed",
            label="Jitter",
            id="jitter",
        ),
        Group(  # Info
            Group(
                Item("perf_info", style="readonly", show_label=False),
                label="Performance",
            ),
            Group(Item("instructions", style="readonly", show_label=False), label="User's Guide"),
            Group(Item("console_log", style="custom", show_label=False), label="Console", id="console"),
            layout="tabbed",
            label="Info",
            id="info",
        ),
        layout="tabbed",
        springy=True,
        id="tabs",
    ),
    menubar=MenuBar(
        Menu(
            Action(name="&Quit", action="close_app", accelerator="Ctrl+Q"),  # CloseAction()
            Separator(),
            Action(name="Load Results", action="do_load_data"),
            Action(name="Save Results", action="do_save_data"),
            Separator(),
            Action(name="Load Config.", action="do_load_cfg", accelerator="Ctrl+O"),
            Action(name="Save Config.", action="do_save_cfg", accelerator="Ctrl+S"),
            Action(name="Save Config. As...", action="do_save_cfg_as", accelerator="Ctrl+Shift+S"),
            id="file",
            name="&File",
        ),
        Menu(
            Action(
                name="Debug Mode",
                action="toggle_debug_clicked",
                accelerator="Ctrl+`",
                style="toggle",
                checked_when="debug == True",
            ),
            Action(
                name="Clear Loaded Waveforms",
                action="do_clear_data",
            ),
            id="view",
            name="&View",
        ),
        Menu(
            Action(name="Run", action="do_run_simulation", accelerator="Ctrl+R"),
            Action(name="Abort", action="do_stop_simulation"),
            id="simulation",
            name="Simulation",
        ),
        Menu(
            Action(name="Use EQ", action="do_use_eq", accelerator="Ctrl+U"),
            Action(name="Reset EQ", action="do_reset_eq"),
            Separator(),
            Action(name="Tune EQ", action="do_tune_eq", accelerator="Ctrl+T"),
            Action(name="Abort", action="do_stop_tune", accelerator="Ctrl+Esc"),
            id="optimization",
            name="Optimization",
        ),
        Menu(
            Action(name="Getting Started", action="getting_started_clicked"),
            Action(name="&About", action="show_about_clicked"),
            id="help",
            name="&Help",
        ),
    ),
    buttons=NoButtons,
    handler=MyHandler(),
    icon=ImageResource("icon.png"),
    resizable=True,
    statusbar="status_str",
    title="PyBERT",
)
