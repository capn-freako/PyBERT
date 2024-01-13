"""Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""

from enable.component_editor import ComponentEditor
from numpy                   import log10
from pyface.image_resource   import ImageResource
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
from traitsui.ui_editors.array_view_editor import ArrayViewEditor
from pybert.gui.handler import MyHandler

# Main window layout definition.
traits_view = View(
    Group(  # Members correspond to top-level tabs.
        VGroup(  # "Config." tab
            HGroup(
                VGroup(
                    HGroup(  # Simulation Control
                        VGroup(
                            Item(
                                name="bit_rate",
                                label="Bit Rate (Gbps)",
                                tooltip="bit rate",
                                show_label=True,
                                enabled_when="True",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                            ),
                            Item(
                                name="nbits",
                                label="Nbits",
                                tooltip="# of bits to run",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="nspb",
                                label="Nspb",
                                tooltip="# of samples per bit",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                            Item(
                                name="mod_type",
                                label="Modulation",
                                tooltip="line signalling/modulation scheme",
                                editor=CheckListEditor(values=[(0, "NRZ"), (1, "Duo-binary"), (2, "PAM-4")]),
                            ),
                        ),
                        VGroup(
                            Item(name="do_sweep", label="Do Sweep", tooltip="Run parameter sweeps."),
                            Item(
                                name="sweep_aves",
                                label="SweepAves",
                                tooltip="# of trials, per sweep, for averaging.",
                                enabled_when="do_sweep == True",
                            ),
                            HGroup(
                                Item(
                                    name="pattern",
                                    label="Pattern",
                                    tooltip="pattern to use to construct bit stream",
                                ),
                                Item(
                                    name="seed",
                                    label="Seed",
                                    tooltip="LFSR seed. 0 means new random seed for each run.",
                                ),
                            ),
                            Item(
                                name="eye_bits",
                                label="EyeBits",
                                tooltip="# of bits to use to form eye diagrams",
                                editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
                            ),
                        ),
                        VGroup(
                            Item(name="vod", label="Vod (V)", tooltip="Tx output voltage into matched load"),
                            Item(name="rn", label="Rn (V)", tooltip="standard deviation of random noise"),
                            Item(name="pn_mag", label="Pn (V)", tooltip="peak magnitude of periodic noise"),
                            Item(name="pn_freq", label="f(Pn) (MHz)", tooltip="frequency of periodic noise"),
                        ),
                    ),
                    label="Simulation Control",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="impulse_length",
                        label="Impulse Response Length (ns)",
                        tooltip="Manual impulse response length override",
                    ),
                    Item(
                        name="thresh",
                        label="Pj Threshold (sigma)",
                        tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)",
                    ),
                    Item(name="debug", label="Debug", tooltip="Enable to log extra information to console."),
                    label="Analysis Parameters",
                    show_border=True,
                ),
            ),
            HGroup(  # Channel
                VGroup(  # Tx
                    VGroup(
                        Item(
                            name="tx_ibis_file",
                            label="File",
                            springy=True,
                            editor=FileEditor(dialog_style="open", filter=["*.ibs"]),
                        ),
                        Item(name="tx_ibis_valid", label="Valid", style="simple", enabled_when="False"),
                        HGroup(
                            Item(name="tx_use_ibis", label="Use IBIS"),
                            Item(name="btn_sel_tx", show_label=False),
                            Item(name="btn_view_tx", show_label=False),
                            spring,
                            enabled_when="tx_ibis_valid == True",
                        ),
                        Item(
                            name="tx_use_ts4",
                            label="Use on-die S-parameters.",
                            enabled_when="tx_ibis_valid == True and tx_use_ibis and tx_has_ts4",
                        ),
                        spring,
                        label="IBIS",
                        show_border=True,
                    ),
                    VGroup(
                        Item(
                            name="rs",
                            label="Tx_Rs (Ohms)",
                            tooltip="Tx differential source impedance",
                        ),
                        Item(
                            name="cout",
                            label="Tx_Cout (pF)",
                            tooltip="Tx parasitic output capacitance (each pin)",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="tx_use_ibis == False",
                    ),
                    spring,
                    label="Tx",
                    show_border=True,
                ),
                VGroup(  # Interconnect
                    VGroup(  # From File
                        VGroup(
                            # HGroup(
                                Item(
                                    name="ch_file",
                                    label="File",
                                    editor=FileEditor(dialog_style="open"),
                                ),
                            # ),
                            HGroup(
                                Item(
                                    name="use_ch_file",
                                    label="Use file",
                                ),
                                Item(
                                    name="do_xtalk",
                                    label="Include crosstalk",
                                    enabled_when="use_ch_file == True and ch_is_s32p",
                                ),
                                spring,
                            ),
                        ),
                        HGroup(
                            Item(
                                name="f_step",
                                label="f_step",
                                tooltip="Frequency step to use in generating H(f).",
                            ),
                            Item(label="MHz"),
                            enabled_when="use_ch_file == True",
                        ),
                        label="From File",
                        show_border=True,
                    ),
                    VGroup(  # Native (i.e. - Howard Johnson's) interconnect model.
                        Item(
                            name="l_ch",
                            label="Length (m)",
                            tooltip="interconnect length",
                        ),
                        Item(
                            name="Theta0",
                            label="Loss Tan.",
                            tooltip="dielectric loss tangent",
                        ),
                        Item(
                            name="Z0",
                            label="Z0 (Ohms)",
                            tooltip="characteristic differential impedance",
                        ),
                        Item(
                            name="v0",
                            label="v_rel (c)",
                            tooltip="normalized propagation velocity",
                        ),
                        Item(
                            name="Rdc",
                            label="Rdc (Ohms)",
                            tooltip="d.c. resistance",
                        ),
                        Item(
                            name="w0",
                            label="w0 (rads./s)",
                            tooltip="transition frequency",
                        ),
                        Item(
                            name="R0",
                            label="R0 (Ohms)",
                            tooltip="skin effect resistance",
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="use_ch_file == False",
                        layout="normal",
                    ),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(
                    VGroup(
                        Item(
                            name="rx_ibis_file",
                            label="File",
                            springy=True,
                            editor=FileEditor(dialog_style="open", filter=["*.ibs"]),
                        ),
                        Item(name="rx_ibis_valid", label="Valid", style="simple", enabled_when="False"),
                        HGroup(
                            Item(name="rx_use_ibis", label="Use IBIS"),
                            Item(name="btn_sel_rx", show_label=False),
                            Item(name="btn_view_rx", show_label=False),
                            spring,
                            enabled_when="rx_ibis_valid == True",
                        ),
                        Item(
                            name="rx_use_ts4",
                            label="Use on-die S-parameters.",
                            enabled_when="rx_ibis_valid == True and rx_use_ibis and rx_has_ts4",
                        ),
                        spring,
                        label="IBIS",
                        show_border=True,
                    ),
                    VGroup(
                        Item(
                            name="rin",
                            label="Rx_Rin (Ohms)",
                            tooltip="Rx differential input impedance",
                        ),
                        Item(
                            name="cin",
                            label="Rx_Cin (pF)",
                            tooltip="Rx parasitic input capacitance (each pin)",
                            editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(
                            name="cac",
                            label="Rx_Cac (uF)",
                            tooltip="Rx a.c. coupling capacitance (each pin)",
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="rx_use_ibis == False",
                    ),
                    spring,
                    label="Rx",
                    show_border=True,
                ),
                label="Channel",
                show_border=True,
            ),
            # spring,
            label="Config.",
            id="config",
        ),
        # "Equalization" tab.
        VGroup(
            HGroup(
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
                                    ObjectColumn(name="min_val", horizontal_alignment="center"),
                                    ObjectColumn(name="max_val", horizontal_alignment="center"),
                                    ObjectColumn(name="value", format="%+05.3f", horizontal_alignment="center"),
                                    ObjectColumn(name="steps", horizontal_alignment="center"),
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
                        VGroup(
                            VGroup(
                                HGroup(
                                    Item(
                                        name="use_ctle_file",
                                        label="fromFile",
                                        tooltip="Select CTLE impulse/step response from file.",
                                    ),
                                    Item(
                                        name="ctle_file",
                                        label="Filename",
                                        enabled_when="use_ctle_file == True",
                                        editor=FileEditor(dialog_style="open"),
                                    ),
                                ),
                                HGroup(
                                    Item(
                                        name="peak_freq",
                                        label="CTLE fp (GHz)",
                                        tooltip="CTLE peaking frequency (GHz)",
                                        enabled_when="use_ctle_file == False",
                                    ),
                                    Item(
                                        name="rx_bw",
                                        label="Bandwidth (GHz)",
                                        tooltip="unequalized signal path bandwidth (GHz).",
                                        enabled_when="use_ctle_file == False",
                                    ),
                                ),
                                HGroup(
                                    Item(
                                        name="peak_mag",
                                        label="CTLE boost (dB)",
                                        tooltip="CTLE peaking magnitude (dB)",
                                        format_str="%4.1f",
                                        enabled_when="use_ctle_file == False",
                                    ),
                                    Item(
                                        name="ctle_mode",
                                        label="CTLE mode",
                                        tooltip="CTLE Operating Mode",
                                        enabled_when="use_ctle_file == False",
                                    ),
                                    Item(
                                        name="ctle_offset",
                                        tooltip="CTLE d.c. offset (dB)",
                                        show_label=False,
                                        enabled_when='ctle_mode == "Manual"',
                                    ),
                                ),
                                label="CTLE",
                                show_border=True,
                                enabled_when="rx_use_ami == False",
                            ),
                        ),
                        HGroup(
                            VGroup(
                                Item(
                                    name="delta_t",
                                    label="Delta-t (ps)",
                                    tooltip="magnitude of CDR proportional branch",
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
                                HGroup(
                                    Item(
                                        name="use_dfe",
                                        label="Use DFE",
                                        tooltip="Include DFE in simulation.",
                                    ),
                                    Item(
                                        name="sum_ideal",
                                        label="Ideal",
                                        tooltip="Use ideal DFE. (performance boost)",
                                        enabled_when="use_dfe == True",
                                    ),
                                ),
                                VGroup(
                                    Item(name="n_taps", label="Taps", tooltip="# of taps"),
                                    Item(name="gain", label="Gain", tooltip="error feedback gain"),
                                    Item(name="decision_scaler", label="Level", tooltip="target output magnitude"),
                                    Item(
                                        name="n_ave", label="Nave.", tooltip="# of CDR adaptations per DFE adaptation"
                                    ),
                                    Item(
                                        name="sum_bw",
                                        label="BW (GHz)",
                                        tooltip="summing node bandwidth",
                                        enabled_when="sum_ideal == False",
                                    ),
                                    enabled_when="use_dfe == True",
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
                springy=True,
            ),
            HGroup(),
            label="Equalization",
            id="channel",
        ),
        # "Optimizer" tab.
        VGroup(
            HGroup(
                Group(
                    Item(
                        name="tx_tap_tuners",
                        editor=TableEditor(
                            columns=[
                                ObjectColumn(name="name", editable=False),
                                ObjectColumn(name="enabled"),
                                ObjectColumn(name="min_val"),
                                ObjectColumn(name="max_val"),
                                ObjectColumn(name="value", format="%+05.3f"),
                            ],
                            configurable=False,
                            reorderable=False,
                            sortable=False,
                            selection_mode="cell",
                            auto_size=False,
                            rows=4,
                            orientation="horizontal",
                            is_grid_cell=True,
                        ),
                        show_label=False,
                    ),
                    label="Tx Equalization",
                    show_border=True,
                    springy=True,
                ),
                # HGroup(
                VGroup(
                    HGroup(
                        Item(
                            name="peak_mag_tune",
                            label="CTLE: boost (dB)",
                            tooltip="CTLE peaking magnitude (dB)",
                            format_str="%4.1f",
                        ),
                        Item(
                            name="max_mag_tune",
                            label="Max boost (dB)",
                            tooltip="CTLE maximum peaking magnitude (dB)",
                            format_str="%4.1f",
                        ),
                    ),
                    HGroup(
                        Item(name="peak_freq_tune", label="fp (GHz)", tooltip="CTLE peaking frequency (GHz)"),
                        Item(
                            name="rx_bw_tune",
                            label="BW (GHz)",
                            tooltip="unequalized signal path bandwidth (GHz).",
                        ),
                    ),
                    HGroup(
                        Item(name="ctle_mode_tune", label="mode", tooltip="CTLE Operating Mode"),
                        Item(
                            name="ctle_offset_tune",
                            tooltip="CTLE d.c. offset (dB)",
                            show_label=False,
                            enabled_when='ctle_mode_tune == "Manual"',
                        ),
                    ),
                    HGroup(
                        Item(name="use_dfe_tune", label="DFE: Enable", tooltip="Include ideal DFE in optimization."),
                        Item(name="n_taps_tune", label="Taps", tooltip="Number of DFE taps."),
                    ),
                    label="Rx Equalization",
                    show_border=True,
                ),
                # ),
                VGroup(
                    Item(
                        name="max_iter",
                        label="Max. Iterations",
                        tooltip="Maximum number of iterations to allow, during optimization.",
                    ),
                    Item(
                        name="rel_opt",
                        label="Rel. Opt.:",
                        format_str="%7.4f",
                        tooltip="Relative optimization metric.",
                        style="readonly",
                    ),
                    Item(
                        name="przf_err",
                        label="PRZF Err.:",
                        format_str="%5.3f",
                        tooltip="Pulse Response Zero Forcing approximation error.",
                        style="readonly",
                    ),
                    label="Tuning Options",
                    show_border=True,
                ),
                springy=False,
            ),
            Item(
                label="Note: Only CTLE boost will be optimized; please, set peak frequency, bandwidth, and mode appropriately.",
            ),
            Item("plot_h_tune", editor=ComponentEditor(high_resolution=False), show_label=False, springy=True),
            HGroup(
                Item("btn_rst_eq", show_label=False, tooltip="Reset all values to those on the 'Config.' tab."),
                Item("btn_save_eq", show_label=False, tooltip="Store all values to 'Config.' tab."),
                Item("btn_opt_tx", show_label=False, tooltip="Run Tx tap weight optimization."),
                Item("btn_opt_rx", show_label=False, tooltip="Run Rx CTLE optimization."),
                Item("btn_coopt", show_label=False, tooltip="Run co-optimization."),
                Item("btn_abort", show_label=False, tooltip="Abort all optimizations."),
            ),
            label="Optimizer",
            id="eq_tune",
        ),
        # "COM" tab.
        Group(
            HGroup(
                VGroup(  # COM parameters
                    Item("standard", label="Standard"),
                    HGroup(
                        Item("com_ser",     label="DER0",  format_str="%7.1e",
                            tooltip="Symbol error rate for COM calculation."),
                        Item("com_nTx",     label="N FFE",
                            tooltip="Number of Tx FFE filter taps."),
                        Item("com_nDFE",     label="N DFE",
                            tooltip="Number of Rx DFE filter taps."),
                    ),
                    HGroup(
                        Item("com_Add",     label="Add (UI)",      format_str="%5.3f",
                            tooltip="Deterministic jitter induced voltage noise amplitude (UI)."),
                        Item("com_sigRj",   label="Sigma_Rj (UI)", format_str="%5.3f",
                            tooltip="Standard deviation of random noise."),
                    ),
                    HGroup(
                        Item("com_TxSNR",   label="Tx SNR (dB)",   format_str="%5.1f",
                            tooltip="Variance of Tx voltage noise (dB)."),
                        Item("com_eta0",    label="Eta0 (V^2/GHz)", format_str="%9.3e",
                            tooltip="One sided noise spectral density (V^2/GHz)."),
                    ),
                    HGroup(
                        Item("com_z",       label="Z",  format_str="%5.2f",
                            tooltip="Zero of CTLE response (GHz)."),
                        Item("com_p1",      label="P1", format_str="%5.2f",
                            tooltip="First pole of CTLE response (GHz)."),
                        Item("com_p2",      label="P2", format_str="%5.2f",
                            tooltip="Second pole of CTLE response (GHz)."),
                        Item("com_fHP",     label="fHP", format_str="%5.2f",
                            tooltip="2nd stage pole/zero (GHz)."),
                    ),
                    HGroup(
                        Item("com_gDC_min", label="gDC min.",  format_str="%5.1f",
                            tooltip="Minimum d.c. gain of CTLE 1st stage (dB)."),
                        Item("com_gDC_max", label="gDC max.",  format_str="%5.1f",
                            tooltip="Maximum d.c. gain of CTLE 1st stage (dB)."),
                        Item("com_gHP_min", label="gHP min.",  format_str="%5.1f",
                            tooltip="Minimum d.c. gain of CTLE 2nd stage (dB)."),
                        Item("com_gHP_max", label="gHP max.",  format_str="%5.1f",
                            tooltip="Maximum d.c. gain of CTLE 2nd stage (dB)."),
                    ),
                    Item("com_tx_min",  label="FFE min.",
                        tooltip="Minimum values of Tx FFE filter taps."),
                    Item("com_tx_max",  label="FFE max.",
                        tooltip="Maximum values of Tx FFE filter taps."),
                    HGroup(
                        Item("com_dfe_min",  label="DFE min.",
                            editor=ArrayViewEditor(transpose=True, show_index=False),
                            tooltip="Minimum values of Rx DFE filter taps."),
                        Item("com_dfe_max",  label="DFE max.",
                            editor=ArrayViewEditor(transpose=True, show_index=False),
                            tooltip="Maximum values of Rx DFE filter taps."),
                    ),
                    Item("btn_com", show_label=False, tooltip="Calculate COM."),
                    label="Input Parameters",
                    show_border=True,
                    id="com_params",
                ),
                VGroup(  # COM result
                    Item("com",           label="COM (dB)",           format_str="%4.1f"),
                    Item("com_tx_taps",   label="Tx Taps",            format_str="%5.3f"),
                    HGroup(
                        Item("com_ctle_gain", label="CTLE Gain 1st (dB)", format_func=(lambda x: f"{-20*log10(x):6.3f}")),
                        Item("com_hp_gain",   label="CTLE Gain 2nd (dB)", format_func=(lambda x: f"{-20*log10(x):6.3f}")),
                    ),
                    Item("com_dfe_taps",  label="DFE Taps",           format_str="%5.3f",
                        editor=ArrayViewEditor(transpose=True, show_index=False),),
                    label="Results",
                    show_border=True,
                    id="com_results",
                ),
                label="Data",
                id="com_data",
            ),
            HGroup(
                Item("plots_com", editor=ComponentEditor(high_resolution=False), show_label=False,
                        resizable=True, springy=True),
                label="Plots",
                id="com_plots",
            ),
            layout="tabbed",
            label="COM",
            id="com",
        ),
        Group(  # Responses
            Group(
                Item("plots_h", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Impulses",
                id="plots_h",
            ),
            Group(
                Item("plots_s", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Steps",
                id="plots_s",
            ),
            Group(
                Item("plots_p", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Pulses",
                id="plots_p",
            ),
            Group(
                Item("plots_H", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Freq. Resp.",
                id="plots_H",
            ),
            layout="tabbed",
            label="Responses",
            id="responses",
        ),
        Group(  # Results
            Group(
                Item("plots_dfe", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="DFE",
                id="plots_dfe",
            ),
            Group(
                Item("plots_out", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Outputs",
                id="plots_out",
            ),
            Group(
                Item("plots_eye", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Eyes",
                id="plots_eye",
            ),
            Group(
                Item("plots_bathtub", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Bathtubs",
                id="plots_bathtub",
            ),
            Group(Item("sweep_info", style="readonly", show_label=False), label="Sweep Info"),
            layout="tabbed",
            label="Results",
            id="results",
        ),
        Group(  # Jitter
            Group(
                Item("plots_jitter_dist", editor=ComponentEditor(high_resolution=False), show_label=False),
                label="Jitter Dist.",
                id="plots_jitter_dist",
            ),
            Group(
                Item("plots_jitter_spec", editor=ComponentEditor(high_resolution=False), show_label=False),
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
            Group(Item("console_log", style="readonly", show_label=False), label="Console", id="console"),
            layout="tabbed",
            label="Info",
            id="info",
        ),
        layout="tabbed",
        springy=True,
        id="tabs",
    ),
    resizable=True,
    handler=MyHandler(),
    menubar=MenuBar(
        Menu(
            Action(name="Load Config.", action="do_load_cfg", accelerator="Ctrl+O"),
            Action(name="Load Results", action="do_load_data"),
            Separator(),
            Action(name="Save Config.", action="do_save_cfg", accelerator="Ctrl+S"),
            Action(name="Save Config. As...", action="do_save_cfg_as", accelerator="Ctrl+Shift+S"),
            Action(name="Save Results", action="do_save_data"),
            Separator(),
            Action(name="&Quit", action="close_app", accelerator="Ctrl+Q"),  # CloseAction()
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
            Action(name="Getting Started", action="getting_started_clicked"),
            Action(name="&About", action="show_about_clicked"),
            id="help",
            name="&Help",
        ),
    ),
    buttons=NoButtons,
    statusbar="status_str",
    title="PyBERT",
    icon=ImageResource("icon.png"),
)
