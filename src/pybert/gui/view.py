"""Default view definition for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 24, 2014 (Copied from pybert.py, as part of a major code cleanup.)

Copyright (c) 2014 David Banas; all rights reserved World wide.
"""
import sys
import webbrowser
from pathlib import Path
from threading import Thread

from pyface.api import OK, FileDialog, MessageDialog
from pyface.image_resource import ImageResource
from traits.api import Instance
from traitsui.api import (  # CloseAction,
    Action,
    CheckListEditor,
    FileEditor,
    Group,
    Handler,
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

from enable.component_editor import ComponentEditor
from pybert import __authors__, __copy__, __date__, __version__
from pybert.configuration import CONFIG_LOAD_WILDCARD, CONFIG_SAVE_WILDCARD
from pybert.models.bert import my_run_sweeps
from pybert.results import RESULTS_FILEDIALOG_WILDCARD


class RunSimThread(Thread):
    """Used to run the simulation in its own thread, in order to preserve GUI
    responsiveness."""

    def run(self):
        """Run the simulation(s)."""
        my_run_sweeps(self.the_pybert)


class MyHandler(Handler):
    """This handler is instantiated by the View and handles user button
    clicks."""

    run_sim_thread = Instance(RunSimThread)

    def do_run_simulation(self, info):
        """Spawn a simulation thread and run with the current settings."""
        the_pybert = info.object
        if self.run_sim_thread and self.run_sim_thread.is_alive():
            pass
        else:
            self.run_sim_thread = RunSimThread()
            self.run_sim_thread.the_pybert = the_pybert
            self.run_sim_thread.start()

    def do_stop_simulation(self):
        """Kill the simulation thread."""
        if self.run_sim_thread and self.run_sim_thread.is_alive():
            self.run_sim_thread.stop()

    def do_save_cfg(self, info):
        """Save the configuration.

        If no config file is set, use the `self.do_save_cfg_as()` method to prompt the user.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        configuration_file = Path(pybert.cfg_file)
        if pybert.cfg_file and configuration_file.exists():
            pybert.save_configuration(configuration_file)
        else:  # A configuration file hasn't been set yet so use the save-as method
            self.do_save_cfg_as(info)

    def do_save_cfg_as(self, info):
        """Prompt the user to choose where to save the config and save it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(
            action="save as",
            wildcard=CONFIG_SAVE_WILDCARD,
            default_path=pybert.cfg_file,
        )
        if dialog.open() == OK:
            pybert.save_configuration(Path(dialog.path))

    def do_load_cfg(self, info):
        """Prompt the user to choose where to load the config from and load it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(
            action="open",
            wildcard=CONFIG_LOAD_WILDCARD,
            default_path=pybert.cfg_file,
        )
        if dialog.open() == OK:
            pybert.load_configuration(Path(dialog.path))

    def do_save_data(self, info):
        """Prompt the user to choose where to save the results and save it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(action="save as", wildcard=RESULTS_FILEDIALOG_WILDCARD, default_path=pybert.data_file)
        if dialog.open() == OK:
            pybert.save_results(Path(dialog.path))

    def do_load_data(self, info):
        """Prompt the user to choose where to load the results from and load it.

        Pybert will load these as "reference" plots for the responses.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        dialog = FileDialog(action="open", wildcard=RESULTS_FILEDIALOG_WILDCARD, default_path=pybert.data_file)
        if dialog.open() == OK:
            pybert.load_results(Path(dialog.path))

    def do_clear_data(self, info):
        """If any reference or prior results has been loaded, clear it.

        Args:
            info: When an action is clicked, it passes the whole trait instance to this function.
        """
        # pylint: disable=no-self-use
        pybert = info.object
        pybert.clear_reference_from_plots()

    def toggle_debug_clicked(self, info):
        """Toggle whether debug mode is enabled or not."""
        # pylint: disable=no-self-use
        info.object.debug = not info.object.debug

    def getting_started_clicked(self, info):
        """Open up Pybert's wiki."""
        # pylint: disable=no-self-use,unused-argument
        webbrowser.open("https://github.com/capn-freako/PyBERT/wiki")

    def show_about_clicked(self, info):
        """Open a dialog and show the user the about info."""
        # pylint: disable=no-self-use,unused-argument
        dialog = MessageDialog(
            title="About Pybert",
            message=f"PyBERT v{__version__} - a serial communication link design tool, written in Python.",
            informative=(
                f"{__authors__}<BR>\n" f"{__date__}<BR><BR>\n\n" f"{__copy__};<BR>\n" "All rights reserved World wide."
            ),
            severity="information",
        )
        dialog.open()

    def close_app(self, info):
        """Close the app.

        Can't use CloseAction until https://github.com/enthought/traitsui/issues/1442 is resolved.
        """
        # pylint: disable=no-self-use,unused-argument
        sys.exit(0)


# Main window layout definition.
traits_view = View(
    Group(
        VGroup(
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
                                    # editor=TextEditor(auto_set=False, enter_set=True, evaluate=int),
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
                        name="thresh",
                        label="Pj Threshold (sigma)",
                        tooltip="Threshold for identifying periodic jitter spectral elements. (sigma)",
                    ),
                    Item(
                        name="impulse_length",
                        label="Impulse Response Length (ns)",
                        tooltip="Manual impulse response length override",
                    ),
                    Item(name="debug", label="Debug", tooltip="Enable to log extra information to console."),
                    label="Analysis Parameters",
                    show_border=True,
                ),
            ),
            HGroup(
                VGroup(
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
                            Item(
                                name="tx_use_ts4",
                                label="Use on-die S-parameters.",
                                enabled_when="tx_use_ibis and tx_has_ts4",
                            ),
                            enabled_when="tx_ibis_valid == True",
                        ),
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
                    label="Tx",
                    show_border=True,
                ),
                VGroup(  # Interconnect
                    VGroup(  # From File
                        VGroup(
                            HGroup(
                                Item(
                                    name="ch_file",
                                    label="File",
                                    springy=True,
                                    editor=FileEditor(dialog_style="open"),
                                ),
                            ),
                            HGroup(
                                Item(
                                    name="use_ch_file",
                                    label="Use file",
                                ),
                                spring,
                                # Item(name="padded",   label="Zero-padded", enabled_when="use_ch_file == True"),
                                # Item(name="windowed", label="Windowed",    enabled_when="use_ch_file == True"),
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
                    HGroup(  # Native (i.e. - Howard Johnson's) interconnect model.
                        VGroup(
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
                        ),
                        VGroup(
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
                        ),
                        label="Native",
                        show_border=True,
                        enabled_when="use_ch_file == False",
                    ),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(
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
                            Item(
                                name="rx_use_ts4",
                                label="Use on-die S-parameters.",
                                enabled_when="rx_use_ibis and rx_has_ts4",
                            ),
                            enabled_when="rx_ibis_valid == True",
                        ),
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
        VGroup(  # Channel Parameters
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
                                # auto_size=True,
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
        Menu(Action(name="Run", action="do_run_simulation", accelerator="Ctrl+R"), id="simulation", name="Simulation"),
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
    # width=0.95,
    # height=0.9,
    icon=ImageResource("icon.png"),
)
