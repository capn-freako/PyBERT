#! /usr/bin/env python

# pylint: disable=too-many-lines

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.

ToDo:
    1. Add optional AFE (4th-order Bessel-Thomson).
    2. Add eye contour plots.
"""

import platform
import time
from datetime import datetime
from os.path import dirname, join
from pathlib import Path

import numpy as np  # type: ignore
import skrf as rf
from chaco.api import ArrayPlotData, GridPlotContainer
from numpy import arange, array, cos, exp, pad, pi, sinc, where, zeros
from numpy.fft import irfft, rfft  # type: ignore
from numpy.random import randint  # type: ignore
from traits.api import (
    Array,
    Bool,
    Button,
    File,
    Float,
    HasTraits,
    Instance,
    Int,
    List,
    Map,
    Property,
    Range,
    String,
    cached_property,
)
from traits.etsconfig.api import ETSConfig
from traitsui.message import message, error
from scipy.interpolate import interp1d

from pyibisami import __version__ as PyAMI_VERSION  # type: ignore
from pyibisami.ami.model import AMIModel
from pyibisami.ami.parser import AMIParamConfigurator
from pyibisami.ibis.file import IBISModel

from pybert import __version__ as VERSION
from pybert.configuration import InvalidFileType, PyBertCfg
from pybert.gui.help import help_str
from pybert.gui.plot import make_plots
from pybert.models.bert import my_run_simulation
from pybert.models.tx_tap import TxTapTuner
from pybert.results import PyBertData
from pybert.threads.optimization import OptThread
from pybert.utility import (
    calc_gamma,
    import_channel,
    lfsr_bits,
    raised_cosine,
    safe_log10,
    sdd_21,
    trim_impulse,
)

gDebugStatus = False
gUseDfe      = True     # Include DFE when running simulation.
gMaxCTLEPeak =    20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
gPeakFreq    =     5.0  # CTLE peaking frequency (GHz)
gPeakMag     =     1.7  # CTLE peaking magnitude (dB)
gCTLEOffset  =     0.0  # CTLE d.c. offset (dB)
gNtaps       =     5


class PyBERT(HasTraits):  # pylint: disable=too-many-instance-attributes
    """A serial communication link bit error rate tester (BERT) simulator with
    a GUI interface.

    Useful for exploring the concepts of serial communication link
    design.
    """

    # Independent variables

    # - Simulation Control
    bit_rate = Range(low=0.1, high=250.0, value=10.0)    #: (Gbps)
    nbits = Range(low=1000, high=10000000, value=15000)  #: Number of bits to simulate.
    eye_bits = Int(10160)                                #: Number of bits used to form eye.
    pattern = Map(
        {
            "PRBS-7": [7, 6],
            "PRBS-9": [9, 5],
            "PRBS-11": [11, 9],
            "PRBS-13": [13, 12, 2, 1],
            "PRBS-15": [15, 14],
            "PRBS-20": [20, 3],
            "PRBS-23": [23, 18],
            "PRBS-31": [31, 28],
        },
        default_value="PRBS-7",
    )
    seed = Int(1)  # LFSR seed. 0 means regenerate bits, using a new random seed, each run.
    nspui = Range(low=2, high=256, value=32)  #: Signal vector samples per unit interval.
    mod_type   = List([0])                   #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
    do_sweep   = Bool(False)  #: Run sweeps? (Default = False)
    debug      = Bool(False)  #: Send log messages to terminal, as well as console, when True. (Default = False)
    thresh     = Float(3.0)   #: Spectral threshold for identifying periodic components (sigma). (Default = 3.0)

    # - Channel Control
    ch_file = File(
        "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
    )  #: Channel file name.
    use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
    renumber = Bool(False)  #: Automically fix "1=>3/2=>4" port numbering? (Default = False)
    f_step = Float(10)  #: Frequency step to use when constructing H(f) (MHz). (Default = 10 MHz)
    f_max = Float(40)  #: Frequency maximum to use when constructing H(f) (GHz). (Default = 40 GHz)
    impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
    Rdc = Float(0.1876)  #: Channel d.c. resistance (Ohms/m).
    w0 = Float(10e6)  #: Channel transition frequency (rads./s).
    R0 = Float(1.452)  #: Channel skin effect resistance (Ohms/m).
    Theta0 = Float(0.02)  #: Channel loss tangent (unitless).
    Z0 = Float(100)  #: Channel characteristic impedance, in LC region (Ohms).
    v0 = Float(0.67)  #: Channel relative propagation velocity (c).
    l_ch = Float(0.5)  #: Channel length (m).
    use_window = Bool(False)  #: Apply raised cosine to frequency response before FFT()-ing? (Default = False)

    # - EQ Tune
    tx_tap_tuners = List(
        [
            TxTapTuner(name="Pre-tap3",  pos=-3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
            TxTapTuner(name="Pre-tap2",  pos=-2, enabled=True, min_val=-0.1,  max_val=0.1,  step=0.05),
            TxTapTuner(name="Pre-tap1",  pos=-1, enabled=True, min_val=-0.2,  max_val=0.2,  step=0.1),
            TxTapTuner(name="Post-tap1", pos=1,  enabled=True, min_val=-0.2,  max_val=0.2,  step=0.1),
            TxTapTuner(name="Post-tap2", pos=2,  enabled=True, min_val=-0.1,  max_val=0.1,  step=0.05),
            TxTapTuner(name="Post-tap3", pos=3,  enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
        ]
    )  #: EQ optimizer list of TxTapTuner objects.
    rx_bw_tune = Float(12.0)  #: EQ optimizer CTLE bandwidth (GHz).
    peak_freq_tune = Float(gPeakFreq)  #: EQ optimizer CTLE peaking freq. (GHz).
    peak_mag_tune = Float(gPeakMag)  #: EQ optimizer CTLE peaking mag. (dB).
    min_mag_tune = Float(2)   #: EQ optimizer CTLE peaking mag. min. (dB).
    max_mag_tune = Float(12)  #: EQ optimizer CTLE peaking mag. max. (dB).
    step_mag_tune = Float(1)  #: EQ optimizer CTLE peaking mag. step (dB).
    ctle_enable_tune = Bool(True)  #: EQ optimizer CTLE enable
    dfe_tap_tuners = List(
        [TxTapTuner(name="Tap1",  enabled=True,  min_val=0.1,   max_val=0.4,  value=0.1),
         TxTapTuner(name="Tap2",  enabled=True,  min_val=-0.15, max_val=0.15, value=0.0),
         TxTapTuner(name="Tap3",  enabled=True,  min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap4",  enabled=True,  min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap5",  enabled=True,  min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap6",  enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap7",  enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap8",  enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap9",  enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap10", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap11", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap12", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap13", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap14", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap15", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap16", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap17", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap18", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap19", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),
         TxTapTuner(name="Tap20", enabled=False, min_val=-0.05, max_val=0.1,  value=0.0),]
    )  #: EQ optimizer list of DFE tap tuner objects.
    opt_thread = Instance(OptThread)  #: EQ optimization thread.

    # - Tx
    vod = Float(1.0)  #: Tx differential output voltage (V)
    rs = Float(100)  #: Tx source impedance (Ohms)
    cout = Range(low=0.001, high=1000, value=0.5)  #: Tx parasitic output capacitance (pF)
    pn_mag = Float(0.01)  #: Periodic noise magnitude (V).
    pn_freq = Float(11)  #: Periodic noise frequency (MHz).
    rn = Float(0.01)  #: Standard deviation of Gaussian random noise (V).
    tx_taps = List(
        [
            TxTapTuner(name="Pre-tap3",  pos=-3, enabled=True, min_val=-0.05, max_val=0.05),
            TxTapTuner(name="Pre-tap2",  pos=-2, enabled=True, min_val=-0.1,  max_val=0.1),
            TxTapTuner(name="Pre-tap1",  pos=-1, enabled=True, min_val=-0.2,  max_val=0.2),
            TxTapTuner(name="Post-tap1", pos=1,  enabled=True, min_val=-0.2,  max_val=0.2),
            TxTapTuner(name="Post-tap2", pos=2,  enabled=True, min_val=-0.1,  max_val=0.1),
            TxTapTuner(name="Post-tap3", pos=3,  enabled=True, min_val=-0.05, max_val=0.05),
        ]
    )  #: List of TxTapTuner objects.
    rel_power = Float(1.0)  #: Tx power dissipation (W).
    tx_use_ami = Bool(False)  #: (Bool)
    tx_has_ts4 = Bool(False)  #: (Bool)
    tx_use_ts4 = Bool(False)  #: (Bool)
    tx_use_getwave = Bool(False)  #: (Bool)
    tx_has_getwave = Bool(False)  #: (Bool)
    tx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    tx_ami_valid = Bool(False)  #: (Bool)
    tx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    tx_dll_valid = Bool(False)  #: (Bool)
    tx_ibis_file = File(
        "",
        entries=5,
        filter=[
            "IBIS Models (*.ibs)|*.ibs",
        ],
    )  #: (File)
    tx_ibis_valid = Bool(False)  #: (Bool)
    tx_use_ibis = Bool(False)  #: (Bool)

    # - Rx
    rin = Float(100)  #: Rx input impedance (Ohm)
    cin = Float(0.5)  #: Rx parasitic input capacitance (pF)
    cac = Float(1.0)  #: Rx a.c. coupling capacitance (uF)
    use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
    ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
    rx_bw = Float(12.0)  #: CTLE bandwidth (GHz).
    peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
    peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
    ctle_enable = Bool(True)  #: CTLE enable.
    rx_use_ami = Bool(False)  #: (Bool)
    rx_has_ts4 = Bool(False)  #: (Bool)
    rx_use_ts4 = Bool(False)  #: (Bool)
    rx_use_getwave = Bool(False)  #: (Bool)
    rx_has_getwave = Bool(False)  #: (Bool)
    rx_use_clocks = Bool(False)  #: (Bool)
    rx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    rx_ami_valid = Bool(False)  #: (Bool)
    rx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    rx_dll_valid = Bool(False)  #: (Bool)
    rx_ibis_file = File("", entries=5, filter=["*.ibs"])  #: (File)
    rx_ibis_valid = Bool(False)  #: (Bool)
    rx_use_ibis = Bool(False)  #: (Bool)
    rx_use_viterbi = Bool(False)  #: (Bool)
    rx_viterbi_symbols = Int(7)  #: Number of symbols to track in Viterbi decoder.

    # - DFE
    sum_ideal = Bool(True)  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
    decision_scaler = Float(0.5)  #: DFE slicer output voltage (V).
    gain = Float(0.2)  #: DFE error gain (unitless).
    n_ave = Float(100)  #: DFE # of averages to take, before making tap corrections.
    sum_bw = Float(12.0)  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).
    use_agc = Bool(False)  #: Continuously adjust `decision_scalar` when True.

    # - CDR
    delta_t = Float(0.1)  #: CDR proportional branch magnitude (ps).
    alpha = Float(0.01)  #: CDR integral branch magnitude (unitless).
    n_lock_ave = Int(500)  #: CDR # of averages to take in determining lock.
    rel_lock_tol = Float(0.1)  #: CDR relative tolerance to use in determining lock.
    lock_sustain = Int(500)  #: CDR hysteresis to use in determining lock.

    # Misc.
    cfg_file = File("", entries=5, filter=["*.pybert_cfg"])  #: PyBERT configuration data storage file (File).
    data_file = File("", entries=5, filter=["*.pybert_data"])  #: PyBERT results data storage file (File).

    # Plots (plot containers, actually)
    plotdata = ArrayPlotData()
    plots_h = Instance(GridPlotContainer)
    plots_s = Instance(GridPlotContainer)
    plots_p = Instance(GridPlotContainer)
    plots_H = Instance(GridPlotContainer)
    plots_dfe = Instance(GridPlotContainer)
    plots_eye = Instance(GridPlotContainer)
    plots_jitter_dist = Instance(GridPlotContainer)
    plots_jitter_spec = Instance(GridPlotContainer)
    plots_bathtub = Instance(GridPlotContainer)

    # Status
    status = String("Ready.")  #: PyBERT status (String).
    jitter_perf = Float(0.0)
    total_perf = Float(0.0)
    sweep_results = List([])
    len_h = Int(0)
    chnl_dly = Float(0.0)  #: Estimated channel delay (s).
    bit_errs = Int(0)  #: # of bit errors observed in last run.
    bit_errs_viterbi = Int(0)  #: # of bit errors observed in last run.
    run_count = Int(0)  # Used as a mechanism to force bit stream regeneration.

    # About
    perf_info = Property(String, depends_on=["total_perf"])

    # Help
    instructions = help_str

    # Console
    console_log = String("PyBERT Console Log\n\n")

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables,
    #   which don't freeze the GUI noticeably.)
    #
    # - Note: Don't make properties, which have a high calculation overhead,
    #         dependencies of other properties!
    #         This will slow the GUI down noticeably.
    jitter_info = Property(String, depends_on=["jitter_perf"])
    status_str = Property(String, depends_on=["status"])
    sweep_info = Property(String, depends_on=["sweep_results"])
    t = Property(Array, depends_on=["ui", "nspui", "nbits"])
    t_ns = Property(Array, depends_on=["t"])
    f = Property(Array, depends_on=["f_step", "f_max"])
    w = Property(Array, depends_on=["f"])
    t_irfft = Property(Array, depends_on=["f"])
    bits = Property(Array, depends_on=["pattern", "nbits", "mod_type", "run_count"])
    symbols = Property(Array, depends_on=["bits", "mod_type", "vod"])
    ffe = Property(Array, depends_on=["tx_taps.value", "tx_taps.enabled"])
    ui = Property(Float, depends_on=["bit_rate", "mod_type"])
    nui = Property(Int, depends_on=["nbits", "mod_type"])
    eye_uis = Property(Int, depends_on=["eye_bits", "mod_type"])
    dfe_out_p = Array()

    # Custom buttons, which we'll use in particular tabs.
    # (Globally applicable buttons, such as "Run" and "Ok", are handled more simply, in the View.)
    btn_disable = Button(label="Disable All")  # Disable all DFE taps in optimizer.
    btn_enable = Button(label="Enable All")  # Enable all DFE taps in optimizer.
    btn_cfg_tx = Button(label="Configure")  # Configure AMI parameters.
    btn_cfg_rx = Button(label="Configure")
    btn_sel_tx = Button(label="Select")  # Select IBIS model.
    btn_sel_rx = Button(label="Select")
    btn_view_tx = Button(label="View")  # View IBIS model.
    btn_view_rx = Button(label="View")

    # Logger & Pop-up
    def log(self, msg, alert=False, exception=None):
        """Log a message to the console and, optionally, to terminal and/or pop-up dialog."""
        _msg = msg.strip()
        txt = f"[{datetime.now()}]: PyBERT: {_msg}"
        if self.debug:
            # In case PyBERT crashes, before we can read this in its `Console` tab:
            print(txt, flush=True)
        self.console_log += txt + "\n"
        if exception:
            raise exception
        if alert and self.GUI:
            message(_msg, title="PyBERT Alert")

    # User "yes"/"no" alert box.
    def alert(self, msg):
        "Prompt for a yes/no response, using simple alert dialog."
        _msg = msg.strip()
        if self.GUI:
            return error(_msg, "PyBERT Alert")
        raise RuntimeError("Alert box requested, but no GUI!")

    # Default initialization
    def __init__(self, run_simulation=True, gui=True):
        """Initial plot setup occurs here.

        In order to populate the data structure we need to
        construct the plots, we must run the simulation.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
            gui(Bool): Set to `False` for script based usage.
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.GUI = gui
        self.log("Started.")
        self.log_information()
        if self.debug:
            self.log("Debug Mode Enabled.")

        INIT_LEN = 640
        self.plotdata.set_data("t_ns_opt", self.t_ns[:INIT_LEN])
        self.plotdata.set_data("clocks_tune", zeros(INIT_LEN))
        self.plotdata.set_data("ctle_out_h_tune", zeros(INIT_LEN))
        self.plotdata.set_data("s_ctle", zeros(INIT_LEN))
        self.plotdata.set_data("s_ctle_out", zeros(INIT_LEN))
        self.plotdata.set_data("s_tx", zeros(INIT_LEN))
        self.plotdata.set_data("p_chnl", zeros(INIT_LEN))
        self.plotdata.set_data("p_ctle", zeros(INIT_LEN))
        self.plotdata.set_data("p_ctle_out", zeros(INIT_LEN))
        self.plotdata.set_data("p_tx", zeros(INIT_LEN))
        self.plotdata.set_data("p_tx_out", zeros(INIT_LEN))
        self.plotdata.set_data("curs_ix", [0, 0])
        self.plotdata.set_data("curs_amp", [0, 0])

        if run_simulation:
            self.simulate(initial_run=True)

    def _btn_disable_fired(self):
        if self.opt_thread and self.opt_thread.is_alive():
            pass
        else:
            for tap in self.dfe_tap_tuners:
                tap.enabled = False

    def _btn_enable_fired(self):
        if self.opt_thread and self.opt_thread.is_alive():
            pass
        else:
            for tap in self.dfe_tap_tuners:
                tap.enabled = True

    def _btn_cfg_tx_fired(self):
        self._tx_cfg()

    def _btn_cfg_rx_fired(self):
        self._rx_cfg()
        if self.debug:
            self.log(f"User configuration resulted in the following `In`/`InOut` parameter dictionary:\n{self._rx_cfg.input_ami_params}")

    def _btn_sel_tx_fired(self):
        self._tx_ibis()
        if self._tx_ibis.dll_file and self._tx_ibis.ami_file:
            self.tx_dll_file = join(self._tx_ibis_dir, self._tx_ibis.dll_file)
            self.tx_ami_file = join(self._tx_ibis_dir, self._tx_ibis.ami_file)
        else:
            self.tx_dll_file = ""
            self.tx_ami_file = ""

    def _btn_sel_rx_fired(self):
        self._rx_ibis()
        if self._rx_ibis.dll_file and self._rx_ibis.ami_file:
            self.rx_dll_file = join(self._rx_ibis_dir, self._rx_ibis.dll_file)
            self.rx_ami_file = join(self._rx_ibis_dir, self._rx_ibis.ami_file)
        else:
            self.rx_dll_file = ""
            self.rx_ami_file = ""

    def _btn_view_tx_fired(self):
        self._tx_ibis.model()

    def _btn_view_rx_fired(self):
        self._rx_ibis.model()

    # Independent variable setting intercepts
    # (Primarily, for debugging.)
    def _set_ctle_peak_mag_tune(self, val):
        if val > gMaxCTLEPeak or val < 0.0:
            raise RuntimeError("CTLE peak magnitude out of range!")
        self.peak_mag_tune = val

    # Dependent variable definitions
    @cached_property
    def _get_t(self):
        """Calculate the system time vector, in seconds."""

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @cached_property
    def _get_t_ns(self):
        """Calculate the system time vector, in ns."""

        return self.t * 1.0e9

    @cached_property
    def _get_f(self):
        """
        Calculate the frequency vector for channel model construction.
        """
        fstep = self.f_step * 1e6
        fmax  = self.f_max  * 1e9
        return arange(0, fmax + fstep, fstep)  # "+fstep", so fmax gets included

    @cached_property
    def _get_w(self):
        """
        Channel modeling frequency vector, in rads./sec.
        """
        return 2 * pi * self.f

    @cached_property
    def _get_t_irfft(self):
        """
        Calculate the time vector appropriate for indexing `irfft()` output.
        """
        f = self.f
        tmax = 1 / f[1]
        tstep = 0.5 / f[-1]
        return arange(0, tmax, tstep)

    @cached_property
    def _get_bits(self):
        "Generate the bit stream."
        pattern = self.pattern_
        seed = self.seed
        nbits = self.nbits

        if not seed:  # The user sets `seed` to zero when she wants a new random seed generated for each run.
            seed = randint(128)
            while not seed:  # We don't want to seed our LFSR with zero.
                seed = randint(128)
        bit_gen = lfsr_bits(pattern, seed)
        bits = [next(bit_gen) for _ in range(nbits)]
        return array(bits)

    @cached_property
    def _get_ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """

        mod_type = self.mod_type[0]
        bit_rate = self.bit_rate * 1.0e9

        ui = 1.0 / bit_rate
        if mod_type == 2:  # PAM-4
            ui *= 2.0

        return ui

    @cached_property
    def _get_nui(self):
        """Returns the number of unit intervals in the test vectors."""

        mod_type = self.mod_type[0]
        nbits = self.nbits

        nui = nbits
        if mod_type == 2:  # PAM-4
            nui //= 2

        return nui

    @cached_property
    def _get_eye_uis(self):
        """Returns the number of unit intervals to use for eye construction."""

        mod_type = self.mod_type[0]
        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if mod_type == 2:  # PAM-4
            eye_uis //= 2

        return eye_uis

    @cached_property
    def _get_ideal_h(self):
        """Returns the ideal link impulse response."""

        ui = self.ui.value
        nspui = self.nspui
        t = self.t
        mod_type = self.mod_type[0]
        ideal_type = self.ideal_type[0]

        t = array(t) - t[-1] / 2.0

        if ideal_type == 0:  # delta
            ideal_h = zeros(len(t))
            ideal_h[len(t) / 2] = 1.0
        elif ideal_type == 1:  # sinc
            ideal_h = sinc(t / (ui / 2.0))
        elif ideal_type == 2:  # raised cosine
            ideal_h = (cos(pi * t / (ui / 2.0)) + 1.0) / 2.0
            ideal_h = where(t < -ui / 2.0, zeros(len(t)), ideal_h)
            ideal_h = where(t >  ui / 2.0, zeros(len(t)), ideal_h)
        else:
            raise ValueError("PyBERT._get_ideal_h(): ERROR: Unrecognized ideal impulse response type.")

        if mod_type == 1:  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (ideal_h + pad(ideal_h[:-1 * nspui], (nspui, 0), "constant", constant_values=(0, 0)))

        return ideal_h

    @cached_property
    def _get_symbols(self):
        """Generate the symbol stream."""

        mod_type = self.mod_type[0]
        vod = self.vod
        bits = self.bits

        if mod_type == 0:  # NRZ
            symbols = 2 * bits - 1
        elif mod_type == 1:  # Duo-binary
            symbols = [bits[0]]
            for bit in bits[1:]:  # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = 2 * array(symbols) - 1
        elif mod_type == 2:  # PAM-4
            symbols = []
            for bits in zip(bits[0::2], bits[1::2]):
                if bits == (0, 0):
                    symbols.append(-1.0)
                elif bits == (0, 1):
                    symbols.append(-1.0 / 3.0)
                elif bits == (1, 0):
                    symbols.append(1.0 / 3.0)
                else:
                    symbols.append(1.0)
        else:
            raise ValueError("ERROR: _get_symbols(): Unknown modulation type requested!")

        return array(symbols) * vod

    @cached_property
    def _get_ffe(self):
        """Generate the Tx pre-emphasis FIR numerator."""

        tap_tuners = self.tx_taps

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        curs_pos = -tap_tuners[0].pos
        curs_val = 1.0 - sum(abs(array(taps)))
        if curs_pos < 0:
            taps.insert(0, curs_val)
        else:
            taps.insert(curs_pos, curs_val)

        return taps

    # pylint: disable=too-many-locals,consider-using-f-string,too-many-branches,too-many-statements
    # @cached_property
    def _get_jitter_info(self):
        isi_chnl = self.isi_chnl * 1.0e12
        dcd_chnl = self.dcd_chnl * 1.0e12
        pj_chnl = self.pj_chnl * 1.0e12
        rj_chnl = self.rj_chnl * 1.0e12
        isi_tx = self.isi_tx * 1.0e12
        dcd_tx = self.dcd_tx * 1.0e12
        pj_tx = self.pj_tx * 1.0e12
        rj_tx = self.rj_tx * 1.0e12
        isi_ctle = self.isi_ctle * 1.0e12
        dcd_ctle = self.dcd_ctle * 1.0e12
        pj_ctle = self.pj_ctle * 1.0e12
        rj_ctle = self.rj_ctle * 1.0e12
        isi_dfe = self.isi_dfe * 1.0e12
        dcd_dfe = self.dcd_dfe * 1.0e12
        pj_dfe = self.pj_dfe * 1.0e12
        rj_dfe = self.rj_dfe * 1.0e12

        isi_rej_tx = 1.0e20
        dcd_rej_tx = 1.0e20
        isi_rej_ctle = 1.0e20
        dcd_rej_ctle = 1.0e20
        pj_rej_ctle = 1.0e20
        rj_rej_ctle = 1.0e20
        isi_rej_dfe = 1.0e20
        dcd_rej_dfe = 1.0e20
        pj_rej_dfe = 1.0e20
        rj_rej_dfe = 1.0e20
        isi_rej_total = 1.0e20
        dcd_rej_total = 1.0e20
        pj_rej_total = 1.0e20
        rj_rej_total = 1.0e20

        if isi_tx:
            isi_rej_tx = isi_chnl / isi_tx
        if dcd_tx:
            dcd_rej_tx = dcd_chnl / dcd_tx
        if isi_ctle:
            isi_rej_ctle = isi_tx / isi_ctle
        if dcd_ctle:
            dcd_rej_ctle = dcd_tx / dcd_ctle
        if pj_ctle:
            pj_rej_ctle = pj_tx / pj_ctle
        if rj_ctle:
            rj_rej_ctle = rj_tx / rj_ctle
        if isi_dfe:
            isi_rej_dfe = isi_ctle / isi_dfe
        if dcd_dfe:
            dcd_rej_dfe = dcd_ctle / dcd_dfe
        if pj_dfe:
            pj_rej_dfe = pj_ctle / pj_dfe
        if rj_dfe:
            rj_rej_dfe = rj_ctle / rj_dfe
        if isi_dfe:
            isi_rej_total = isi_chnl / isi_dfe
        if dcd_dfe:
            dcd_rej_total = dcd_chnl / dcd_dfe
        if pj_dfe:
            pj_rej_total = pj_tx / pj_dfe
        if rj_dfe:
            rj_rej_total = rj_tx / rj_dfe

        # Temporary, until I figure out DPI independence.
        info_str = "<style>\n"
        info_str += " table td {font-size: 12em;}\n"
        info_str += " table th {font-size: 14em;}\n"
        info_str += "</style>\n"
        # End Temp.

        info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"

        info_str += "<H2>Tx Preemphasis</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_chnl,
            isi_tx,
            10.0 * safe_log10(isi_rej_tx),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += f'<TD align="center">DCD</TD><TD>{dcd_chnl:6.3f}</TD><TD>{dcd_tx:6.3f}</TD><TD>{10.0 * safe_log10(dcd_rej_tx):4.1f}</TD>\n'
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (
            pj_chnl,
            pj_tx,
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (
            rj_chnl,
            rj_tx,
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>CTLE (+ AMI DFE)</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_tx,
            isi_ctle,
            10.0 * safe_log10(isi_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_tx,
            dcd_ctle,
            10.0 * safe_log10(dcd_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_tx,
            pj_ctle,
            10.0 * safe_log10(pj_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_tx,
            rj_ctle,
            10.0 * safe_log10(rj_rej_ctle),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>DFE</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_ctle,
            isi_dfe,
            10.0 * safe_log10(isi_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_ctle,
            dcd_dfe,
            10.0 * safe_log10(dcd_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_ctle,
            pj_dfe,
            10.0 * safe_log10(pj_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_ctle,
            rj_dfe,
            10.0 * safe_log10(rj_rej_dfe),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        info_str += "<H2>TOTAL</H2>\n"
        info_str += '<TABLE border="1">\n'
        info_str += '<TR align="center">\n'
        info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            isi_chnl,
            isi_dfe,
            10.0 * safe_log10(isi_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            dcd_chnl,
            dcd_dfe,
            10.0 * safe_log10(dcd_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            pj_tx,
            pj_dfe,
            10.0 * safe_log10(pj_rej_total),
        )
        info_str += "</TR>\n"
        info_str += '<TR align="right">\n'
        info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
            rj_tx,
            rj_dfe,
            10.0 * safe_log10(rj_rej_total),
        )
        info_str += "</TR>\n"
        info_str += "</TABLE>\n"

        return info_str

    # @cached_property
    def _get_perf_info(self):
        info_str = "<H2>Performance by Component</H2>\n"
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += "      <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>\n"
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Channel</TD><TD>{self.channel_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Tx Preemphasis</TD><TD>{self.tx_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">CTLE</TD><TD>{self.ctle_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">DFE</TD><TD>{self.dfe_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Viterbi</TD><TD>{self.viterbi_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Jitter Analysis</TD><TD>{self.jitter_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center"><strong>TOTAL</strong></TD><TD><strong>{self.total_perf * 60.0e-6:6.3f}</strong></TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Plotting</TD><TD>{self.plotting_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += "  </TABLE>\n"

        return info_str

    # @cached_property
    def _get_sweep_info(self):
        sweep_results = self.sweep_results

        info_str = "<H2>Sweep Results</H2>\n"
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += "      <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>\n"
        info_str += "    </TR>\n"

        for item in sweep_results:
            info_str += '    <TR align="center">\n'
            info_str += str(item)
            info_str += "    </TR>\n"

        info_str += "  </TABLE>\n"

        return info_str

    # @cached_property
    def _get_status_str(self):
        status_str = f"{self.status:20s} | Perf. (Msmpls./min.): {self.total_perf * 60.0e-6:4.1f}"
        dly_str = f"    | ChnlDly (ns): {self.chnl_dly * 1000000000.0:5.3f}"
        if self.bit_errs_viterbi >= 0:
            err_str = f"    | BitErrs: {int(self.bit_errs)} ({int(self.bit_errs_viterbi)})"
        else:
            err_str = f"    | BitErrs: {int(self.bit_errs)}"
        pwr_str = f"    | TxPwr (mW): {self.rel_power * 1e3:3.0f}"
        status_str += dly_str + err_str + pwr_str
        jit_str = "    | Jitter (ps):  ISI=%6.1f  DCD=%6.1f  Pj=%6.1f (%6.1f)  Rj=%6.1f (%6.1f)" % (
            self.isi_dfe * 1.0e12,
            self.dcd_dfe * 1.0e12,
            self.pj_dfe * 1.0e12,
            self.pjDD_dfe * 1.0e12,
            self.rj_dfe * 1.0e12,
            self.rjDD_dfe * 1.0e12,
        )
        status_str += jit_str

        return status_str

    # Changed property handlers.
    def _status_str_changed(self):
        if gDebugStatus:
            print(self.status_str, flush=True)

    def _use_dfe_changed(self, new_value):
        if not new_value:
            for i in range(1, 4):
                self.tx_taps[i].enabled = True
        else:
            for i in range(1, 4):
                self.tx_taps[i].enabled = False

    def _dfe_tap_tuners_changed(self, new_value):
        limits = []
        for tuner in new_value:
            limits.append((tuner.min_val, tuner.max_val))
        self.dfe.limits = limits
        print(f"limits: {limits}", flush=True)

    def _tx_ibis_file_changed(self, new_value):
        self.status = f"Parsing IBIS file: {new_value}"
        dName = ""
        try:
            self.tx_ibis_valid = False
            self.tx_use_ami = False
            self.log(f"Parsing Tx IBIS file, '{new_value}'...")
            ibis = IBISModel(new_value, True, debug=self.debug, gui=self.GUI)
            self.log(f"  Result:\n{ibis.ibis_parsing_errors}")
            self._tx_ibis = ibis
            self.tx_ibis_valid = True
            dName = dirname(new_value)
            if self._tx_ibis.dll_file and self._tx_ibis.ami_file:
                self.tx_dll_file = join(dName, self._tx_ibis.dll_file)
                self.tx_ami_file = join(dName, self._tx_ibis.ami_file)
            else:
                self.tx_dll_file = ""
                self.tx_ami_file = ""
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.status = "IBIS file parsing error!"
            error_message = f"Failed to open and/or parse IBIS file!\n{err}"
            self.log(error_message, alert=True, exception=err)
        self._tx_ibis_dir = dName
        self.status = "Done."

    def _tx_ami_file_changed(self, new_value):
        try:
            self.tx_ami_valid = False
            if new_value:
                self.log(f"Parsing Tx AMI file, '{new_value}'...")
                with open(new_value, mode="r", encoding="utf-8") as pfile:
                    pcfg = AMIParamConfigurator(pfile.read())
                if pcfg.ami_parsing_errors:
                    self.log(f"Non-fatal parsing errors:\n{pcfg.ami_parsing_errors}")
                else:
                    self.log("Success.")
                self.tx_has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
                _tx_returns_impulse = pcfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"])
                if not _tx_returns_impulse:
                    self.tx_use_getwave = True
                if pcfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]):
                    self.tx_has_ts4 = True
                else:
                    self.tx_has_ts4 = False
                self._tx_cfg = pcfg
                self.tx_ami_valid = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            self.log(error_message, alert=True)
            raise

    def _tx_dll_file_changed(self, new_value):
        try:
            self.tx_dll_valid = False
            if new_value:
                model = AMIModel(str(new_value))
                self._tx_model = model
                self.tx_dll_valid = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open DLL/SO file!\n{err}"
            self.log(error_message, alert=True)

    def _rx_ibis_file_changed(self, new_value):
        self.status = f"Parsing IBIS file: {new_value}"
        dName = ""
        try:
            self.rx_ibis_valid = False
            self.rx_use_ami = False
            self.log(f"Parsing Rx IBIS file, '{new_value}'...")
            ibis = IBISModel(new_value, False, self.debug, gui=self.GUI)
            self.log(f"  Result:\n{ibis.ibis_parsing_errors}")
            self._rx_ibis = ibis
            self.rx_ibis_valid = True
            dName = dirname(new_value)
            if self._rx_ibis.dll_file and self._rx_ibis.ami_file:
                self.rx_dll_file = join(dName, self._rx_ibis.dll_file)
                self.rx_ami_file = join(dName, self._rx_ibis.ami_file)
            else:
                self.rx_dll_file = ""
                self.rx_ami_file = ""
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.status = "IBIS file parsing error!"
            error_message = f"Failed to open and/or parse IBIS file!\n{err}"
            self.log(error_message, alert=True)
            raise
        self._rx_ibis_dir = dName
        self.status = "Done."

    def _rx_ami_file_changed(self, new_value):
        try:
            self.rx_ami_valid = False
            if new_value:
                with open(new_value, mode="r", encoding="utf-8") as pfile:
                    pcfg = AMIParamConfigurator(pfile.read())
                self.log(f"Parsing Rx AMI file, '{new_value}'...")
                if pcfg.ami_parsing_errors:
                    self.log(f"Encountered the following errors:\n{pcfg.ami_parsing_errors}")
                else:
                    self.log("Success!")
                self.rx_has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
                _rx_returns_impulse = pcfg.fetch_param_val(["Reserved_Parameters", "Init_Returns_Impulse"])
                if not _rx_returns_impulse:
                    self.rx_use_getwave = True
                if pcfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]):
                    self.rx_has_ts4 = True
                else:
                    self.rx_has_ts4 = False
                self._rx_cfg = pcfg
                self.rx_ami_valid = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            self.log(error_message, alert=True)

    def _rx_dll_file_changed(self, new_value):
        try:
            self.rx_dll_valid = False
            if new_value:
                model = AMIModel(str(new_value))
                self._rx_model = model
                self.rx_dll_valid = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open DLL/SO file!\n{err}"
            self.log(error_message, alert=True)

    def _rx_use_ami_changed(self, new_value):
        if new_value:
            self._btn_disable_fired()

    def check_pat_len(self):
        "Validate chosen pattern length against number of bits being run."
        taps = self.pattern_
        pat_len = 2 * pow(2, max(taps))  # "2 *", to accommodate PAM-4.
        if self.eye_bits < 5 * pat_len:
            self.log("\n".join([
                "Accurate jitter decomposition may not be possible with the current configuration!",
                "Try to keep `EyeBits` > 10 * 2^n, where `n` comes from `PRBS-n`.",]),
                alert=True,
            )

    def check_eye_bits(self):
        "Validate user selected number of eye bits."
        if self.eye_bits > self.nbits:
            self.eye_bits = self.nbits
            self.log("`EyeBits` has been held at `Nbits`.", alert=True)

    def _pattern_changed(self):
        self.check_pat_len()

    def _nbits_changed(self):
        self.check_eye_bits()

    def _eye_bits_changed(self):
        self.check_eye_bits()
        self.check_pat_len()

    def _f_max_changed(self, new_value):
        fmax = 0.5e-9 / self.t[1]  # Nyquist frequency, given our sampling rate (GHz).
        if new_value > fmax:
            self.f_max = fmax
            self.log("`fMax` has been held at the Nyquist frequency.", alert=True)

    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    # pylint: disable=attribute-defined-outside-init
    def calc_chnl_h(self):
        """Calculates the channel impulse response.

        Also sets, in 'self':
         - chnl_dly:
             group delay of channel
         - start_ix:
             first element of trimmed response
         - t_ns_chnl:
             the x-values, in ns, for plotting 'chnl_h'
         - chnl_H:
             channel frequency response
         - chnl_s:
             channel step response
         - chnl_p:
             channel pulse response
        """

        t = self.t  # This time vector has NO relationship to `f`/`w`!
        t_irfft = self.t_irfft  # This time vector IS related to `f`/`w`.
        f = self.f
        w = self.w
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9
        Rs = self.rs
        Cs = self.cout * 1.0e-12
        RL = self.rin
        Cp = self.cin * 1.0e-12
        # CL = self.cac * 1.0e-6  # pylint: disable=unused-variable

        ts = t[1]
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.use_ch_file:
            ch_s2p_pre = import_channel(self.ch_file, ts, f, renumber=self.renumber)
            self.log(str(ch_s2p_pre))
            H = ch_s2p_pre.s21.s.flatten()
        else:
            # Construct PyBERT default channel model (i.e. - Howard Johnson's UTP model).
            # - Grab model parameters from PyBERT instance.
            l_ch = self.l_ch
            v0 = self.v0 * 3.0e8
            R0 = self.R0
            w0 = self.w0
            Rdc = self.Rdc
            Z0 = self.Z0
            Theta0 = self.Theta0
            # - Calculate propagation constant, characteristic impedance, and transfer function.
            gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            self.Zc = Zc
            H = exp(-l_ch * gamma)  # pylint: disable=invalid-unary-operand-type
            self.H = H
            # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
            tmp = np.array(list(zip(zip(zeros(len_f), H), zip(H, zeros(len_f)))))
            ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)
            # - And, finally, renormalize to driver impedance.
            ch_s2p_pre.renormalize(Rs)
        try:
            ch_s2p_pre.name = "ch_s2p_pre"
        except Exception:  # pylint: disable=broad-exception-caught
            print(f"ch_s2p_pre: {ch_s2p_pre}")
            raise
        self.ch_s2p_pre = ch_s2p_pre
        ch_s2p = ch_s2p_pre  # In case neither set of on-die S-parameters is being invoked, below.

        # Augment w/ IBIS-AMI on-die S-parameters, if appropriate.
        def add_ondie_s(s2p, ts4f, isRx=False):
            """Add the effect of on-die S-parameters to channel network.

            Args:
                s2p(skrf.Network): initial 2-port network.
                ts4f(string): on-die S-parameter file name.

            Keyword Args:
                isRx(bool): True when Rx on-die S-params. are being added. (Default = False).

            Returns:
                skrf.Network: Resultant 2-port network.
            """
            ts4N = rf.Network(ts4f)  # Grab the 4-port single-ended on-die network.
            ntwk = sdd_21(ts4N)  # Convert it to a differential, 2-port network.
            # Interpolate to system freqs.
            ntwk2 = ntwk.extrapolate_to_dc().windowed(normalize=False).interpolate(
                s2p.f, coords='polar', bounds_error=False, fill_value='extrapolate')
            if isRx:
                res = s2p**ntwk2
            else:  # Tx
                res = ntwk2**s2p
            return (res, ts4N, ntwk2)

        if self.tx_use_ibis:
            model = self._tx_ibis.model
            Rs = model.zout * 2
            Cs = model.ccomp[0] / 2  # They're in series.
            self.Rs = Rs  # Primarily for debugging.
            self.Cs = Cs
            if self.tx_use_ts4:
                fname = join(self._tx_ibis_dir, self._tx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname)
                self.ts4N = ts4N
                self.ntwk = ntwk
        if self.rx_use_ibis:
            model = self._rx_ibis.model
            RL = model.zin * 2
            Cp = model.ccomp[0] / 2
            self.RL = RL  # Primarily for debugging.
            self.Cp = Cp
            if self.debug:
                self.log(f"RL: {RL}, Cp: {Cp}")
            if self.rx_use_ts4:
                fname = join(self._rx_ibis_dir, self._rx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
                self.ts4N = ts4N
                self.ntwk = ntwk
        ch_s2p.name = "ch_s2p"
        self.ch_s2p = ch_s2p

        # Calculate channel impulse response.
        Zs = Rs / (1 + 1j * w * Rs * Cs)  # Tx termination impedance
        Zt = RL / (1 + 1j * w * RL * Cp)  # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:, 0] = Zs
        ch_s2p_term_z0[:, 1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term

        # We take the transfer function, H, to be a ratio of voltages.
        # So, we must normalize our (now generalized) S-parameters.
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:, 1] / ch_s2p_term.z0[:, 0])
        if self.use_window:
            chnl_h = irfft(raised_cosine(chnl_H))
        else:
            chnl_h = irfft(chnl_H)
        krnl = interp1d(t_irfft, chnl_h, kind="cubic",
                        bounds_error=False, fill_value=0, assume_sorted=True)
        temp = krnl(t)
        chnl_h = temp * t[1] / t_irfft[1]
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = int(impulse_length / ts)
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len,
                                        front_porch=True, kept_energy=0.999)
        krnl = interp1d(t[:len(chnl_h)], chnl_h, kind="cubic",
                        bounds_error=False, fill_value=0, assume_sorted=True)
        chnl_trimmed_H = rfft(krnl(t_irfft)) * t_irfft[1] / t[1]

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))  # pylint: disable=invalid-unary-operand-type

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_H_raw = H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix: start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    def simulate(self, initial_run=False, update_plots=True):
        """Run all queued simulations."""
        # Running the simulation will fill in the required data structure.
        my_run_simulation(self, initial_run=initial_run, update_plots=update_plots)
        # Once the required data structure is filled in, we can create the plots.
        make_plots(self, n_dfe_taps=len(self.dfe_tap_tuners))

    def load_configuration(self, filepath: Path):
        """Load in a configuration into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertCfg.load_from_file(filepath, self)
            self.cfg_file = filepath
            self.status = "Loaded configuration."
        except InvalidFileType:
            self.log("This filetype is not currently supported.")
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.log("Failed to load configuration. See the console for more detail.")
            self.log(str(err))

    def save_configuration(self, filepath: Path):
        """Save out a configuration from pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertCfg(self, time.asctime(), VERSION).save(filepath)
            self.cfg_file = filepath
            self.status = "Configuration saved."
        except InvalidFileType:
            self.log("This filetype is not currently supported. Please try again as a yaml file.")
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.log(f"Failed to save configuration:\n\t{err}", alert=True)

    def load_results(self, filepath: Path):
        """Load results from a file into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData.load_from_file(filepath, self)
            self.data_file = filepath
            self.status = "Loaded results."
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.log("Failed to load results from file. See the console for more detail.")
            self.log(str(err))

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData(self, time.asctime(), VERSION).save(filepath)
            self.data_file = filepath
            self.status = "Saved results."
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.log("Failed to save results to file. See the console for more detail.")
            self.log(str(err))

    def clear_reference_from_plots(self):
        """If any plots have ref in the name, delete them and then regenerate the plots.

        If we don't actually delete any data, skip regenerating the plots.
        """
        atleast_one_reference_removed = False

        for reference_plot in self.plotdata.list_data():
            if "ref" in reference_plot:
                try:
                    atleast_one_reference_removed = True
                    self.plotdata.del_data(reference_plot)
                except KeyError:
                    pass

        if atleast_one_reference_removed:
            make_plots(self, n_dfe_taps=len(self.dfe_tap_tuners))

    def log_information(self):
        """Log the system information."""
        self.log(f"System: {platform.system()} {platform.release()}")
        self.log(f"Python Version: {platform.python_version()}")
        self.log(f"PyBERT Version: {VERSION}")
        self.log(f"PyAMI Version: {PyAMI_VERSION}")
        self.log(f"GUI Toolkit: {ETSConfig.toolkit}")
        self.log(f"Kiva Backend: {ETSConfig.kiva_backend}")

    _tx_ibis = Instance(IBISModel)
    _tx_ibis_dir = ""
    _tx_cfg = Instance(AMIParamConfigurator)
    _tx_model = Instance(AMIModel)
    _rx_ibis = Instance(IBISModel)
    _rx_ibis_dir = ""
    _rx_cfg = Instance(AMIParamConfigurator)
    _rx_model = Instance(AMIModel)

    isi_chnl = 0
    dcd_chnl = 0
    pj_chnl = 0
    rj_chnl = 0
    pjDD_chnl = 0
    rjDD_chnl = 0
    isi_tx = 0
    dcd_tx = 0
    pj_tx = 0
    rj_tx = 0
    pjDD_tx = 0
    rjDD_tx = 0
    isi_ctle = 0
    dcd_ctle = 0
    pj_ctle = 0
    rj_ctle = 0
    pjDD_ctle = 0
    rjDD_ctle = 0
    isi_dfe = 0
    dcd_dfe = 0
    pj_dfe = 0
    rj_dfe = 0
    pjDD_dfe = 0
    rjDD_dfe = 0
