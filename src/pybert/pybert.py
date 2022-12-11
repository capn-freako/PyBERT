#! /usr/bin/env python

"""Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
# fmt: off
# isort: off
from traits.etsconfig.api import ETSConfig
# ETSConfig.toolkit = 'qt4'  # Yields unacceptably small font sizes in plot axis labels.
# ETSConfig.toolkit = 'qt4.celiagg'   # Yields unacceptably small font sizes in plot axis labels.
# ETSConfig.toolkit = 'qt.qpainter'  # Was causing crash on Mac.
# ETSConfig.toolkit = 'qt.image'     # Program runs, but very small fonts in plot titles and axis labels.
# ETSConfig.toolkit = 'wx'           # Crashes on launch.
# fmt: on
# isort: on
import platform
import time
from datetime import datetime
from os.path import dirname, join
from pathlib import Path
from threading import Event, Thread
from time import sleep

import numpy as np
import skrf as rf
from numpy import array, convolve, cos, exp, ones, pad, pi, sinc, where, zeros
from numpy.fft import fft, irfft
from numpy.random import randint
from pyibisami import __version__ as PyAMI_VERSION
from pyibisami.ami.model import AMIModel
from pyibisami.ami.parser import AMIParamConfigurator
from pyibisami.ibis.file import IBISModel
from scipy.optimize import minimize, minimize_scalar
from traits.api import (
    Array,
    Bool,
    Button,
    Enum,
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
from traitsui.message import message

from chaco.api import ArrayPlotData, GridPlotContainer
from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY
from pybert import __date__ as DATE
from pybert import __version__ as VERSION
from pybert.configuration import InvalidFileType, PyBertCfg
from pybert.gui.help import help_str
from pybert.gui.plot import make_plots
from pybert.models.bert import my_run_simulation
from pybert.results import PyBertData
from pybert.utility import (
    calc_gamma,
    import_channel,
    interp_s2p,
    lfsr_bits,
    make_ctle,
    pulse_center,
    safe_log10,
    sdd_21,
    trim_impulse,
)

gDebugStatus = False
gDebugOptimize = False
gMaxCTLEPeak = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
gMaxCTLEFreq = 20.0  # max. allowed CTLE peak frequency (GHz) (when optimizing, only)

# Default model parameters - Modify these to customize the default simulation.
# - Simulation Control
gBitRate = 10  # (Gbps)
gNbits = 8000  # number of bits to run
gPatLen = 127  # repeating bit pattern length
gNspb = 32  # samples per bit
gNumAve = 1  # Number of bit error samples to average, when sweeping.
# - Channel Control
#     - parameters for Howard Johnson's "Metallic Transmission Model"
#     - (See "High Speed Signal Propagation", Sec. 3.1.)
#     - ToDo: These are the values for 24 guage twisted copper pair; need to add other options.
gRdc = 0.1876  # Ohms/m
gw0 = 10.0e6  # 10 MHz is recommended in Ch. 8 of his second book, in which UTP is described in detail.
gR0 = 1.452  # skin-effect resistance (Ohms/m)log
gTheta0 = 0.02  # loss tangent
gZ0 = 100.0  # characteristic impedance in LC region (Ohms)
gv0 = 0.67  # relative propagation velocity (c)
gl_ch = 1.0  # cable length (m)
gRn = (
    0.001  # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)
)
# - Tx
gVod = 1.0  # output drive strength (Vp)
gRs = 100  # differential source impedance (Ohms)
gCout = 0.50  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)
# - Rx
gRin = 100  # differential input resistance
gCin = 0.50  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gCac = 1.0  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseDfe = True  # Include DFE when running simulation.
gDfeIdeal = True  # DFE ideal summing node selector
gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 1.7  # CTLE peaking magnitude (dB)
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)
# - DFE
gDecisionScaler = 0.5
gNtaps = 5
gGain = 0.5
gNave = 100
gDfeBW = 12.0  # DFE summing node bandwidth (GHz)
# - CDR
gDeltaT = 0.1  # (ps)
gAlpha = 0.01
gNLockAve = 500  # number of UI used to average CDR locked status.
gRelLockTol = 0.1  # relative lock tolerance of CDR.
gLockSustain = 500
# - Analysis
gThresh = 6  # threshold for identifying periodic jitter spectral elements (sigma)


class StoppableThread(Thread):
    """Thread class with a stop() method.

    The thread itself has to check regularly for the stopped() condition.

    All PyBERT thread classes are subclasses of this class.
    """

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = Event()

    def stop(self):
        """Called by thread invoker, when thread should be stopped
        prematurely."""
        self._stop_event.set()

    def stopped(self):
        """Should be called by thread (i.e. - subclass) periodically and, if this function
        returns True, thread should clean itself up and quit ASAP.
        """
        return self._stop_event.is_set()


class TxOptThread(StoppableThread):
    """Used to run Tx tap weight optimization in its own thread, in order to
    preserve GUI responsiveness."""

    def run(self):
        """Run the Tx equalization optimization thread."""

        pybert = self.pybert

        if self.update_status:
            pybert.status = "Optimizing Tx..."

        max_iter = pybert.max_iter

        old_taps = []
        min_vals = []
        max_vals = []
        for tuner in pybert.tx_tap_tuners:
            if tuner.enabled:
                old_taps.append(tuner.value)
                min_vals.append(tuner.min_val)
                max_vals.append(tuner.max_val)

        cons = {"type": "ineq", "fun": lambda x: 0.7 - sum(abs(x))}

        bounds = list(zip(min_vals, max_vals))

        try:
            if gDebugOptimize:
                res = minimize(
                    self.do_opt_tx,
                    old_taps,
                    bounds=bounds,
                    constraints=cons,
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize(
                    self.do_opt_tx,
                    old_taps,
                    bounds=bounds,
                    constraints=cons,
                    options={"disp": False, "maxiter": max_iter},
                )

            if self.update_status:
                if res["success"]:
                    pybert.status = "Optimization succeeded."
                else:
                    pybert.status = f"Optimization failed: {res['message']}"

        except Exception as err:
            pybert.status = err

    def do_opt_tx(self, taps):
        """Run the Tx Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        tuners = pybert.tx_tap_tuners
        taps = list(taps)
        for tuner in tuners:
            if tuner.enabled:
                tuner.value = taps.pop(0)
        return pybert.cost


class RxOptThread(StoppableThread):
    """Used to run Rx tap weight optimization in its own thread, in order to
    preserve GUI responsiveness."""

    def run(self):
        """Run the Rx equalization optimization thread."""

        pybert = self.pybert

        pybert.status = "Optimizing Rx..."
        max_iter = pybert.max_iter
        max_mag_tune = pybert.max_mag_tune

        try:
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = f"Optimization failed: {res['message']}"

        except Exception as err:
            pybert.status = err

    def do_opt_rx(self, peak_mag):
        """Run the Rx Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        pybert.peak_mag_tune = peak_mag
        return pybert.cost


class CoOptThread(StoppableThread):
    """Used to run co-optimization in its own thread, in order to preserve GUI
    responsiveness."""

    def run(self):
        """Run the Tx/Rx equalization co-optimization thread."""

        pybert = self.pybert

        pybert.status = "Co-optimizing..."
        max_iter = pybert.max_iter
        max_mag_tune = pybert.max_mag_tune

        try:
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, max_mag_tune),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = f"Optimization failed: {res['message']}"

        except Exception as err:
            pybert.status = err

    def do_coopt(self, peak_mag):
        """Run the Tx and Rx Co-Optimization."""
        sleep(0.001)  # Give the GUI a chance to acknowledge user clicking the Abort button.

        if self.stopped():
            raise RuntimeError("Optimization aborted.")

        pybert = self.pybert
        pybert.peak_mag_tune = peak_mag
        if any([pybert.tx_tap_tuners[i].enabled for i in range(len(pybert.tx_tap_tuners))]):
            while pybert.tx_opt_thread and pybert.tx_opt_thread.is_alive():
                sleep(0.001)
            pybert._do_opt_tx(update_status=False)
            while pybert.tx_opt_thread and pybert.tx_opt_thread.is_alive():
                sleep(0.001)
        return pybert.cost


class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    name = String("(noname)")
    enabled = Bool(False)
    min_val = Float(0.0)
    max_val = Float(0.0)
    value = Float(0.0)
    steps = Int(0)  # Non-zero means we want to sweep it.

    def __init__(self, name="(noname)", enabled=False, min_val=0.0, max_val=0.0, value=0.0, steps=0):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.name = name
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.steps = steps


class PyBERT(HasTraits):
    """A serial communication link bit error rate tester (BERT) simulator with
    a GUI interface.

    Useful for exploring the concepts of serial communication link
    design.
    """

    # Independent variables

    # - Simulation Control
    bit_rate = Range(low=0.1, high=120.0, value=gBitRate)  #: (Gbps)
    nbits = Range(low=1000, high=10000000, value=gNbits)  #: Number of bits to simulate.
    pattern = Map(
        {
            "PRBS-7": [7, 6],
            "PRBS-15": [15, 14],
            "PRBS-23": [23, 18],
        },
        default_value="PRBS-7",
    )
    seed = Int(1)  # LFSR seed. 0 means regenerate bits, using a new random seed, each run.
    nspb = Range(low=2, high=256, value=gNspb)  #: Signal vector samples per bit.
    eye_bits = Int(gNbits // 5)  #: # of bits used to form eye. (Default = last 20%)
    mod_type = List([0])  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
    num_sweeps = Int(1)  #: Number of sweeps to run.
    sweep_num = Int(1)
    sweep_aves = Int(gNumAve)
    do_sweep = Bool(False)  #: Run sweeps? (Default = False)
    debug = Bool(False)  #: Send log messages to terminal, as well as console, when True. (Default = False)

    # - Channel Control
    ch_file = File(
        "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
    )  #: Channel file name.
    use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
    f_step = Float(10)  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
    impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
    Rdc = Float(gRdc)  #: Channel d.c. resistance (Ohms/m).
    w0 = Float(gw0)  #: Channel transition frequency (rads./s).
    R0 = Float(gR0)  #: Channel skin effect resistance (Ohms/m).
    Theta0 = Float(gTheta0)  #: Channel loss tangent (unitless).
    Z0 = Float(gZ0)  #: Channel characteristic impedance, in LC region (Ohms).
    v0 = Float(gv0)  #: Channel relative propagation velocity (c).
    l_ch = Float(gl_ch)  #: Channel length (m).

    # - EQ Tune
    tx_tap_tuners = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )  #: EQ optimizer list of TxTapTuner objects.
    rx_bw_tune = Float(gBW)  #: EQ optimizer CTLE bandwidth (GHz).
    peak_freq_tune = Float(gPeakFreq)  #: EQ optimizer CTLE peaking freq. (GHz).
    peak_mag_tune = Float(gPeakMag)  #: EQ optimizer CTLE peaking mag. (dB).
    max_mag_tune = Float(20)  #: EQ optimizer CTLE peaking mag. (dB).
    ctle_offset_tune = Float(gCTLEOffset)  #: EQ optimizer CTLE d.c. offset (dB).
    ctle_mode_tune = Enum("Off", "Passive", "AGC", "Manual")  #: EQ optimizer CTLE mode
    use_dfe_tune = Bool(gUseDfe)  #: EQ optimizer DFE select (Bool).
    n_taps_tune = Int(gNtaps)  #: EQ optimizer # DFE taps.
    max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.
    tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
    rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
    coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.

    # - Tx
    vod = Float(gVod)  #: Tx differential output voltage (V)
    rs = Float(gRs)  #: Tx source impedance (Ohms)
    cout = Range(low=0.001, high=1000, value=gCout)  #: Tx parasitic output capacitance (pF)
    pn_mag = Float(gPnMag)  #: Periodic noise magnitude (V).
    pn_freq = Float(gPnFreq)  #: Periodic noise frequency (MHz).
    rn = Float(gRn)  #: Standard deviation of Gaussian random noise (V).
    tx_taps = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=-0.066),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
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
    rin = Float(gRin)  #: Rx input impedance (Ohm)
    cin = Float(gCin)  #: Rx parasitic input capacitance (pF)
    cac = Float(gCac)  #: Rx a.c. coupling capacitance (uF)
    use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
    ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
    rx_bw = Float(gBW)  #: CTLE bandwidth (GHz).
    peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
    peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
    ctle_offset = Float(gCTLEOffset)  #: CTLE d.c. offset (dB)
    ctle_mode = Enum("Off", "Passive", "AGC", "Manual")  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    ctle_mode = "Passive"
    rx_use_ami = Bool(False)  #: (Bool)
    rx_has_ts4 = Bool(False)  #: (Bool)
    rx_use_ts4 = Bool(False)  #: (Bool)
    rx_use_getwave = Bool(False)  #: (Bool)
    rx_has_getwave = Bool(False)  #: (Bool)
    rx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    rx_ami_valid = Bool(False)  #: (Bool)
    rx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    rx_dll_valid = Bool(False)  #: (Bool)
    rx_ibis_file = File("", entries=5, filter=["*.ibs"])  #: (File)
    rx_ibis_valid = Bool(False)  #: (Bool)
    rx_use_ibis = Bool(False)  #: (Bool)

    # - DFE
    use_dfe = Bool(gUseDfe)  #: True = use a DFE (Bool).
    sum_ideal = Bool(gDfeIdeal)  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
    decision_scaler = Float(gDecisionScaler)  #: DFE slicer output voltage (V).
    gain = Float(gGain)  #: DFE error gain (unitless).
    n_ave = Float(gNave)  #: DFE # of averages to take, before making tap corrections.
    n_taps = Int(gNtaps)  #: DFE # of taps.
    _old_n_taps = n_taps
    sum_bw = Float(gDfeBW)  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

    # - CDR
    delta_t = Float(gDeltaT)  #: CDR proportional branch magnitude (ps).
    alpha = Float(gAlpha)  #: CDR integral branch magnitude (unitless).
    n_lock_ave = Int(gNLockAve)  #: CDR # of averages to take in determining lock.
    rel_lock_tol = Float(gRelLockTol)  #: CDR relative tolerance to use in determining lock.
    lock_sustain = Int(gLockSustain)  #: CDR hysteresis to use in determining lock.

    # - Analysis
    thresh = Int(gThresh)  #: Threshold for identifying periodic jitter components (sigma).

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
    run_count = Int(0)  # Used as a mechanism to force bit stream regeneration.

    # About
    perf_info = Property(String, depends_on=["total_perf"])

    # Help
    instructions = help_str

    # Console
    console_log = String("PyBERT Console Log\n\n")

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
    #
    # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
    #         This will slow the GUI down noticeably.
    jitter_info = Property(String, depends_on=["jitter_perf"])
    status_str = Property(String, depends_on=["status"])
    sweep_info = Property(String, depends_on=["sweep_results"])
    tx_h_tune = Property(Array, depends_on=["tx_tap_tuners.value", "nspui"])
    ctle_h_tune = Property(
        Array,
        depends_on=[
            "peak_freq_tune",
            "peak_mag_tune",
            "rx_bw_tune",
            "w",
            "len_h",
            "ctle_mode_tune",
            "ctle_offset_tune",
            "use_dfe_tune",
            "n_taps_tune",
        ],
    )
    ctle_out_h_tune = Property(Array, depends_on=["tx_h_tune", "ctle_h_tune", "chnl_h"])
    cost = Property(Float, depends_on=["ctle_out_h_tune", "nspui"])
    rel_opt = Property(Float, depends_on=["cost"])
    t = Property(Array, depends_on=["ui", "nspb", "nbits"])
    t_ns = Property(Array, depends_on=["t"])
    f = Property(Array, depends_on=["t"])
    w = Property(Array, depends_on=["f"])
    bits = Property(Array, depends_on=["pattern", "nbits", "mod_type", "run_count"])
    symbols = Property(Array, depends_on=["bits", "mod_type", "vod"])
    ffe = Property(Array, depends_on=["tx_taps.value", "tx_taps.enabled"])
    ui = Property(Float, depends_on=["bit_rate", "mod_type"])
    nui = Property(Int, depends_on=["nbits", "mod_type"])
    nspui = Property(Int, depends_on=["nspb", "mod_type"])
    eye_uis = Property(Int, depends_on=["eye_bits", "mod_type"])
    dfe_out_p = Array()
    przf_err = Property(Float, depends_on=["dfe_out_p"])

    # Custom buttons, which we'll use in particular tabs.
    # (Globally applicable buttons, such as "Run" and "Ok", are handled more simply, in the View.)
    btn_rst_eq = Button(label="ResetEq")
    btn_save_eq = Button(label="SaveEq")
    btn_opt_tx = Button(label="OptTx")
    btn_opt_rx = Button(label="OptRx")
    btn_coopt = Button(label="CoOpt")
    btn_abort = Button(label="Abort")
    btn_cfg_tx = Button(label="Configure")  # Configure AMI parameters.
    btn_cfg_rx = Button(label="Configure")
    btn_sel_tx = Button(label="Select")  # Select IBIS model.
    btn_sel_rx = Button(label="Select")
    btn_view_tx = Button(label="View")  # View IBIS model.
    btn_view_rx = Button(label="View")

    # Logger & Pop-up
    def log(self, msg, alert=False, exception=None):
        """Log a message to the console and, optionally, to terminal and/or
        pop-up dialog."""
        _msg = msg.strip()
        txt = f"[{datetime.now()}]: PyBERT: {_msg}"
        if self.debug:
            ## In case PyBERT crashes, before we can read this in its `Console` tab:
            print(txt, flush=True)
        self.console_log += txt + "\n"
        if exception:
            raise exception
        if alert and self.GUI:
            message(_msg, "PyBERT Alert")

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

        if run_simulation:
            self.simulate(initial_run=True)
        else:
            self.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().

    # Custom button handlers
    def _btn_rst_eq_fired(self):
        """Reset the equalization."""
        for i in range(4):
            self.tx_tap_tuners[i].value = self.tx_taps[i].value
            self.tx_tap_tuners[i].enabled = self.tx_taps[i].enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_mode_tune = self.ctle_mode
        self.ctle_offset_tune = self.ctle_offset
        self.use_dfe_tune = self.use_dfe
        self.n_taps_tune = self.n_taps

    def _btn_save_eq_fired(self):
        """Save the equalization."""
        for i in range(4):
            self.tx_taps[i].value = self.tx_tap_tuners[i].value
            self.tx_taps[i].enabled = self.tx_tap_tuners[i].enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_mode = self.ctle_mode_tune
        self.ctle_offset = self.ctle_offset_tune
        self.use_dfe = self.use_dfe_tune
        self.n_taps = self.n_taps_tune

    def _btn_opt_tx_fired(self):
        if (
            self.tx_opt_thread
            and self.tx_opt_thread.is_alive()
            or not any([self.tx_tap_tuners[i].enabled for i in range(len(self.tx_tap_tuners))])
        ):
            pass
        else:
            self._do_opt_tx()

    def _do_opt_tx(self, update_status=True):
        self.tx_opt_thread = TxOptThread()
        self.tx_opt_thread.pybert = self
        self.tx_opt_thread.update_status = update_status
        self.tx_opt_thread.start()

    def _btn_opt_rx_fired(self):
        if self.rx_opt_thread and self.rx_opt_thread.is_alive() or self.ctle_mode_tune == "Off":
            pass
        else:
            self.rx_opt_thread = RxOptThread()
            self.rx_opt_thread.pybert = self
            self.rx_opt_thread.start()

    def _btn_coopt_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            pass
        else:
            self.coopt_thread = CoOptThread()
            self.coopt_thread.pybert = self
            self.coopt_thread.start()

    def _btn_abort_fired(self):
        if self.coopt_thread and self.coopt_thread.is_alive():
            self.coopt_thread.stop()
            self.coopt_thread.join(10)
        if self.tx_opt_thread and self.tx_opt_thread.is_alive():
            self.tx_opt_thread.stop()
            self.tx_opt_thread.join(10)
        if self.rx_opt_thread and self.rx_opt_thread.is_alive():
            self.rx_opt_thread.stop()
            self.rx_opt_thread.join(10)

    def _btn_cfg_tx_fired(self):
        self._tx_cfg()

    def _btn_cfg_rx_fired(self):
        self._rx_cfg()

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
        """Calculate the frequency vector appropriate for indexing non-shifted
        FFT output, in Hz.

        # (i.e. - [0, f0, 2 * f0, ... , fN] + [-(fN - f0), -(fN - 2 * f0), ... , -f0]

        Note: Changed to positive freqs. only, in conjunction w/ irfft() usage.
        """
        t = self.t
        npts = len(t)
        f0 = 1.0 / (t[1] * npts)
        half_npts = npts // 2
        return array([i * f0 for i in range(half_npts)])

    @cached_property
    def _get_w(self):
        """System frequency vector, in rads./sec."""
        return 2 * pi * self.f

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
    def _get_nspui(self):
        """Returns the number of samples per unit interval."""

        mod_type = self.mod_type[0]
        nspb = self.nspb

        nspui = nspb
        if mod_type == 2:  # PAM-4
            nspui *= 2

        return nspui

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

        ui = self.ui
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
            ideal_h = where(t > ui / 2.0, zeros(len(t)), ideal_h)
        else:
            raise Exception("PyBERT._get_ideal_h(): ERROR: Unrecognized ideal impulse response type.")

        if mod_type == 1:  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (ideal_h + pad(ideal_h[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)))

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
            raise Exception("ERROR: _get_symbols(): Unknown modulation type requested!")

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
        taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

        return taps

    @cached_property
    def _get_jitter_info(self):
        try:
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
            # info_str += ' table td {font-size: 36px;}\n'
            # info_str += ' table th {font-size: 38px;}\n'
            info_str += " table td {font-size: 12em;}\n"
            info_str += " table th {font-size: 14em;}\n"
            info_str += "</style>\n"
            # info_str += '<font size="+3">\n'
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
            info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
                dcd_chnl,
                dcd_tx,
                10.0 * safe_log10(dcd_rej_tx),
            )
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
        except Exception as err:
            info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"
            info_str += "Sorry, the following error occurred:\n"
            info_str += str(err)

        return info_str

    @cached_property
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
        info_str += f'      <TD align="center">Jitter Analysis</TD><TD>{self.jitter_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center"><strong>TOTAL</strong></TD><TD><strong>%6.3f</strong></TD>\n' % (
            self.total_perf * 60.0e-6
        )
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += f'      <TD align="center">Plotting</TD><TD>{self.plotting_perf * 6e-05:6.3f}</TD>\n'
        info_str += "    </TR>\n"
        info_str += "  </TABLE>\n"

        return info_str

    @cached_property
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
            # info_str += "      <TD>%+06.3f</TD><TD>%+06.3f</TD><TD>%d</TD><TD>%d</TD>\n" % (
            #     item[0],
            #     item[1],
            #     item[2],
            #     item[3],
            # )
            info_str += "    </TR>\n"

        info_str += "  </TABLE>\n"

        return info_str

    @cached_property
    def _get_status_str(self):
        status_str = "%-20s | Perf. (Msmpls./min.):  %4.1f" % (
            self.status,
            self.total_perf * 60.0e-6,
        )
        dly_str = f"         | ChnlDly (ns):    {self.chnl_dly * 1000000000.0:5.3f}"
        err_str = f"         | BitErrs: {int(self.bit_errs)}"
        pwr_str = f"         | TxPwr (W): {self.rel_power:4.2f}"
        status_str += dly_str + err_str + pwr_str

        try:
            jit_str = "         | Jitter (ps):    ISI=%6.3f    DCD=%6.3f    Pj=%6.3f    Rj=%6.3f" % (
                self.isi_dfe * 1.0e12,
                self.dcd_dfe * 1.0e12,
                self.pj_dfe * 1.0e12,
                self.rj_dfe * 1.0e12,
            )
        except:
            jit_str = "         | (Jitter not available.)"

        status_str += jit_str

        return status_str

    @cached_property
    def _get_tx_h_tune(self):
        nspui = self.nspui
        tap_tuners = self.tx_tap_tuners

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        taps.insert(1, 1.0 - sum(map(abs, taps)))  # Assume one pre-tap.

        h = sum([[x] + list(zeros(nspui - 1)) for x in taps], [])

        return h

    @cached_property
    def _get_ctle_h_tune(self):
        w = self.w
        len_h = self.len_h
        rx_bw = self.rx_bw_tune * 1.0e9
        peak_freq = self.peak_freq_tune * 1.0e9
        peak_mag = self.peak_mag_tune
        offset = self.ctle_offset_tune
        mode = self.ctle_mode_tune

        _, H = make_ctle(rx_bw, peak_freq, peak_mag, w, mode, offset)
        h = irfft(H)

        return h

    @cached_property
    def _get_ctle_out_h_tune(self):
        chnl_h = self.chnl_h
        tx_h = self.tx_h_tune
        ctle_h = self.ctle_h_tune

        tx_out_h = convolve(tx_h, chnl_h)
        return convolve(ctle_h, tx_out_h)

    @cached_property
    def _get_cost(self):
        nspui = self.nspui
        h = self.ctle_out_h_tune
        mod_type = self.mod_type[0]

        s = h.cumsum()
        p = s - pad(s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        (clock_pos, thresh) = pulse_center(p, nspui)
        if clock_pos == -1:
            return 1.0  # Returning a large cost lets it know it took a wrong turn.
        clocks = thresh * ones(len(p))
        if mod_type == 1:  # Handle duo-binary.
            clock_pos -= nspui // 2
        clocks[clock_pos] = 0.0
        if mod_type == 1:  # Handle duo-binary.
            clocks[clock_pos + nspui] = 0.0

        # Cost is simply ISI minus main lobe amplitude.
        # Note: post-cursor ISI is NOT included in cost, when we're using the DFE.
        isi = 0.0
        ix = clock_pos - nspui
        while ix >= 0:
            clocks[ix] = 0.0
            isi += abs(p[ix])
            ix -= nspui
        ix = clock_pos + nspui
        if mod_type == 1:  # Handle duo-binary.
            ix += nspui
        while ix < len(p):
            clocks[ix] = 0.0
            if not self.use_dfe_tune:
                isi += abs(p[ix])
            ix += nspui
        if self.use_dfe_tune:
            for i in range(self.n_taps_tune):
                if clock_pos + nspui * (1 + i) < len(p):
                    p[int(clock_pos + nspui * (0.5 + i)) :] -= p[clock_pos + nspui * (1 + i)]
        plot_len = len(self.chnl_h)
        self.plotdata.set_data("ctle_out_h_tune", p[:plot_len])
        self.plotdata.set_data("clocks_tune", clocks[:plot_len])

        if mod_type == 1:  # Handle duo-binary.
            return isi - p[clock_pos] - p[clock_pos + nspui] + 2.0 * abs(p[clock_pos + nspui] - p[clock_pos])
        return isi - p[clock_pos]

    @cached_property
    def _get_rel_opt(self):
        return -self.cost

    @cached_property
    def _get_przf_err(self):
        p = self.dfe_out_p
        nspui = self.nspui
        n_taps = self.n_taps

        (clock_pos, _) = pulse_center(p, nspui)
        err = 0
        len_p = len(p)
        for i in range(n_taps):
            ix = clock_pos + (i + 1) * nspui
            if ix < len_p:
                err += p[ix] ** 2

        return err / p[clock_pos] ** 2

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

    def _use_dfe_tune_changed(self, new_value):
        if not new_value:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = True
        else:
            for i in range(1, 4):
                self.tx_tap_tuners[i].enabled = False

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
        except Exception as err:
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
        except Exception as err:
            raise
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            self.log(error_message, alert=True)

    def _tx_dll_file_changed(self, new_value):
        try:
            self.tx_dll_valid = False
            if new_value:
                model = AMIModel(str(new_value))
                self._tx_model = model
                self.tx_dll_valid = True
        except Exception as err:
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
        except Exception as err:
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
                self.log(f"Parsing Rx AMI file, '{new_value}'...\n{pcfg.ami_parsing_errors}")
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
        except Exception as err:
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            self.log(error_message, alert=True)

    def _rx_dll_file_changed(self, new_value):
        try:
            self.rx_dll_valid = False
            if new_value:
                model = AMIModel(str(new_value))
                self._rx_model = model
                self.rx_dll_valid = True
        except Exception as err:
            error_message = f"Failed to open DLL/SO file!\n{err}"
            self.log(error_message, alert=True)

    def _rx_use_ami_changed(self, new_value):
        if new_value:
            self.use_dfe = False

    def check_pat_len(self):
        taps = self.pattern_
        pat_len = 2 * pow(2, max(taps))
        if pat_len > 5 * self.nbits:
            self.log(
                "Accurate jitter decomposition may not be possible with the current configuration!\n \
Try to keep Nbits & EyeBits > 10 * 2^n, where `n` comes from `PRBS-n`.",
                alert=True,
            )

    def _pattern_changed(self, new_value):
        self.check_pat_len()

    def _nbits_changed(self, new_value):
        self.check_pat_len()

    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
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

        t = self.t
        f = self.f
        w = self.w
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9
        Rs = self.rs
        Cs = self.cout * 1.0e-12
        RL = self.rin
        Cp = self.cin * 1.0e-12
        CL = self.cac * 1.0e-6

        ts = t[1]
        len_t = len(t)
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.use_ch_file:
            ch_s2p_pre = import_channel(self.ch_file, ts, self.f)
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
            H = exp(-l_ch * gamma)
            self.H = H
            # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
            tmp = np.array(list(zip(zip(zeros(len_f), H), zip(H, zeros(len_f)))))
            ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)
            # - And, finally, renormalize to driver impedance.
            ch_s2p_pre.renormalize(Rs)
        ch_s2p_pre.name = "ch_s2p_pre"
        self.ch_s2p_pre = ch_s2p_pre
        ch_s2p = ch_s2p_pre  # In case neither set of on-die S-parameters is being invoked, below.

        # Augment w/ IBIS-AMI on-die S-parameters, if appropriate.
        def add_ondie_s(s2p, ts4f, isRx=False):
            """Add the effect of on-die S-parameters to channel network.

            Args:
                s2p(skrf.Network): initial 2-port network.
                ts4f(string): on-die S-parameter file name.

            KeywordArgs:
                isRx(bool): True when Rx on-die S-params. are being added. (Default = False).

            Returns:
                skrf.Network: Resultant 2-port network.
            """
            ts4N = rf.Network(ts4f)  # Grab the 4-port single-ended on-die network.
            ntwk = sdd_21(ts4N)  # Convert it to a differential, 2-port network.
            ntwk2 = interp_s2p(ntwk, s2p.f)  # Interpolate to system freqs.
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
                fname = join(self._tx_ibis_dir, self._tx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
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
                fname = join(self._rx_ibis_dir, self._rx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"])[0])
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
                self.ts4N = ts4N
                self.ntwk = ntwk
        ch_s2p.name = "ch_s2p"
        self.ch_s2p = ch_s2p

        # Calculate channel impulse response.
        Zt = RL / (1 + 1j * w * RL * Cp)  # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:, 1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term
        # We take the transfer function, H, to be a ratio of voltages.
        # So, we must normalize our (now generalized) S-parameters.
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:, 1] / ch_s2p_term.z0[:, 0])
        chnl_h = irfft(chnl_H)
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        temp = chnl_h.copy()
        temp.resize(len(t), refcheck=False)
        chnl_trimmed_H = fft(temp)

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    def simulate(self, initial_run=False, update_plots=True):
        """Run all queued simulations."""
        # Running the simulation will fill in the required data structure.
        my_run_simulation(self, initial_run=initial_run, update_plots=update_plots)
        # Once the required data structure is filled in, we can create the plots.
        make_plots(self, n_dfe_taps=gNtaps)

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
        except Exception as exp:
            self.log("Failed to load configuration. See the console for more detail.")
            self.log(str(exp))

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
        except Exception as exp:
            self.log("Failed to save current user configuration. See the console for more detail.")
            self.log(str(exp))

    def load_results(self, filepath: Path):
        """Load results from a file into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData.load_from_file(filepath, self)
            self.data_file = filepath
            self.status = "Loaded results."
        except Exception as exp:
            self.log("Failed to load results from file. See the console for more detail.")
            self.log(str(exp))

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            PyBertData(self, time.asctime(), VERSION).save(filepath)
            self.data_file = filepath
            self.status = "Saved results."
        except Exception as exp:
            self.log("Failed to save results to file. See the console for more detail.")
            self.log(str(exp))

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
            make_plots(self, n_dfe_taps=gNtaps)

    def log_information(self):
        """Log the system information."""
        self.log(f"System: {platform.system()} {platform.release()}")
        self.log(f"Python Version: {platform.python_version()}")
        self.log(f"PyBERT Version: {VERSION}")
        self.log(f"PyAMI Version: {PyAMI_VERSION}")
        self.log(f"GUI Toolkit: {ETSConfig.toolkit}")
        self.log(f"Kiva Backend: {ETSConfig.kiva_backend}")
        # self.log(f"Pixel Scale: {self.trait_view().window.base_pixel_scale}")
