#! /usr/bin/env python

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.
"""
# from traits.trait_base import ETSConfig
# ETSConfig.toolkit = "qt4"
# ETSConfig.toolkit = "wx"

from datetime import datetime
from threading import Event, Thread
from time import sleep

from chaco.api import ArrayPlotData, GridPlotContainer
from numpy import array, convolve, cos, diff, exp, ones, pad, pi, real, resize, sinc, where, zeros
from numpy.fft import fft, ifft
from numpy.random import randint
from scipy.optimize import minimize, minimize_scalar
from traits.api import (
    HTML,
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
    Property,
    Range,
    String,
    cached_property,
    Trait,
)
from traitsui.message import message

from pyibisami.ami_parse import AMIParamConfigurator
from pyibisami.ami_model import AMIModel

from pybert import __version__ as VERSION
from pybert import __date__ as DATE
from pybert import __authors__ as AUTHORS
from pybert import __copy__ as COPY

from pybert.pybert_cntrl import my_run_simulation
from pybert.pybert_help import help_str
from pybert.pybert_plot import make_plots
from pybert.pybert_util import (
    calc_G,
    calc_gamma,
    import_channel,
    lfsr_bits,
    make_ctle,
    pulse_center,
    safe_log10,
    trim_impulse,
    draw_channel,
    submodules,
)
from pybert.pybert_view import traits_view

import pybert.solvers

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
gR0 = 1.452  # skin-effect resistance (Ohms/m)
gTheta0 = 0.02  # loss tangent
gZ0 = 100.0  # characteristic impedance in LC region (Ohms)
gv0 = 0.67  # relative propagation velocity (c)
gl_ch = 1.0  # cable length (m)
gRn = (
    0.001
)  # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)
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
gPeakMag = 10.0  # CTLE peaking magnitude (dB)
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
    """
    Thread class with a stop() method.

    The thread itself has to check regularly for the stopped() condition.

    All PyBERT thread classes are subclasses of this class.
    """

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = Event()

    def stop(self):
        """Called by thread invoker, when thread should be stopped prematurely."""
        self._stop_event.set()

    def stopped(self):
        """Should be called by thread (i.e. - subclass) periodically and, if this function
        returns True, thread should clean itself up and quit ASAP.
        """
        return self._stop_event.is_set()


class TxOptThread(StoppableThread):
    """Used to run Tx tap weight optimization in its own thread,
    in order to preserve GUI responsiveness.
    """

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
                    pybert.status = "Optimization failed: {}".format(res["message"])

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
    """Used to run Rx tap weight optimization in its own thread,
    in order to preserve GUI responsiveness.
    """

    def run(self):
        """Run the Rx equalization optimization thread."""

        pybert = self.pybert

        pybert.status = "Optimizing Rx..."
        max_iter = pybert.max_iter

        try:
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_opt_rx,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = "Optimization failed: {}".format(res["message"])

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
    """Used to run co-optimization in its own thread, in order to preserve GUI responsiveness."""

    def run(self):
        """Run the Tx/Rx equalization co-optimization thread."""

        pybert = self.pybert

        pybert.status = "Co-optimizing..."
        max_iter = pybert.max_iter

        try:
            if gDebugOptimize:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": True, "maxiter": max_iter},
                )
            else:
                res = minimize_scalar(
                    self.do_coopt,
                    bounds=(0, gMaxCTLEPeak),
                    method="Bounded",
                    options={"disp": False, "maxiter": max_iter},
                )

            if res["success"]:
                pybert.status = "Optimization succeeded."
            else:
                pybert.status = "Optimization failed: {}".format(res["message"])

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
            while pybert.tx_opt_thread and pybert.tx_opt_thread.isAlive():
                sleep(0.001)
            pybert._do_opt_tx(update_status=False)
            while pybert.tx_opt_thread and pybert.tx_opt_thread.isAlive():
                sleep(0.001)
        return pybert.cost


class ChnlSolveThread(StoppableThread):
    """Used to run custom channel cross-section solving in its own thread,
    in order to preserve GUI responsiveness."""

    def run(self):
        """Run the custom channel solver thread."""

        pybert = self.pybert
        pybert.status = "Solving custom channel..."

        try:
            res = self.do_solve()

            if res["success"]:
                pybert.status = "Channel solve succeeded."
            else:
                pybert.status = "Channel solve failed: {}".format(res["message"])

        except Exception as err:
            pybert.status = err

    def do_solve(self):
        """Run the selected solver."""
        sleep(0.001)  # Give the GUI a chance to update status, etc.

        res = {
            "success" : False,
            "message" : "Not yet run.",
        }
        pybert = self.pybert
        solver = pybert.solver_
        print(solver)
        print(type(solver))
        return res


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
        super(TxTapTuner, self).__init__()

        self.name = name
        self.enabled = enabled
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.steps = steps


class PyBERT(HasTraits):
    """
    A serial communication link bit error rate tester (BERT) simulator with a GUI interface.

    Useful for exploring the concepts of serial communication link design.
    """

    # Independent variables

    # - Simulation Control
    bit_rate = Range(low=0.1, high=120.0, value=gBitRate)  #: (Gbps)
    nbits = Range(low=1000, high=10000000, value=gNbits)  #: Number of bits to simulate.
    pattern_len = Range(low=7, high=10000000, value=gPatLen)  #: PRBS pattern length.
    nspb = Range(low=2, high=256, value=gNspb)  #: Signal vector samples per bit.
    eye_bits = Int(gNbits // 5)  #: # of bits used to form eye. (Default = last 20%)
    mod_type = List([0])  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
    num_sweeps = Int(1)  #: Number of sweeps to run.
    sweep_num = Int(1)
    sweep_aves = Int(gNumAve)
    do_sweep = Bool(False)  #: Run sweeps? (Default = False)
    debug = Bool(True)  #: Log extra info to console when true. (Default = False)

    # - Channel Control
    use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
    padded = Bool(False)  #: Zero pad imported Touchstone data? (Default = False)
    windowed = Bool(False)  #: Apply windowing to the Touchstone data? (Default = False)
    f_step = Float(10)  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
    ch_file = File(
        "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
    )  #: Channel file name.
    impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
    Rdc = Float(gRdc)  #: Channel d.c. resistance (Ohms/m).
    w0 = Float(gw0)  #: Channel transition frequency (rads./s).
    R0 = Float(gR0)  #: Channel skin effect resistance (Ohms/m).
    Theta0 = Float(gTheta0)  #: Channel loss tangent (unitless).
    Z0 = Float(gZ0)  #: Channel characteristic impedance, in LC region (Ohms).
    v0 = Float(gv0)  #: Channel relative propagation velocity (c).
    l_ch = Float(gl_ch)  #: Channel length (m).
    height     = Float(0.127)  #: Dielectric thickness (mm).
    width      = Float(0.254)  #: Trace width (mm).
    thickness  = Float(0.036)  #: Trace thickness (mm).
    separation = Float(0.508)  #: Trace separation (mm).
    roughness  = Float(0.005)  #: Average surface roughness (mm).
    solver     = Trait("None", {"None": None,})
    
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
    ctle_offset_tune = Float(gCTLEOffset)  #: EQ optimizer CTLE d.c. offset (dB).
    ctle_mode_tune = Enum(
        "Off", "Passive", "AGC", "Manual"
    )  #: EQ optimizer CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    use_dfe_tune = Bool(gUseDfe)  #: EQ optimizer DFE select (Bool).
    n_taps_tune = Int(gNtaps)  #: EQ optimizer # DFE taps.
    max_iter = Int(50)  #: EQ optimizer max. # of optimization iterations.
    tx_opt_thread = Instance(TxOptThread)  #: Tx EQ optimization thread.
    rx_opt_thread = Instance(RxOptThread)  #: Rx EQ optimization thread.
    coopt_thread = Instance(CoOptThread)  #: EQ co-optimization thread.
    chnl_solve_thread = Instance(ChnlSolveThread)  #: Custom channel solving thread.

    # - Tx
    vod = Float(gVod)  #: Tx differential output voltage (V)
    rs = Float(gRs)  #: Tx source impedance (Ohms)
    cout = Range(low=0.001, value=gCout)  #: Tx parasitic output capacitance (pF)
    pn_mag = Float(gPnMag)  #: Periodic noise magnitude (V).
    pn_freq = Float(gPnFreq)  #: Periodic noise frequency (MHz).
    rn = Float(gRn)  #: Standard deviation of Gaussian random noise (V).
    tx_taps = List(
        [
            TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
            TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
            TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
            TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
        ]
    )  #: List of TxTapTuner objects.
    rel_power = Float(1.0)  #: Tx power dissipation (W).
    tx_use_ami = Bool(False)  #: (Bool)
    tx_use_getwave = Bool(False)  #: (Bool)
    tx_has_getwave = Bool(False)  #: (Bool)
    tx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    tx_ami_valid = Bool(False)  #: (Bool)
    tx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    tx_dll_valid = Bool(False)  #: (Bool)

    # - Rx
    rin = Float(gRin)  #: Rx input impedance (Ohm)
    cin = Range(low=0.001, value=gCin)  #: Rx parasitic input capacitance (pF)
    cac = Float(gCac)  #: Rx a.c. coupling capacitance (uF)
    use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
    ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
    rx_bw = Float(gBW)  #: CTLE bandwidth (GHz).
    peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
    peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
    ctle_offset = Float(gCTLEOffset)  #: CTLE d.c. offset (dB)
    ctle_mode = Enum("Off", "Passive", "AGC", "Manual")  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
    rx_use_ami = Bool(False)  #: (Bool)
    rx_use_getwave = Bool(False)  #: (Bool)
    rx_has_getwave = Bool(False)  #: (Bool)
    rx_ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
    rx_ami_valid = Bool(False)  #: (Bool)
    rx_dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
    rx_dll_valid = Bool(False)  #: (Bool)

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
    drawdata = ArrayPlotData()
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
    ident = String(
        "PyBERT v{} - a serial communication link design tool, written in Python.\n\n \
    {}\n \
    {}\n\n \
    {};\n \
    All rights reserved World wide.".format(
            VERSION, AUTHORS, DATE, COPY
        )
    )

    # Help
    instructions = help_str

    # Console
    console_log = String("PyBERT Console Log\n\n")

    # Dependent variables
    # - Handled by the Traits/UI machinery. (Should only contain "low overhead" variables, which don't freeze the GUI noticeably.)
    #
    # - Note: Don't make properties, which have a high calculation overhead, dependencies of other properties!
    #         This will slow the GUI down noticeably.
    jitter_info = Property(HTML, depends_on=["jitter_perf"])
    perf_info = Property(HTML, depends_on=["total_perf"])
    status_str = Property(String, depends_on=["status"])
    sweep_info = Property(HTML, depends_on=["sweep_results"])
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
    bits = Property(Array, depends_on=["pattern_len", "nbits", "run_count"])
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
    btn_cfg_tx = Button(label="Configure")
    btn_cfg_rx = Button(label="Configure")
    btn_solve = Button(label="Solve")

    # Logger
    def log(self, msg):
        """Log a message to the console."""
        self.console_log += "\n[{}]: {}\n".format(datetime.now(), msg.strip())

    def dbg(self, msg):
        """Only if debug is true, log this message to the console."""
        if self.debug:
            print("\n[{}]: {}\n".format(datetime.now(), msg.strip()))
            self.log(msg)

    def handle_error(self, error):
        """If debug, raise else just prompt the user."""
        self.log(error)
        if self.debug:
            message("{}\nPlease, check terminal for more information.".format(error), "PyBERT Alert")
            raise error
        message(error, "PyBERT Alert")

    # Default initialization
    def __init__(self, run_simulation=True):
        """
        Initial plot setup occurs here.

        In order to populate the data structure we need to
        construct the plots, we must run the simulation.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(PyBERT, self).__init__()

        self.log("Started.")

        channel = draw_channel(self.height, self.width, self.thickness, self.separation)
        self.drawdata.set_data("channel", channel)

        slvr_dict = submodules(pybert.solvers)
        print("slvr_dict:", slvr_dict)
        tmp_dict = {"None": None}
        tmp_dict.update(slvr_dict)
        print("tmp_dict:", tmp_dict)
        # tmp_dict   = {"One": 1, "Two": 2}
        #tmp_dict   = {"One": 1}.update({"Two": 2})
        #print(tmp_dict)
        # solver     = Trait("None", tmp_dict)
        # self.solver = Trait("None", tmp_dict)
        # solver     = Trait("One", {"One": 1, "Two": 2})
        # solver     = Trait("None", {"None": None,})
        # self.solver = Trait("None", {"None": None,}.update(submodules(pybert.solvers)))
        # self.solver = Trait("One", {"One": 1, "Two": 2})
        self.remove_trait("solver")
        self.add_trait("solver", Trait("None", tmp_dict))
        print(self.solver)
        print(self.solver_)
        # solver_dict = {"None": None,}
        # solvers = pybert.solvers.__all__
        # names = map(lambda x : x.__name__, solvers)
        # mods = map(lambda x : x.__spec__, solvers)
        # solver_dict.update(dict(zip(names, mods)))
        # self.solver = Trait(solver_dict)

        if run_simulation:
            # Running the simulation will fill in the required data structure.
            my_run_simulation(self, initial_run=True)

            # Once the required data structure is filled in, we can create the plots.
            make_plots(self, n_dfe_taps=gNtaps)
        else:
            self.calc_chnl_h()  # Prevents missing attribute error in _get_ctle_out_h_tune().

    # Button handlers
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
            and self.tx_opt_thread.isAlive()
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
        if self.rx_opt_thread and self.rx_opt_thread.isAlive() or self.ctle_mode_tune == "Off":
            pass
        else:
            self.rx_opt_thread = RxOptThread()
            self.rx_opt_thread.pybert = self
            self.rx_opt_thread.start()

    def _btn_coopt_fired(self):
        if self.coopt_thread and self.coopt_thread.isAlive():
            pass
        else:
            self.coopt_thread = CoOptThread()
            self.coopt_thread.pybert = self
            self.coopt_thread.start()

    def _btn_abort_fired(self):
        if self.coopt_thread and self.coopt_thread.isAlive():
            self.coopt_thread.stop()
            self.coopt_thread.join(10)
        if self.tx_opt_thread and self.tx_opt_thread.isAlive():
            self.tx_opt_thread.stop()
            self.tx_opt_thread.join(10)
        if self.rx_opt_thread and self.rx_opt_thread.isAlive():
            self.rx_opt_thread.stop()
            self.rx_opt_thread.join(10)

    def _btn_cfg_tx_fired(self):
        self._tx_cfg()

    def _btn_cfg_rx_fired(self):
        self._rx_cfg()

    def _btn_solve_fired(self):
        if self.chnl_solve_thread and self.chnl_solve_thread.isAlive():
            pass
        else:
            self.chnl_solve_thread = ChnlSolveThread()
            self.chnl_solve_thread.pybert = self
            self.chnl_solve_thread.start()

    # Independent variable setting intercepts
    # (Primarily, for debugging.)
    def _set_ctle_peak_mag_tune(self, val):
        if val > gMaxCTLEPeak or val < 0.0:
            raise RuntimeError("CTLE peak magnitude out of range!")
        self.peak_mag_tune = val

    # Dependent variable definitions
    @cached_property
    def _get_t(self):
        """
        Calculate the system time vector, in seconds.

        """

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @cached_property
    def _get_t_ns(self):
        """
        Calculate the system time vector, in ns.
        """

        return self.t * 1.0e9

    @cached_property
    def _get_f(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in Hz.
        # (i.e. - [0, f0, 2 * f0, ... , fN] + [-(fN - f0), -(fN - 2 * f0), ... , -f0]
        """

        t = self.t

        npts = len(t)
        f0 = 1.0 / (t[1] * npts)
        half_npts = npts // 2

        return array([i * f0 for i in range(half_npts + 1)] + [(half_npts - i) * -f0 for i in range(1, half_npts)])

    @cached_property
    def _get_w(self):
        """
        Calculate the frequency vector appropriate for indexing non-shifted FFT output, in rads./sec.
        """

        return 2 * pi * self.f

    @cached_property
    def _get_bits(self):
        """
        Generate the bit stream.
        """

        pattern_len = self.pattern_len
        nbits = self.nbits
        mod_type = self.mod_type[0]

        bits = []
        seed = randint(128)
        while not seed:  # We don't want to seed our LFSR with zero.
            seed = randint(128)
        bit_gen = lfsr_bits([7, 6], seed)
        for _ in range(pattern_len - 4):
            bits.append(next(bit_gen))

        # The 4-bit prequels, below, are to ensure that the first zero crossing
        # in the actual slicer input signal occurs. This is necessary, because
        # we assume it does, when aligning the ideal and actual signals for
        # jitter calculation.
        #
        # We may want to talk to Mike Steinberger, of SiSoft, about his
        # correlation based approach to this alignment chore. It's
        # probably more robust.
        if mod_type == 1:  # Duo-binary precodes, using XOR.
            return resize(array([0, 0, 1, 0] + bits), nbits)
        return resize(array([0, 0, 1, 1] + bits), nbits)

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
        """
        Returns the number of unit intervals in the test vectors.
        """

        mod_type = self.mod_type[0]
        nbits = self.nbits

        nui = nbits
        if mod_type == 2:  # PAM-4
            nui /= 2

        return nui

    @cached_property
    def _get_nspui(self):
        """
        Returns the number of samples per unit interval.
        """

        mod_type = self.mod_type[0]
        nspb = self.nspb

        nspui = nspb
        if mod_type == 2:  # PAM-4
            nspui *= 2

        return nspui

    @cached_property
    def _get_eye_uis(self):
        """
        Returns the number of unit intervals to use for eye construction.
        """

        mod_type = self.mod_type[0]
        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if mod_type == 2:  # PAM-4
            eye_uis /= 2

        return eye_uis

    @cached_property
    def _get_ideal_h(self):
        """
        Returns the ideal link impulse response.
        """

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
        """
        Generate the symbol stream.
        """

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
        """
        Generate the Tx pre-emphasis FIR numerator.
        """

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
            info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (pj_chnl, pj_tx)
            info_str += "</TR>\n"
            info_str += '<TR align="right">\n'
            info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>\n' % (rj_chnl, rj_tx)
            info_str += "</TR>\n"
            info_str += "</TABLE>\n"

            info_str += "<H2>CTLE</H2>\n"
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
        except:
            info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"
            info_str += "Sorry, an error occurred.\n"
            raise

        return info_str

    @cached_property
    def _get_perf_info(self):
        info_str = "<H2>Performance by Component</H2>\n"
        info_str += '  <TABLE border="1">\n'
        info_str += '    <TR align="center">\n'
        info_str += "      <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>\n"
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Channel</TD><TD>%6.3f</TD>\n' % (self.channel_perf * 60.0e-6)
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Tx Preemphasis</TD><TD>%6.3f</TD>\n' % (self.tx_perf * 60.0e-6)
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">CTLE</TD><TD>%6.3f</TD>\n' % (self.ctle_perf * 60.0e-6)
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">DFE</TD><TD>%6.3f</TD>\n' % (self.dfe_perf * 60.0e-6)
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Jitter Analysis</TD><TD>%6.3f</TD>\n' % (self.jitter_perf * 60.0e-6)
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center"><strong>TOTAL</strong></TD><TD><strong>%6.3f</strong></TD>\n' % (
            self.total_perf * 60.0e-6
        )
        info_str += "    </TR>\n"
        info_str += '    <TR align="right">\n'
        info_str += '      <TD align="center">Plotting</TD><TD>%6.3f</TD>\n' % (self.plotting_perf * 60.0e-6)
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
            info_str += "      <TD>%+06.3f</TD><TD>%+06.3f</TD><TD>%d</TD><TD>%d</TD>\n" % (
                item[0],
                item[1],
                item[2],
                item[3],
            )
            info_str += "    </TR>\n"

        info_str += "  </TABLE>\n"

        return info_str

    @cached_property
    def _get_status_str(self):
        status_str = "%-20s | Perf. (Ms/m):    %4.1f" % (self.status, self.total_perf * 60.0e-6)
        dly_str = "         | ChnlDly (ns):    %5.3f" % (self.chnl_dly * 1.0e9)
        err_str = "         | BitErrs: %d" % self.bit_errs
        pwr_str = "         | TxPwr (W): %4.2f" % self.rel_power
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
        h = real(ifft(H))[:len_h]
        h *= abs(H[0]) / sum(h)

        return h

    @cached_property
    def _get_ctle_out_h_tune(self):
        chnl_h = self.chnl_h
        tx_h = self.tx_h_tune
        ctle_h = self.ctle_h_tune

        tx_out_h = convolve(tx_h, chnl_h)
        h = convolve(ctle_h, tx_out_h)

        return h

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

        self.plotdata.set_data("ctle_out_h_tune", p)
        self.plotdata.set_data("clocks_tune", clocks)

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
        for i in range(n_taps):
            err += p[clock_pos + (i + 1) * nspui] ** 2

        return err / p[clock_pos] ** 2

    # Changed property handlers.
    def _status_str_changed(self):
        if gDebugStatus:
            print(self.status_str)

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

    def _tx_ami_file_changed(self, new_value):
        try:
            self.tx_ami_valid = False
            with open(new_value) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log("Parsing Tx AMI file, '{}'...\n{}".format(new_value, pcfg.ami_parsing_errors))
            self.tx_has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self._tx_cfg = pcfg.open_gui
            self.tx_ami_valid = True
        except Exception as err:
            error_message = "Failed to open and/or parse AMI file!\n{}".format(err)
            self.handle_error(error_message)

    def _tx_dll_file_changed(self, new_value):
        try:
            self.tx_dll_valid = False
            model = AMIModel(str(new_value))
            self._tx_model = model
            self.tx_dll_valid = True
        except Exception as err:
            error_message = "Failed to open DLL/SO file!\n{}".format(err)
            self.handle_error(error_message)

    def _rx_ami_file_changed(self, new_value):
        try:
            self.rx_ami_valid = False
            with open(new_value) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log("Parsing Rx AMI file, '{}'...\n{}".format(new_value, pcfg.ami_parsing_errors))
            self.rx_has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self._rx_cfg = pcfg.open_gui
            self.rx_ami_valid = True
        except Exception as err:
            error_message = "Failed to open and/or parse AMI file!\n{}".format(err)
            self.handle_error(error_message)

    def _rx_dll_file_changed(self, new_value):
        try:
            self.rx_dll_valid = False
            model = AMIModel(str(new_value))
            self._rx_model = model
            self.rx_dll_valid = True
        except Exception as err:
            error_message = "Failed to open DLL/SO file!\n{}".format(err)
            self.handle_error(error_message)

    def _height_changed(self, new_value):
        channel = draw_channel(new_value, self.width, self.thickness, self.separation)
        self.drawdata.set_data("channel", channel)
        
    def _width_changed(self, new_value):
        channel = draw_channel(self.height, new_value, self.thickness, self.separation)
        self.drawdata.set_data("channel", channel)
        
    def _thickness_changed(self, new_value):
        channel = draw_channel(self.height, self.width, new_value, self.separation)
        self.drawdata.set_data("channel", channel)
        
    def _separation_changed(self, new_value):
        channel = draw_channel(self.height, self.width, self.thickness, new_value)
        self.drawdata.set_data("channel", channel)
        
    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    def calc_chnl_h(self):
        """
        Calculates the channel impulse response.

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
        ts = t[1]
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9

        if self.use_ch_file:
            chnl_h = import_channel(self.ch_file, ts, self.padded, self.windowed)
            if chnl_h[-1] > (max(chnl_h) / 2.0):  # step response?
                chnl_h = diff(chnl_h)  # impulse response is derivative of step response.
            chnl_h /= sum(chnl_h)  # Normalize d.c. to one.
            chnl_dly = t[where(chnl_h == max(chnl_h))[0][0]]
            chnl_h.resize(len(t))
            chnl_H = fft(chnl_h)
        else:
            l_ch = self.l_ch
            v0 = self.v0 * 3.0e8
            R0 = self.R0
            w0 = self.w0
            Rdc = self.Rdc
            Z0 = self.Z0
            Theta0 = self.Theta0
            w = self.w
            Rs = self.rs
            Cs = self.cout * 1.0e-12
            RL = self.rin
            Cp = self.cin * 1.0e-12
            CL = self.cac * 1.0e-6

            chnl_dly = l_ch / v0
            gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            H = exp(-l_ch * gamma)
            chnl_H = 2.0 * calc_G(H, Rs, Cs, Zc, RL, Cp, CL, w)  # Compensating for nominal /2 divider action.
            chnl_h = real(ifft(chnl_H))

        min_len = 10 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        chnl_h /= sum(chnl_h)  # a temporary crutch.
        temp = chnl_h.copy()
        temp.resize(len(t))
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


# So that we can be used in stand-alone, or imported, fashion.
def main():
    PyBERT().configure_traits(view=traits_view)


if __name__ == "__main__":
    main()
