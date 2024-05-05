"""
The Channel Operating Margin (COM) model, as per IEEE 802.3-22 Annex 93A.

Original author: David Banas <capn.freako@gmail.com>  

Original date:   February 29, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

Notes:
    1. Throughout this file, equation numbers refer to Annex 93A of the IEEE 802.3-22 standard.

ToDo:
    1. Provide type hints for imports.
"""

import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore
import skrf as rf  # type: ignore

from typing import TypeVar, TypeAlias, Any
from functools import cache
from traits.api import HasTraits, Property, Array, Float, cached_property  # type: ignore

from pybert.utility import import_s32p  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
COMParams: TypeAlias = dict[str, Any]  # ToDo: Specify this concretely, perhaps in `standards` module.
COMFiles: TypeAlias = str | list[str]

_pi: float  = np.pi
_2pi: float = 2 * np.pi

# Globals used by `calc_Hffe()`, to minimize the size of its cache.
# They are initialized by `COM.__init__()`.
gFreqs: Rvec = None  # type: ignore
gFb: float = None  # type: ignore
gC0min: float = None  # type: ignore
gNtaps: int = None  # type: ignore

T = TypeVar('T', Any, Any)
def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss([[T]]): The lists of candidates for each position in the final output.

    Returns:
        [[T]]: All possible combinations of input lists.
    """
    if not xss:
        return [[]]
    head, *tail = xss
    yss = all_combs(tail)
    return [[x] + ys for x in head for ys in yss]


# @cache
# def calc_Hffe(tap_weights: Rvec) -> Cvec:
def calc_Hffe(tap_weights: list[float]) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for the Tx FFE,
    according to (93A-21).

    Args:
        tap_weights: The vector of filter tap weights, excluding the cursor tap.

    Returns:
        The complex voltage transfer function, H(f), for the Tx FFE.

    Raises:
        RuntimeError: If the global variables above haven't been initialized.
        ValueError: If the length of the given tap weight vector is incorrect.

    Notes:
        1. This function has been (awkwardly) pulled outside of the
            `COM` class and made to use global variables, strictly for
            performance reasons.
            (Note that `@cached_property` decorated instance functions
            of `HasTraits` subclasses are not actually memoized, like
            `@cache` decorated ordinary functions are.)
            (It is used in the innermost layer of the nested loop structure
            used to find the optimal EQ solution. And its input argument
            is repeated often.)
            (See the `opt_eq()` method of the `COM` class.)
        2. Currently, a single post-cursor tap is assumed.

    ToDo:
        1. Remove the single post-cursor tap assumption.
    """

    assert gFreqs and gFb and gC0min and gNtaps, RuntimeError(
        "Called before global variables were initialized!")
    assert len(tap_weights) == gNtaps, ValueError(
        "Length of given tap weight vector is incorrect!")

    c0 = 1 - sum(list(map(abs, tap_weights)))
    if c0 < gC0min:
        return np.ones(len(gFreqs))
    else:
        cs = tap_weights
        cs.insert(-1, c0)  # Note the assumption of only one post-cursor tap!
        return sum(list(map( lambda n_c: n_c[1] * np.exp(-1j*_2pi*n_c[0]*gFreqs/gFb)
                           , enumerate(cs))))


class COM(HasTraits):
    """
    Encoding of the IEEE 802.3-22 Annex 93A "Channel Operating Margin"
    (COM) specification, as a Python class making use of the Enthought
    Traits/UI machinery, for both calculation efficiency and easy GUI display.
    """

    # Independent variable definitions
    ui = Float(100e-12)  # Unit interval (s).
    freqs = Array(np.arange(0, 40_010e6, 10e6))  # System frequencies (Hz).
    gDC = Float(0)  # D.C. gain of Rx CTLE first stage (dB).
    gDC2 = Float(0)  # D.C. gain of Rx CTLE first stage (dB).

    # Dependent variable definitions
    Xsinc = Property(Array, depends_on=["ui", "freqs"])
    @cached_property
    def _get_Xsinc(self):
        """Frequency domain sinc(f) corresponding to Rect(ui)."""
        ui = self.ui
        return ui * np.sinc(ui * self.freqs)

    Hr = Property(Array, depends_on=['freqs'])
    @cached_property
    def _get_Hr(self):
        """
        Return the voltage transfer function, H(f), of the Rx AFE,
        according to (93A-20).
        """
        f = self.freqs / (self.params['fr'] * self.params['fb'])
        return 1/(1 - 3.414214*f**2 + f**4 + 2.613126j*(f - f**3))

    Hctf = Property(Array, depends_on=['freqs', 'gDC', 'gDC2'])
    @cached_property
    def _get_Hctf(self):
        """
        Return the voltage transfer function, H(f), of the Rx CTLE,
        according to (93A-22).
        """
        f = self.freqs
        gDC = self.gDC
        gDC2 = self.gDC2
        num = (pow(10, gDC/20) + 1j*f/self.params['fz'])*(pow(10, gDC2/20) + 1j*f/self.params['fLF'])
        den = (1 + 1j*f/self.params['fp1'])/(1 + 1j*f/self.params['fp2'])/(1 + 1j*f/self.params['fLF'])
        return num / den


    # Reserved functions
    def __call__(self):
        """
        Calculate the COM value.
        """

        assert self.opt_eq(), RuntimeError("EQ optimization failed!")
        return 20 * np.log10(self.As / self.calc_noise)


    def __init__(self, params: COMParams, chnl_files: COMFiles,
        vic_chnl_ix: int = 4, num_ui: int = 100, gui: bool = True):
        """
        COM class initializer.

        Args:
            params: COM configuration parameters for desired standard.
                Note: Assumed immutable. ToDo: Can we encode this assumption, using type annotations?
            chnl_files: Touchstone file(s) representing channel, either:
                1. 8 s4p files: [victim, ], or
                2. 1 s32p file, according to VITA 68.2 convention.

        KeywordArgs:
            vic_chnl_ix: Victim channel index (from 1).
                Default: 4
            num_ui: Number of unit intervals to include in system time vector.
                Default: 100
            gui: Set to `False` for script/CLI based usage.
                Default: True
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        # Set default parameter values, as necessary.
        if 'zc' not in params:
            params['zc'] = 78.2
        if 'fLF' not in params:
            params['fLF'] = 1

        # Stash function parameters.
        self.params = params
        self.chnl_files = chnl_files
        self.gui = gui

        # Calculate intermediate variable values.
        ui = 1 / params['fb']
        sample_per = ui / params['M']
        N = self.params['N']
        fstep = self.params['fstep']
        trips = list(zip(self.params['tx_min'], self.params['tx_max'], self.params['tx_step']))
        self.tx_combs = all_combs(list(map(
            lambda trip: list(np.arange(trip[0], trip[1]+trip[2], trip[2]))
          , trips)))

        # Import channel Touchstone file(s).
        if chnl_files is str:
            (vic, aggs) = import_s32p(chnl_files, params['Av'], params['Afe'], params['Ane'], vic_chnl_ix)
            fmax = vic.f[-1]
            ntwks = [(vic, 'THRU')] + aggs  # `aggs` already annotated w/ type, 'NEXT' or 'FEXT'.
        else:  # ToDo: Do I have the ordering correct here?
            ntwks = []
            fmax = 1000e9
            n_files = len(chnl_files)
            n_lanes = params['N']
            assert n_files == 2 * n_lanes, ValueError(
                "Received {n_files} files for an {n_lanes} lane interface!")
            for n, chnl_file in enumerate(chnl_files):
                ntwk = rf.Network(chnl_file)
                fmax = min(fmax, ntwk.f[-1])
                if n > N:  # This one is a NEXT aggressor.
                    ntwks.append((ntwk, 'NEXT'))
                elif n > 0:  # This one is a FEXT aggressor.
                    ntwks.append((ntwk, 'FEXT'))
                else:  # This is the victim.
                    ntwks.append((ntwk, 'THRU'))

        # Calculate system time/frequency vectors.
        freqs = np.array([n * fstep for n in range((fmax + fstep) // fstep)])
        times = np.array([n * sample_per for n in range(num_ui * params['M'])])

        # Store calculated results.
        self.ui = ui
        self.sample_per = sample_per
        self.freqs = freqs
        self.times = times
        self.fmax = fmax
        self.ntwks = ntwks

        # Initialize global variables.
        gFreqs = freqs
        gFb = params['fb']
        gC0min = params['c0min']
        gNtaps = len(params['txffe_min'])


    # General functions
    def sC(self, c: float) -> rf.Network:
        """
        Return the 2-port network corresponding to a shunt capacitance,
        according to (93A-8).

        Args:
            c: Value of shunt capacitance (F).

        Returns:
            ntwk: 2-port network equivalent to shunt capacitance, calculated at given frequencies.

        Raises:
            None
        """

        r0 = self.params['r0']
        freqs = self.params['freqs']
        w = _2pi * freqs
        s = 1j * w
        s2p = 1/(2 + s*c*r0) * np.array(
            [ [-s*c*r0, 2]
            , [2, -s*c*r0]
            ])
        return rf.Network(s=s2p, f=freqs/1e9, z0=[2*r0, 2*r0])  # `f` is presumed to have units: GHz.


    def sZp(self, zp_opt: int = 1) -> rf.Network:
        """
        Return the 2-port network corresponding to a package transmission line,
        according to (93A-9:14).

        KeywordArgs:
            zp_opt: Package TL length option (from 1).
                Default: 1

        Returns:
            ntwk: 2-port network equivalent to package transmission line.

        Raises:
            None
        """

        n_zp_opts = len(self.params['zp'])
        assert zp_opt <= n_zp_opts, ValueError(
            f"Asked for zp option {zp_opt}, but there are only {n_zp_opts}!")
        zc = self.params['zc']
        r0 = self.params['r0']
        zp = self.params['zp'][zp_opt - 1]

        f_GHz  = self.freqs / 1e9
        a1     = 1.734e-3  # sqrt(ns)/mm
        a2     = 1.455e-4  # ns/mm
        tau    = 6.141e-3  # ns/mm
        gamma0 = 0         # 1/mm
        gamma1 = a1*(1 + 1j)
        gamma2 = a2*(1 - 1j*(2/_pi)*np.log(f_GHz)) + 1j*_2pi*tau
        rho    = (zc - 2*r0)/(zc + 2*r0)

        def gamma(f: float) -> complex:
            "Return complex propagation coefficient at frequency f (GHz)."
            if f==0:
                return gamma0
            else:
                return gamma0 + gamma1*np.sqrt(f) + gamma2*f

        g = np.array(list(map(gamma, f_GHz)))
        s11 = s22 = rho*(1 - np.exp(-g*2*zp))/(1 - rho**2*np.exp(-g*2*zp))
        s21 = s12 = (1 - rho**2)*np.exp(-g*zp)/(1 - rho**2*np.exp(-g*2*zp))
        s2p = np.array(
            [ [s11, s12]
            , [s21, s22]
            ])
        return rf.Network(s=s2p, f=f_GHz, z0=[2*r0, 2*r0])


    def sPkg(self, zp_opt: int = 1, isTx: bool = True) -> rf.Network:
        """
        Return the 2-port network corresponding to a complete package model,
        according to (93A-15:16).

        KeywordArgs:
            zp_opt: Package TL length option (from 1).
                Default: 1
            isTx: Requesting Tx package when True.
                Default: True

        Returns:
            ntwk: 2-port network equivalent to complete package model.

        Raises:
            None
        """

        sd = self.sC(self.params['cd'])
        sp = self.sC(self.params['cp'])
        sl = self.sZp(zp_opt)
        if isTx:
            return sd ** sl ** sp
        else:
            return sp ** sl ** sd


    def H21(self, s2p: rf.Network) -> Cvec:
        """
        Return the voltage transfer function, H21(f), of a terminated two
        port network, according to (93A-18).

        Args:
            s2p: Two port network of interest.

        Returns:
            Complex voltage transfer function at given frequencies.

        Raises:
            ValueError: If given network is not two port.
        """

        assert s2p.s.shape() == (2,2), ValueError("I can only convert 2-port networks.")
        g1 = self.gamma1
        g2 = self.gamma2
        dS = s2p.s11 * s2p.s22 - s2p.s12 * s2p.s21
        return (s2p.s21 * (1 - g1) * (1 + g2)) / (1 - s2p.s11*g1 - s2p.s22*g2 + g1*g2*dS)


    def H(self, s2p: rf.Network, tap_weights: Rvec) -> Cvec:
    # def H(self, s2p: rf.Network, tap_weights: list[float]) -> Cvec:
        """
        Return the voltage transfer function, H(f), of a complete COM signal path,
        according to (93A-19).

        Args:
            s2p: Two port network of interest.
            tap_weights: Tx FFE tap weights.

        Returns:
            Complex voltage transfer function of complete path.

        Raises:
            ValueError: If given network is not two port,
                or length of `tap_weights` is incorrect.

        Notes:
            1. Assumes `self.gDC` and `self.gDC2` have been set correctly.
        """

        assert s2p.s.shape() == (2,2), ValueError("I can only convert 2-port networks.")
        return calc_Hffe(list(tap_weights)) * self.H21(s2p) * self.Hr * self.Hctf


    def pulse_resp(self, H: Cvec) -> Rvec:
        """
        Return the unit pulse response, p(t), corresponding to the given
        voltage transfer function, H(f), according to (93A-24).

        Args:
            H: The voltage transfer function, H(f).
                Note: Possitive frequency components only.

        Returns:
            p: The pulse response corresponding to the given voltage transfer function.

        Raises:
            ValueError: If the length of the given voltage transfer
                function differs from that of the system frequency vector.
        """

        Xsinc = self.Xsinc
        assert len(H) == len(Xsinc), ValueError(
            "Length of given H(f) does not match length of f!")
        return np.fft.irfft(Xsinc * H)


    def gen_pulse_resps(self, tx_taps: Rvec) -> list[Rvec]:
        """
        Generate pulse responses for all neworks.

        Args:
            tx_taps: Desired Tx tap weights.

        Returns:
            List of pulse responses.

        Raises:
            None

        Notes:
            1. Assumes `self.gDC` and `self.gDC2` have been set correctly.
        """

        pulse_resps = []
        for ntwk, ntype in self.ntwks:
            pr = self.pulse_resp(self.H(ntwk, tx_taps))
            if ntype == 'THRU':
                pr *= self.params['Av']
            elif ntype == 'NEXT':
                pr *= self.params['Ane']
            else:
                pr *= self.params['Afe']
            pulse_resps.append(pr)
        return pulse_resps


    def opt_eq(self) -> bool:
        """
        Find the optimum values for the linear equalization parameters:
        c(-2), c(-1), c(1), gDC, and gDC2, as per IEEE 802.3-22 93A.1.6.
        """

        # Pull anything we use more than once below.
        L = self.params['L']
        M = self.params['M']
        freqs = self.freqs
        dfe_min = self.dfe_min
        dfe_max = self.dfe_max

        # Run the nested optimization loops.
        fom_max = 0
        for gDC2 in self.params['gDC2s']:
            self.gDC2 = gDC2
            for gDC in self.params['gDCs']:
                self.gDC = gDC
                for n, tx_taps in enumerate(self.tx_combs):
                    # Step a - Pulse response construction.
                    pulse_resps = self.gen_pulse_resps(np.array(tx_taps))
                    # Step b - Cursor identification.
                    vic_pulse_resp = np.array(pulse_resps[0])
                    vic_peak_loc = np.argmax(vic_pulse_resp)
                    valid_ixs = []
                    for ix in range(vic_peak_loc - M//2, vic_peak_loc + M//2):
                        # (93A-26)
                        if vic_pulse_resp[ix+M] / vic_pulse_resp[ix] < dfe_min[0]:
                            b_1 = dfe_min[0]
                        elif vic_pulse_resp[ix+M] / vic_pulse_resp[ix] > dfe_max[0]:
                            b_1 = dfe_max[0]
                        else:
                            b_1 = vic_pulse_resp[ix+M] / vic_pulse_resp[ix]
                        # (93A-25)
                        if vic_pulse_resp[ix-M] == vic_pulse_resp[ix+M] - b_1*vic_pulse_resp[ix]:
                            valid_ixs.append(ix)
                    assert valid_ixs, RuntimeError(
                        "No valid cursor found!")
                    if len(valid_ixs) > 1:
                        pre_pks = list(filter(lambda x: x <= vic_peak_loc, valid_ixs))
                        if pre_pks:
                            cursor_ix = pre_pks[-1]
                        else:
                            cursor_ix = valid_ixs[0]
                    else:
                        cursor_ix = valid_ixs[0]
                    # Step c - As.
                    vic_curs_val = vic_pulse_resp[cursor_ix]
                    As = self.params['Rlm'] * vic_curs_val / (L - 1)
                    # Step d - Tx noise.
                    varX = (L**2 - 1)/(3 * (L-1)**2)  # (93A-29)
                    varTx = vic_curs_val**2 * pow(10, -self.params['TxSNR']/10)  # (93A-30)
                    # Step e - ISI.
                    vic_pulse_resp_isi_samps = vic_pulse_resp[cursor_ix+M::M]
                    dfe_tap_weights = np.maximum(  # (93A-26)
                        self.params['dfe_min'],
                        np.minimum(
                            self.params['dfe_max'],
                            vic_pulse_resp_isi_samps / vic_curs_val)).resize(len(self.params['dfe_max']))
                    hISI = vic_pulse_resp_isi_samps \
                         - vic_curs_val * dfe_tap_weights.resize(len(vic_pulse_resp_isi_samps))  # (93A-27)
                    varISI = varX * sum(hISI**2)  # (93A-31)
                    # Step f - Jitter noise.
                    hJ = ( vic_pulse_resp[cursor_ix-1::M]
                         - vic_pulse_resp[cursor_ix+1::M]) / (2/M)  # (93A-28)
                    varJ = (self.params['Add']**2 + self.params['sigmaRj']**2) * varX * sum(hJ**2)  # (93A-32)
                    # Step g - Crosstalk.
                    varXT = 0
                    for pulse_resp in pulse_resps[1:]:  # (93A-34)
                        varXT += max([sum(np.array(pulse_resp[m::M])**2) for m in range(M)])  # (93A-33)
                    varXT *= varX
                    # Step h - Spectral noise.
                    df = freqs[1] - freqs[0]
                    varN = self.params['eta0'] * sum(abs(self.Hr * self.Hctf)**2) * df  # (93A-35)
                    # Step i - FOM calculation.
                    fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))  # (93A-36)
                    if fom > fom_max:
                        fom_max = fom
                        gDC2_best = gDC2
                        gDC_best = gDC
                        tx_taps_best = tx_taps
                        cursor_ix_best = cursor_ix

        # Check for error and save the best results.
        if gDC2_best is None or gDC_best is None or tx_taps_best is None or cursor_ix_best is None:
            return False
        self.fom_best = fom_max
        self.gDC2_best = gDC2_best
        self.gDC_best = gDC_best
        self.tx_taps_best = tx_taps_best
        self.cursor_ix_best = cursor_ix_best
        return True


    def calc_noise(self, npts: int = 2001) -> float:
        """
        Calculate the interference and noise for COM.

        KeywordArgs:
            npts: Number of vector points.
                Default: 200001 (0.01 mV resolution, as per recommendation in standard)
        """

        L = self.params['L']
        M = self.params['M']
        N = self.params['N']
        freqs = self.freqs

        y = np.linspace(-1, 1, npts)
        delta = np.zeros(npts)
        delta[npts//2] = 1
        varX = (L**2 - 1)/(3 * (L-1)**2)  # (93A-29)
        df = freqs[1] - freqs[0]

        def pn(hn: float) -> Rvec:
            return 1/L * sum([np.roll(delta, (2*l/(L-1) - 1) * hn) for l in range(L)])

        def p(h_samps: Rvec) -> Rvec:
            """
            Calculate the "deltas" for a set of pulse response samples.

            Args:
                h_samps: Vector of pulse response samples. (length N)

            Returns:
                Vector of "deltas" giving amplitude probability distribution.

            Raises:
                ValueError: If length of `h_samps` is incorrect.
            """

            assert len(h_samps) == N, ValueError("Input vector has wrong length!")

            rslt = delta
            for hn in h_samps:
                rslt = np.convolve(rslt, pn(hn), mode='same')
            return rslt

        self.gDC = self.gDC_best
        self.gDC2 = self.gDC2_best
        pulse_resps = self.gen_pulse_resps(np.array(self.tx_taps_best))
        cursor_ix = self.cursor_ix_best
        vic_pulse_resp = pulse_resps[0]
        vic_curs_val = vic_pulse_resp[cursor_ix]

        # Sec. 93A.1.7.2
        varN = self.params['eta0'] * sum(abs(self.Hr * self.Hctf)**2) * df  # (93A-35)
        varTx = vic_curs_val**2 * pow(10, -self.params['TxSNR']/10)  # (93A-30)
        hJ = ( vic_pulse_resp[cursor_ix-1::M]
             - vic_pulse_resp[cursor_ix+1::M]) / (2/M)  # (93A-28)
        varG = varTx + self.params['sigmaRj']**2*varX*sum(hJ**2) + varN  # (93A-41)
        pG = np.exp(-y**2/(2*varG)) / np.sqrt(_2pi*varG)  # (93A-42)
        pN = np.convolve(pG, p(self.params['Add']*hJ), mode='same')  # (93A-43)

        # Sec. 93A.1.7.3
        vic_pulse_resp_isi_samps = vic_pulse_resp[cursor_ix+M::M]
        dfe_tap_weights = np.maximum(  # (93A-26)
            self.params['dfe_min'],
            np.minimum(
                self.params['dfe_max'],
                vic_pulse_resp_isi_samps / vic_curs_val)).resize(len(self.params['dfe_max']))
        hISI = vic_pulse_resp_isi_samps \
             - vic_curs_val * dfe_tap_weights.resize(len(vic_pulse_resp_isi_samps))  # (93A-27)
        py = p(hISI)
        for pulse_resp in pulse_resps[1:]:  # (93A-44)
            i = np.argmax([sum(np.array(pulse_resp[m::M])**2) for m in range(M)])  # (93A-33)
            pk = p(pulse_resp[i::M])
            py = np.convolve(py, pk, mode='same')
        py = np.convolve(py, pN, mode='same')  # (93A-45)

        # Final calculation
        Py = np.cumsum(py)
        Py /= Py[-1]  # Enforce cumulative probability distribution.
        return abs(np.where(Py >= self.params['DER'])[0][0] - npts//2) * (2/(npts-1))

