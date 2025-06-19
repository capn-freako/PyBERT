"""
Python model of a Viterbi decoder.

Original author: David Banas <capn.freako@gmail.com>
Original date: June 12, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional

import numpy as np
import scipy as sp
from scipy.special import erf

from ..common       import TWOPI, Rvec
from ..utility.math import all_combs

SQRT2: float = np.sqrt(2.0)


class ViterbiDecoder():
    """
    Python class modeling a Viterbi decoder.
    """
    
    def __init__(self, L: int, N: int, M: int, sigma: float, pulse_resp_samps: Rvec):
        """
        Args:
            L: Number of symbol voltage levels.
            N: Number of symbols to track in state matrix.
            M: Number of pre-cursor symbols.
            sigma: Standard deviation of Gaussian voltage noise.
            pulse_resp_samps: Upstream channel pulse response samples, one per UI.
                Must agree with `N`/`M`!

        Notes:
            1. The symbol voltages are assumed uniformly distributed.
            2. The pulse response sample vector given must contain both:

                - the correct number of total samples, and
                - the correct number of pre-cursor samples.
        """

        # Validate input.
        if len(pulse_resp_samps) < N:
            raise ValueError(f"Length of `pulse_resp_samps` ({len(pulse_resp_samps)}) must be at least `N` ({N})")

        # Build normalized (to `pulse_resp_samps[0]`) symbol level voltages.
        symbol_level_values = [-1 + l * 2 / (L - 1) for l in range(L)]

        # Build state vectors, including their expected voltage observations.
        _states = all_combs([range(L)] * N)
        states = []
        for state in _states:
            expected_voltage = 0
            for n in range(N):
                expected_voltage += pulse_resp_samps[n] * symbol_level_values[state[-(n + 1)]]
            states.append((state, expected_voltage))

        # Build state transition probability matrix.
        num_states = len(states)
        trans = []
        for state in states:
            # row_vec = np.array([1 if state[0][1:] == states[m][0][0: -1] else 0
            row_vec = np.array([1 if state[0][1: -M] == states[m][0][0: -(1 + M)] else 0
                                    for m in range(num_states)])
            trans.append(row_vec / row_vec.sum())  # Enforce PMF.
            # print(f"{len(list(filter(lambda x: x != 0, row_vec)))}")

        # Build noise voltage interpolator.
        vs = np.linspace(-2, 2, 4_000)  # 1 mV precision
        v_prob = sp.interpolate.interp1d(
            vs, [np.exp(-(v**2) / (2 * sigma**2)) / np.sqrt(TWOPI * sigma**2) for v in vs])

        # Initialize private variables.
        self._states = states
        self._trans  = np.array(trans)
        self._sigma  = sigma
        self._state_curs_ix = -(1 + M)
        self._v_prob = v_prob

    @property
    def states(self):
        return self._states

    @property
    def trans(self):
        return self._trans

    @property
    def sigma(self):
        return self._sigma

    @property
    def state_curs_ix(self):
        return self._state_curs_ix

    @property
    def v_prob(self):
        return self._v_prob

    def decode(self, samps: Rvec, dbg_dict: Optional[dict[str, Any]] = None) -> list[int]:
        """
        Decode a sequence of observed voltages.

        Args:
            samps: Voltage samples from slicer input, one per UI.

        Keyword Args:
            dbg_dict: Debugging dictionary.
                Default: None

        Returns:
            List of symbol ordinates detected.

        Notes:
            1. Only those samples intended for eye diagram construction and BER prediction
            (i.e. - typically, post-DFE adaptation) should be sent as input in `samps`.
            This is because the Viterbi decoder is usually the worst performing block in a
            channel simulation.
            So, minimizing the length of its input is critical to overall performance.
        """

        states = self.states
        state_curs_ix = self.state_curs_ix
        num_states = len(states)
        first_prob = np.array([self.v_prob(samps[0] - expected_voltage)
                                   for (_, expected_voltage) in states])
        probs = [first_prob / first_prob.sum()]
        prevs = [np.arange(num_states)]
        print("Samples processed: ", end="")
        samps_per_star = len(samps) // 20
        for n, samp in enumerate(samps[1:]):
            if not (n + 1) % samps_per_star:
                print("*", end="")
            _prob = np.zeros(num_states)
            _prev = np.zeros(num_states, dtype=int)
            for r in range(num_states):
                new_probs = np.array(
                    [0 if self.trans[r][s] == 0
                       else probs[-1][r] * self.trans[r][s] * self.v_prob(samp - expected_voltage)
                         for s, (_, expected_voltage) in enumerate(states)])
                _prev = np.where(new_probs > _prob, [r] * num_states, _prev)
                _prob = np.maximum(new_probs, _prob)
            probs.append(_prob / _prob.sum())
            prevs.append(_prev)
        if dbg_dict is not None:
            dbg_dict["probs"] = probs
            dbg_dict["prevs"] = prevs.copy()
        path = [np.argmax(probs[-1])]
        prevs.reverse()
        for prev in prevs[: -1]:
            path.append(prev[path[-1]])
        path.reverse()
        if dbg_dict is not None:
            dbg_dict["path"] = path
        return list(map(lambda n: states[n][0][state_curs_ix:], path))
