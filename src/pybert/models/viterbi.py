"""
Python model of a Viterbi decoder.

Original author: David Banas <capn.freako@gmail.com>
Original date: June 12, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional

import numpy as np

from ..common       import TWOPI, Rvec
from ..utility.math import all_combs

class ViterbiDecoder():
    """
    Python class modeling a Viterbi decoder.
    """
    
    def __init__(self, L: int, N: int, sigma: float, pulse_resp_samps: Rvec):
        """
        Args:
            L: Number of symbol voltage levels.
            N: Number of symbols to track in state matrix.
            sigma: Standard deviation of Gaussian voltage noise.
            pulse_resp_samps: Upstream channel pulse response samples, one per UI.

        Notes:
            1. The symbol voltages are assumed uniformly distributed in: [-V, +V].
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
            row_vec = np.array([1 if state[0][1:] == states[m][0][0: -1] else 0
                                    for m in range(num_states)])
            trans.append(row_vec / row_vec.sum())  # Enforce PMF.

        # Initialize private variables.
        self._states = states
        self._trans  = np.array(trans)
        self._sigma  = sigma

    @property
    def states(self):
        return self._states

    @property
    def trans(self):
        return self._trans

    @property
    def sigma(self):
        return self._sigma

    def v_prob(self, x: float) -> float:
        # return 1 / x**2
        sigma = self.sigma
        return np.exp(-(x**2) / (2 * sigma**2)) / np.sqrt(TWOPI * sigma**2)

    def decode(self, samps: Rvec, dbg_dict: Optional[dict[str, Any]] = None) -> list[int]:
        """
        Decode a sequence of observed voltages.

        Args:
            samps: Voltage samples from slicer, one per UI.

        Keyword Args:
            dbg_dict: Debugging dictionary.
                Default: None
        """

        states = self.states
        num_states = len(states)
        first_prob = np.array([self.v_prob(samps[0] - expected_voltage) for (_, expected_voltage) in states])
        probs = [first_prob / first_prob.sum()]
        prevs = [np.array([-1] * num_states)]
        for samp in samps[1:]:
            _prob = np.zeros(num_states)
            _prev = np.zeros(num_states, dtype=int)
            for n in range(num_states):
                new_probs = np.array([
                    probs[-1][n] * self.trans[n][m] * self.v_prob(samp - expected_voltage)
                    for m, (_, expected_voltage) in enumerate(states)])
                _prev = np.where(new_probs > _prob, [n] * num_states, _prev)
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
        return list(map(lambda n: states[n][0][-1], path))
