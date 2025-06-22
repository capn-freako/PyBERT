"""
Python model of a Viterbi decoder.

Original author: David Banas <capn.freako@gmail.com>
Original date: June 12, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.

To use this module to construct your own Viterbi decoder, import the ``ViterbiDecoder`` class as follows:

.. code-block:: python

    from pybert.models.viterbi import ViterbiDecoder

and follow the example given by the ``ViterbiDecoder_ISI`` class definition, below.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeAlias, TypeVar

import numpy as np
import scipy as sp

from ..common       import TWOPI, Rvec, Rmat
from ..utility.math import all_combs

S = TypeVar('S')                # generic state type
X = TypeVar('X')                # generic observation type
SQRT2: float = np.sqrt(2.0)


class ViterbiDecoder(ABC, Generic[S, X]):
    """
    Abstract definition of a Viterbi decoder.
    """

    @property
    @abstractmethod
    def states(self) -> list[S]:
        """
        List of all possible states.
        """

    @property
    @abstractmethod
    def trans(self) -> Rmat:
        """
        State transition probability matrix.

        Notes:
            1. Row/column ordinates match those of `states`.
        """

    @property
    @abstractmethod
    def trellis(self) -> list[list[tuple[float, int]]]:
        """
        Current trellis matrix.

        Notes:
            1. Length of returned list gives trellis depth.
            2. Length of all inner lists should equal `len(states)`.
            3. Each location in the trellis matrix contains the
                probability and previous state index for the corresponding state.
        """

    @abstractmethod
    def prob(self, s: int, x: X) -> float:
        """
        Probability of state at index `s` given observation `x`.

        Notes:
            1. This is sometimes referred to as the "emission probability" in the literature.
        """

    @property
    def path(self) -> list[int]:
        """
        Maximum likelihood forward path through the trellis.

        Notes:
            1. First element in returned list corresponds to the time
            just before the first trellis column.
            2. The decided state of the final trellis column is *not* included.
        """

        trellis = self.trellis
        trellis_depth = len(trellis)

        # Starting with highest probability final state, backtrack through trellis.
        prevs = [trellis[-1][np.argmax(list(map(lambda pr: pr[0], trellis[-1])))][1]]
        for ix in range(2, trellis_depth + 1):
            prevs.append(trellis[-ix][prevs[-1]][1])
        prevs.reverse()
        return prevs

    def step_trellis(self, x: X, priming: bool = False) -> int:
        """
        Shift the trellis one column left, using the given observation sample.

        Args:
            x: The new observation sample.

        Keyword Args:
            priming: Don't perform backtrace when True.
                Default: False

        Returns:
            The decided state index of the exiting (i.e. - leftmost) column.
        """

        trellis = self.trellis
        num_states = len(trellis[-1])

        # Shift trellis contents one column left.
        for col in range(len(trellis) - 1):
            trellis[col] = trellis[col + 1]

        # Calculate maximum state probabilities, along w/ previous state, for new rightmost column.
        probs = np.zeros(num_states)
        prevs = np.array(num_states)
        for r in range(num_states):
            new_probs = np.array(
                [0 if self.trans[r][s] == 0
                 else trellis[-1][r][0] * self.trans[r][s] * self.prob(s, x)
                 for s in range(num_states)])
            prevs = np.where(new_probs > probs, [r] * num_states, prevs)
            probs = np.maximum(new_probs, probs)
        trellis[-1] = list(zip(probs / probs.sum(), prevs))

        prev = 0
        if not priming:
            prev = self.path[0]

        return prev

    def decode(self, samps: list[X], dbg_dict: Optional[dict[str, Any]] = None) -> list[int]:
        """
        Use trellis to decode a list of observations.

        Args:
            samps: List of observations.

        Keyword Args:
            dbg_dict: Dictionary for stashing debugging info.
                Default: None

        Returns:
            Maximum likelihood sequence estimation (MLSE) of state indices.
        """

        trellis = self.trellis
        trellis_depth = len(trellis)
        num_states = len(trellis[-1])

        # Prime the trellis.
        first_col = np.array([self.prob(s, samps[0]) for s in range(num_states)])
        first_col /= first_col.sum()
        trellis[-1] = list(zip(first_col, [0] * num_states))
        for x in samps[1: trellis_depth]:
            self.step_trellis(x, priming=True)

        # Run the remaining samples.
        states = []
        probs_prevs: list[list[tuple[float, int]]] = []
        for x in samps[trellis_depth:]:
            if dbg_dict is not None:
                probs_prevs.append(self.trellis[0])
            states.append(self.step_trellis(x))

        # Purge the trellis.
        states.extend(self.path[1:])
        states.append(int(np.argmax(list(map(lambda pr: pr[0], trellis[-1])))))
        if dbg_dict is not None:
            probs_prevs.extend(self.trellis[1:])

        # Fill in debugging dictionary if appropriate.
        if dbg_dict is not None:
            probs: list[list[float]] = []
            prevs: list[list[int]]   = []
            (probs, prevs) = zip(*list(map(lambda x: zip(*x), probs_prevs)))
            dbg_dict["probs"] = probs
            dbg_dict["prevs"] = prevs

        return states


# Following is an example of creating a concrete Viterbi decoder, using the abstract model above.
State_ISI: TypeAlias = tuple[list[int], float]  # list of symbol values, expected voltage


class ViterbiDecoder_ISI(ViterbiDecoder[State_ISI, float]):
    """
    Viterbi decoder using ISI to define observation probabilities.
    """

    # pylint: disable=too-many-locals
    def __init__(self, L: int, N: int, sigma: float, pulse_resp_samps: Rvec):
        """
        Args:
            L: Number of symbol voltage levels.
            N: Number of symbols per state.
            sigma: Standard deviation of Gaussian voltage noise (V).
            pulse_resp_samps: Upstream channel pulse response samples,
                one per UI, beginning with cursor (V).
                (Must have length >= `N`!)

        Notes:
            1. The symbol voltages are assumed uniformly distributed.
            (This will require modification for photonics!)
        """

        # Validate input.
        if len(pulse_resp_samps) < N:
            raise ValueError(f"Length of `pulse_resp_samps` ({len(pulse_resp_samps)}) must be at least `N` ({N})")

        # Build normalized (to `pulse_resp_samps[0]`) symbol level voltages.
        symbol_level_values = [-1 + v * 2 / (L - 1) for v in range(L)]

        # Build state vectors, including their expected voltage observations.
        _states = all_combs([list(range(L))] * N)
        states = []
        for s in _states:
            expected_voltage = 0
            for n in range(N):
                expected_voltage += pulse_resp_samps[n] * symbol_level_values[s[-(n + 1)]]
            states.append((s, expected_voltage))

        # Build state transition probability matrix.
        num_states = len(states)
        trans = []
        for state in states:
            row_vec = np.array([1 if state[0][1:] == states[m][0][0: -1] else 0
                                for m in range(num_states)])
            trans.append(row_vec / row_vec.sum())  # Enforce PMF.

        # Build noise voltage interpolator.
        vs = np.linspace(-2, 2, 4_000)  # 1 mV precision
        v_prob = sp.interpolate.interp1d(
            vs, [1e-3 * np.exp(-(v**2) / (2 * sigma**2)) / np.sqrt(TWOPI * sigma**2) for v in vs])

        # Build initial trellis.
        trellis = [[(1 / num_states, 0)] * num_states] * N

        # Initialize private variables.
        self._states = states
        self._trans  = np.array(trans)
        self._sigma  = sigma
        self._v_prob = v_prob
        self._trellis = trellis

    @property
    def states(self):
        return self._states

    @property
    def trans(self):
        return self._trans

    @property
    def trellis(self):
        return self._trellis

    @property
    def sigma(self):  # pylint: disable=missing-function-docstring
        return self._sigma

    @property
    def v_prob(self):  # pylint: disable=missing-function-docstring
        return self._v_prob

    def prob(self, s: int, x: float) -> float:
        """
        Probability of state at index `s` given observation `x`.
        """
        return self.v_prob(x - self.states[s][1])
