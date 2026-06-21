"""
Python model of a FEC encoder/decoder.

Original author: David Banas <capn.freako@gmail.com>

Original date: August 31, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from typing import TypeAlias, TypeVar

import numpy as np

from ..common               import Rvec
from ..models.viterbi       import ViterbiDecoder
from ..utility.math         import all_combs

S = TypeVar('S')                # generic state type
X = TypeVar('X')                # generic observation type
SQRT2: float = np.sqrt(2.0)


class FEC_Encoder():
    """
    Model of a simple FEC encoder, ala TI DN504
    """

    def __init__(self, init_state: tuple[int, int, int] = (0, 0, 0)):
        self._state = tuple(map(lambda x: x % 2, init_state))

    def step(self, x: int) -> tuple[int, int]:
        """
        Step the encoder, using the given next input bit.

        Args:
            x: Next input bit. (`x` /= 0 => "1")

        Returns:
            Next output bit pair.

        Notes:
            1. Integers in output pair will be bound to: [0,1].
            2. Function calculates output before shifting state register.
        """

        if x:
            x = 1
        state = self._state
        rslt = ((x + state[0] + state[1] + state[2]) % 2,  # g0
                (x +            state[1] + state[2]) % 2)  # g1
        self._state = (x, state[0], state[1])
        return rslt

    def encode(self, bits: list[int]) -> list[tuple[int, int]]:
        """
        Encode a list of bits.

        Args:
            bits: List of bits to encode.

        Returns:
            List of resultant bit pairs.
        """

        gbits: list[tuple[int, int]] = []
        for bit in bits:
            gbits.append(self.step(bit))

        return gbits

    @property
    def state(self):
        """Current state of decoder."""
        return self._state


# Note: `int` is used for simplicity, but is expected to be bound to: [0,1], in all cases.
Delay3Tap: TypeAlias = list[int]        # current input, previous input, etc.
BitPair:   TypeAlias = tuple[int, int]  # lsb, msb


class FEC_Decoder(ViterbiDecoder[Delay3Tap, BitPair]):
    """
    Viterbi decoder with 3-tap delay chain for state and pair of bits for observations.
    """

    def __init__(self, L: int):
        """
        Args:
            L: Trellis depth.
        """

        # Validate input.
        if L < 2:
            raise ValueError("Minimum trellis depth is 2!")

        # Build state vectors, along with their expected observations.
        states = all_combs([[0, 1],] * 4)
        expecteds = list(map(lambda s: ((s[0] + s[1] + s[2] + s[3]) % 2,   # g0
                                        (s[0]        + s[2] + s[3]) % 2),  # g1
                             states))

        # Build state transition probability matrix.
        num_states = len(states)
        trans = []
        for state in states:
            row_vec = np.array([1 if state[:-1] == states[m][1:] else 0
                                for m in range(num_states)])
            trans.append(row_vec / row_vec.sum())  # Enforce PMF.

        # Build initial trellis.
        trellis = [[(1 / num_states, 0)] * num_states] * L

        # Calculate "emission" probabilities.
        probs: list[list[Rvec]] = [
            [np.zeros(num_states), np.zeros(num_states)],
            [np.zeros(num_states), np.zeros(num_states)]]
        for g0 in range(2):
            for g1 in range(2):
                pvec = np.array(list(map(lambda gs: float(2 - (abs(g0 - gs[0]) + abs(g1 - gs[1]))), expecteds)))  # pylint: disable=cell-var-from-loop
                pvec /= pvec.sum()  # Enforce PMF.
                probs[g0][g1] = pvec.copy()

        # Initialize private variables.
        self._states = states
        self._expecteds = expecteds
        self._trans  = np.array(trans)
        self._trellis = trellis
        self._probs = probs

    def prob(self, s: int, x: BitPair) -> float:
        """
        Probability of state at index ``s`` given observation ``x``.

        Notes:
            1. This is sometimes referred to as the "emission probability" in the literature.
        """

        g0, g1 = x
        return self._probs[g0][g1][s]
