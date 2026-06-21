"""Unit tests for pybert.models.fec (FEC encoder / decoder round-trip)."""

import numpy as np
import pytest

from pybert.models.fec import FEC_Decoder, FEC_Encoder


class TestFECEncoder:
    def test_encode_returns_bit_pairs(self):
        enc = FEC_Encoder()
        result = enc.encode([0, 1, 0, 1, 1])
        assert len(result) == 5
        for pair in result:
            assert len(pair) == 2
            assert pair[0] in (0, 1)
            assert pair[1] in (0, 1)

    def test_step_returns_bit_pair(self):
        enc = FEC_Encoder()
        g0, g1 = enc.step(1)
        assert g0 in (0, 1)
        assert g1 in (0, 1)

    def test_state_advances(self):
        enc = FEC_Encoder()
        state_before = enc.state
        enc.step(1)
        assert enc.state != state_before

    def test_reset_state_is_deterministic(self):
        enc1 = FEC_Encoder(init_state=(0, 0, 0))
        enc2 = FEC_Encoder(init_state=(0, 0, 0))
        for b in [1, 0, 1, 1, 0]:
            assert enc1.step(b) == enc2.step(b)

    def test_nonzero_input_treated_as_one(self):
        enc1 = FEC_Encoder()
        enc2 = FEC_Encoder()
        assert enc1.step(1) == enc2.step(7)  # any non-zero is treated as 1


class TestFECDecoder:
    def _encode_and_decode(self, bits: list[int], trellis_depth: int = 5):
        encoder = FEC_Encoder()
        pairs = encoder.encode(bits)
        decoder = FEC_Decoder(trellis_depth)
        path = decoder.decode(pairs)
        return [decoder.states[ix][0] for ix in path], decoder

    def test_round_trip_all_zeros(self):
        bits = [0] * 40
        decoded, _ = self._encode_and_decode(bits)
        assert len(decoded) == len(bits)
        assert all(b == 0 for b in decoded)

    def test_round_trip_alternating(self):
        bits = [0, 1] * 20
        decoded, _ = self._encode_and_decode(bits)
        assert len(decoded) == len(bits)
        n_errors = sum(a != b for a, b in zip(bits, decoded))
        assert n_errors == 0, f"{n_errors} bit errors in noiseless round-trip"

    def test_round_trip_random(self):
        rng = np.random.default_rng(42)
        bits = list(rng.integers(0, 2, size=100))
        decoded, _ = self._encode_and_decode(bits, trellis_depth=10)
        assert len(decoded) == len(bits)
        n_errors = sum(a != b for a, b in zip(bits, decoded))
        assert n_errors == 0, f"{n_errors} bit errors in noiseless round-trip"

    def test_decoder_raises_on_shallow_trellis(self):
        with pytest.raises(ValueError, match="Minimum trellis depth"):
            FEC_Decoder(L=1)

    def test_decoder_min_trellis_depth(self):
        FEC_Decoder(L=2)  # should not raise
