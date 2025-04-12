"""HSPICE CSDF waveform file parser.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 7, 2022

Copyright (c) 2022 David Banas; all rights reserved World wide.
"""
import re
from functools import reduce
from parsec import count, generate, many, many1, none_of, regex, sepBy1, string  # type: ignore


class CSDF:  # pylint: disable=too-few-public-methods
    """Common Simulation Data Format (CSDF)"""

    def __init__(self, hdr, nms, wvs):
        if len(nms) != int(hdr["NODES"]):
            raise ValueError(f"Length of `nms` ({len(nms)}) must equal `hdr['NODES']` ({int(hdr['NODES'])})")
        self.header = {}
        self.header.update(hdr)
        self.names = nms
        self.waves = wvs


# CSDF grammar definition

# ignore cases.
whitespace = regex(r"\s+", re.MULTILINE)
# comment    = regex(r"^\*.*|\$.*")
# ignore     = many((whitespace | comment))
ignore = many(whitespace)


def lexeme(p):
    """Lexer for words."""
    return p << ignore  # skip all ignored characters.


def flag(ch):
    """Lexer for CSDF flags."""
    return lexeme(string("#") >> string(ch))


symbol = lexeme(regex(r"[a-zA-Z_]+"))
number = lexeme(regex(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"))
nat = lexeme(regex(r"[0-9]+"))
slash = lexeme(string("/"))


@generate("CSDF complex number")
def csdr_num():
    "Parse CSDF complex number."
    xs = yield sepBy1(number, slash)
    if len(xs) > 1:
        return float(xs[0]) + 1j * float(xs[1])
    return float(xs[0])


def val_samps(n):
    """Parser for CSDF value samples line."""
    return count(csdr_num, int(n))


csdr_str = lexeme(string("'") >> many(none_of("'")) << string("'"))
kv_pair = (symbol << string("=")) + csdr_str
header = flag("H") >> many(kv_pair)
sig_names = lexeme(flag("N") >> many(csdr_str))
wave_samps = lexeme((flag("C") >> number) + (nat.bind(val_samps)))


@generate("CSDF data")
def csdf_data():
    "Parse CSDF file contents."
    hdr = yield header
    nms = yield sig_names
    wvs = yield many1(wave_samps)
    return CSDF(
        dict(map(lambda pr: (pr[0], "".join(pr[1])), hdr)),
        # list(map(lambda cs: "".join(cs), nms)),
        list(reduce("".join, nms)),
        list(map(lambda pr: (float(pr[0]), pr[1]), wvs)),
    )
