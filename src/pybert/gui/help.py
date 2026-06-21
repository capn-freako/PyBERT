"""User instructions for PyBERT class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   April 15, 2015 (Copied from pybert.py.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""
help_str = """<H2>PyBERT User's Guide</H2>\n
  <H3>Note to developers</H3>\n
    This is NOT for you. Instead, open 'pybert/doc/_build/html/index.html' in a browser.\n
  <H3>PyBERT User Help Options</H3>\n
    <UL>\n
      <LI>Hover over any user-settable value in the <em>Config.</em> tab, for help message.</LI>\n
      <LI>Peruse the <em>General Tips</em> & <em>Help by Tab</em> sections, below.</LI>\n
      <LI>Visit the PyBERT FAQ at: https://github.com/capn-freako/PyBERT/wiki/pybert_faq.</LI>\n
      <LI>Send e-mail to David Banas at capn.freako@gmail.com.</LI>\n
    </UL>\n
  <H3>General Tips</H3>\n
    <H4>Main Window Status Bar</H4>\n
      The status bar, at the bottom of the window, gives the following information, from left to right:.<p>\n
      (Note: the individual pieces of information are separated by vertical bar, or 'pipe', characters.)\n
        <UL>\n
          <LI>Current state of, and/or activity engaged in by, the program.</LI>\n
          <LI>Simulator performance, in mega-samples per minute. A 'sample' corresponds to a single value in the signal vector being processed.</LI>\n
          <LI>The observed delay in the channel; can be used as a sanity check, if you know your channel.</LI>\n
          <LI>The number of bit errors detected in the last successful simulation run.</LI>\n
          <LI>The average power dissipated by the transmitter, assuming perfect matching to the channel ,no reflections, and a 50-Ohm system impedance.</LI>\n
          <LI>The jitter breakdown for the last run, taken at DFE output. (Parenthesized numbers are Dual-Dirac equivalents.))</LI>\n
        </UL>\n
  <H3>Help by Tab</H3>\n
    <H4>Config.</H4>\n
      This tab allows you to configure the simulation.\n
      Hover over any user configurable element for a help message.\n
"""
# +           "    <H4>DFE</H4>\n" \
# +           "    <H4>EQ Tune</H4>\n" \
# +           "    <H4>Impulses</H4>\n" \
# +           "    <H4>Steps</H4>\n" \
# +           "    <H4>Pulses</H4>\n" \
# +           "    <H4>Freq. Resp.</H4>\n" \
# +           "    <H4>Outputs</H4>\n" \
# +           "    <H4>Eyes</H4>\n" \
# +           "    <H4>Jitter Dist.</H4>\n" \
# +           "    <H4>Jitter Spec.</H4>\n" \
# +           "    <H4>Bathtubs</H4>\n" \
# +           "    <H4>Jitter Info</H4>\n" \
# +           "    <H4>Sweep Info</H4>\n" \
