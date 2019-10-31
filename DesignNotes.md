## Thoughts on how to proceed w/ development on `tx-rx-split` branch

Regarding the new v3.2.0 version of PyBERT (first one w/ new sub-tabbed layout), both David Patterson and I dislike the way the Tx/Rx configuration is currently split between the _Config._ and _Channel_ tabs.
The following are my thoughts on how to proceed.
We've created a branch: `tx-rx-split`, specifically for this work.

The obvious first impulse is to unify all Tx/Rx configuration, rather than have it split between two tabs.
However, I don't think it's quite that simple.
The problem is the Tx/Rx both have equalization-related *and* channel-related attributes.
For instance, the Tx output drive stage properties are intimately associated with the channel impulse response, while the Rx CTLE is closely associated w/ the CDR/DFE.
And the more I think about this, the more I believe that it makes sense to have the channel-related Tx/Rx attributes configured separately from the equalization-related attributes.
However, I think we may currently have the positions of those two sets flipped from what really makes sense.
I think I'd like to try the following and run it by David, to see what he thinks:

- Rename the *Channel* tab to *Equalization*.
- Swap the current positions of the Tx/Rx configuration items, placing the channel-related stuff on the Config. tab, and the equalization-related stuff on the (newly renamed) *Equalization* tab.
- Add the capability (on the *Config*. tab) to select an IBIS, or IBIS-AMI, model for either the Tx, or Rx, or both.
- If an IBIS model was selected, then use it to determine the impedance, etc. of the respective analog component, and don't make any changes to the *Equalization* tab.
- If an IBIS-AMI model was selected, then do as above, but also configure the *Equalization* tab to use the DLL/AMI files for equalization mdoeling of the Tx/Rx, as appropriate.

Also, I think David is correct: having a complete link picture somewhere in the GUI, which reflects the current configuration, would be very helpful.
