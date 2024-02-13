# Raw data format (FPGA output)
Raw data is a sequence of 32-bit words. The following is reverse-engineered from the interpreter (last updated on 2024-02-12 looking at `interpreter.py` in Max's development branch).

## Types of words
Types of words and what they look like (in binary; `x` means any value).
```
TS MSB word:   0100 11xx xxxx xxxx xxxx xxxx xxxx xxxx
TS LSB word:   0100 10xx xxxx xxxx xxxx xxxx xxxx xxxx
TJMono word:   0100 0xxx xxxx xxxx xxxx xxxx xxxx xxxx
TDC word:      0010 xxxx xxxx xxxx xxxx xxxx xxxx xxxx
TLU word:      1xxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx
```
These types are mutually exclusive (no word can belong to more than one). A word can belong to none of the groups (if the first 4 bits are 0, 1, 3, 5, 6, or 7): in this case the interpreter will ignore the word.

## Timestamp (TS) words
```
   ┌─ Fixed part (identifies that this is a TS word)
   │
   │  ┌─ Flag bit (0 means lower/LSB bits, 1 means upper/MSB bits)
   │  │
   │  │    ┌─ Upper or lower 26 bits of the timestamp
   │  │    │
┌──┴─┐│┌───┴──────────────────────────┐
0100 1BTT TTTT TTTT TTTT TTTT TTTT TTTT
```
So the timestamp is made of 52 bits, counting clock cycles at 40 MHz (not synchronized with the TJMonopix clock for the default firmware, synchronous for Max's firmware that breaks the TLU), and is transmitted using _two_ words. This means that the timestamp rolls over every ~1303 days (so basically never).

In case of buffer overflow, it is possible to miss one of the words, resulting in a timestamp that is only partially updated (has only half of the bits right). Note that the lower 26 bits roll over every ~1.68s, which means that
 - in case the lower 26 bits are missed, we will be able to distinguish readout cycles only if they are separated by > 1.68s;
 - in case the upper 26 bites are missed, we will probably be able to distinguish readout cycles anyway (the probability that the lower 26 bits are exactly the same is ~15×10⁻⁹), but we will have the wrong absolute time if the cycles are separated by  > 1.68s.

It is also possible, again in case of buffer overflow, to miss both of the words: in this case the previous timestamp will be (erroneously) assigned to new hits.

## TJMono words
These are the words that contain the hit information from the Monopix2. Here things get complicated in an ungodly way.
```
  ┌─ Fixed part (identifies that this is a TJMono word)
  │
  │     ┌─ First 9-bit sub-word
  │     │
  │     │          ┌─ Second 9-bit sub-word
  │     │          │
  │     │          │          ┌─ Third 9-bit sub-word
  │     │          │          │
┌─┴──┐┌─┴───────┐┌─┴───────┐┌─┴───────┐
0100 0AAA AAAA AABB BBBB BBBC CCCC CCCC
```
Each raw data word (32 bits) contains _three_ 9-bit sub-words (I guess they are actual TJMonopix2 words straight from the serializer). Each of this sub-words needs to be interpreted _independently_ of its position in the 32-bit raw data word (the stream of 32-bit words is an envelope for a separate stream of 9-bit words).

These are the types of 9-bit sub-words (`x` means any value for the bit):
```
Start of frame:      1 1011 1100
End of frame:        1 0111 1100
Idle/nothing:        1 0011 1100
First hit sub-word:  x CCCC CCCC
Second hit sub-word: x LLLL LLLT
Third hit sub-word:  x tttt ttcR
Fourth hit sub-word: x rrrr rrrr
```
The info for each hit is given by _four_ successive sub-words as shown above. The hit information is retrieved as follows (note that LE and TE need to be converted from grey code afterwards):
```
Column number (9 bits):    C CCCC CCCc (from 1st and 3rd words)
Row number (9 bits):       R rrrr rrrr (from 3rd and 4th words)
Leading edge (7 bits):        LLL LLLL (from 2nd word)
Trailing edge (7 bits):       Ttt tttt (from 2nd and 3rd words)
```

Note that this can go wrong in many different ways.
 - The four hit sub-words are not identified by their bits, but only by their sequence: if we miss one (or more) 32-bit word(s), we might end up interpreting one sub-word as if it was another, and of another kind and/or for a different hit, resulting in meaningless information (which still makes sense though, since all hit information can never, by construction, go out of range).
 - Nothing is done to prevent reading hits after EOF. This allows to recover hits received after missing a SOF, but might result in inconsistent interpretation of the words as there is no way to ensure what place in the sequence each word occupies (see above).
 - The `token_id` is increased when an EOF (end of frame) is seen, but nothing is done for the case when a SOF is read after missing the previous EOF. I might make a commit to try and fix this.
 - The `token_id` is independent of the `timestamp` assignment: two hits are _for sure_ in different frames / readout cycles if they _either_ have different token ids, _or_ have different timestamps, _or both_. Still, if words are lost, two hits might appear in the same readout cycle without actually being so, and there is no way to tell if all of our hits make sense, or are just random garbage pieced together from other hits.
 - Finally note that, if we receive words with all zeros, it would appear as if we got an hit in (0,0) with LE=TE=ToT=0. This has actually happened before in Pisa on W8R6 when the LV supply had one bad contact and we were jumping pwell/psub directly to -6V without ramping. But it also happened (briefly) that we saw zeros only from information that comes from the matrix (everything expect CCCC CCCC) fr the same device under the same conditions.

## TDC words
```
  ┌─ Fixed part (identifies that this is a TDC word)
  │
  │    ┌─ 8-bit trigger distance
  │    │
  │    │         ┌─ 8-bit TDC timestamp
  │    │         │
  │    │         │         ┌─ 12-bit TDC value
  │    │         │         │
┌─┴┐ ┌─┴─────┐ ┌─┴─────┐ ┌─┴──────────┐
0010 DDDD DDDD TTTT TTTT VVVV VVVV VVVV
```
The interpreter makes a hit for each TLU word, where the column is 1022 (out of matrix range), the `row` and `le` are the trigger distance, the `token_id` is the TDC value, and the `timestamp` is the TDC timestamp. For the meaning of the three fields, ask [Max](mailto:Maximilian.Babeluk@oeaw.ac.at).

## TLU words
Depending on the trigger data format, a TLU word looks like this:
```
trigger_data_format = 0:  1NNN NNNN NNNN NNNN NNNN NNNN NNNN NNNN
trigger_data_format = 1:  1TTT TTTT TTTT TTTT TTTT TTTT TTTT TTTT
trigger_data_format = 2:  1TTT TTTT TTTT TTTT NNNN NNNN NNNN NNNN
```
where `NNN...` represents the trigger number and `TTT...` represents the trigger timestamp. How these are assigned and the units of the timestamp I do not know (but I suspect an external input is read and every leading or trailing edge increases the number and assigns a 40 MHz timestamp, asynchronous wrt Monopix).

The interpreter makes a hit for each TLU word, where the column is 1023 (out of matrix range), the `token_id` is the trigger number and the `timestamp` is the trigger timestamp.

## Possible improvements
 - Add  an output column (three-bits field) that says whether there was data corruption ongoing while reading the given hit.
    - One bit is set on hits coming after a missing SOF. This data might be garbage.
    - Another bit is set on hits coming after a change of timestamp without a corresponding change of token id. This also indicates a missing SOF, and might also allow to detecting this condition when the EOF is missed.
    - The third bit should be set on hit _before_ a missing EOF. This is detected when a SOF is found without the previous frame being ended, so it must be changed a posteriori. This bit does not necessarily mean that all of the marked hits are garbage (there might be just missed hits, or perhaps some hits came after the missed EOF and are corrupted, but those that come before are not).
    - The flag should be reset at each SOF.
 - When using frames / readout cycles for plotting, check for different token id _and_ different timestamp.

## Utility
```python
def p(x):
    s = f"{x:032b}"
    res = ""
    while s:
        c = s[-4:] if len(s) > 4 else s
        s = s[:-4] if len(s) > 4 else ""
        res = c + (" " if res else "") + res
    #print(f"{x:032b}")
    print(res)
```
