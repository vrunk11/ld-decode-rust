# The `.rfx` archive format, version 1

An `.rfx` file holds several captures ("tracks") of the same LaserDisc RF
signal, losslessly. A track is stored as the error of a predictor that sees the
track's own past and the samples of the tracks stored before it, so that the
disc's own noise, which every capture shares, is paid for once. Unpacking gives
back the original files bit for bit.

This document is enough to write a reader or a writer. Where the reference
implementation (`crates/rfx`) and this text disagree, the implementation is
right and the text has a bug. The format is at version 1 while it is under
development: nothing in version 1 is frozen yet.

There are two variants of the file. The **predictor variant** (sections 1-8) is
what `rfx pack` writes. The **FLAC variant** (section 9, `rfx pack --flac`) is
an ordinary Matroska audio file. A reader tells them apart by the EBML
DocType: `rfx` or `matroska`.

Conventions: integers in EBML elements are big endian, as EBML defines; the
varints of section 7 are LEB128 (7 bits per byte, low group first, high bit set
on every byte but the last); `u8`, `i32`, `f32` have their usual meaning; `f32`
arithmetic is IEEE 754 binary32 with round-to-nearest-even and no fused
operations; hashes are SHA-256, written as 64 lower-case hex digits.

## 1. Samples

A capture file is a sequence of **samples**, integers, by one of five formats
(the `--format` names of ld-decode):

| `fmt`  | bytes per group | samples per group | sample value |
|--------|-----------------|-------------------|--------------|
| `s16`  | 2               | 1                 | signed 16-bit, little endian |
| `r16`  | 2               | 1                 | unsigned 16-bit, little endian |
| `s8`   | 1               | 1                 | signed 8-bit |
| `r8`   | 1               | 1                 | unsigned 8-bit |
| `lds`  | 5               | 4                 | Domesday Duplicator packed 10-bit (below) |

`lds`: a group of five bytes `b0..b4` holds four 10-bit values
`t0 = b0<<2 | b1>>6`, `t1 = (b1&63)<<4 | b2>>4`, `t2 = (b2&15)<<6 | b3>>2`,
`t3 = (b3&3)<<8 | b4`. The sample is `(t - 512) << 6`: the 16-bit scale of the
Duplicator's `.s16`.

A file of `size` bytes has `words = size / bytes * samples` samples (integer
division). The `size - words/samples*bytes` bytes after the last whole group
are the **tail**; the archive keeps them in the metadata. The sample values
above, as 32-bit integers, are the **canonical samples** of the track. (For
`s8` they are -128..127, for `r8` 0..255, for `r16` 0..65535.)

Two numbers describe a track's samples: `shift`, the number of low bits that
are zero in every canonical sample (0 when all are zero; 6 for the
Duplicator), and `mean`, the rounded mean of `sample >> shift` over the track
(arithmetic shift, mean as a real number rounded to the nearest integer, ties
away from zero; 0 for an empty track). The predictor works on
`y = (sample >> shift) - mean` (arithmetic shift). Decoding gives
`sample = (y + mean) << shift`.

## 2. Container

The file is Matroska-shaped, written by hand, in this order:

1. **EBML header** (ID `1A45DFA3`) with `EBMLVersion` 1, `EBMLReadVersion` 1,
   `EBMLMaxIDLength` 4, `EBMLMaxSizeLength` 8, `DocType` `"rfx"`,
   `DocTypeVersion` 1, `DocTypeReadVersion` 1.
2. **Segment** (ID `18538067`) with the *unknown* size `01 FF FF FF FF FF FF FF`.
   Everything below is a child of it. A reader walks the children to the end of
   the file.
3. **Info** (ID `1549A966`): `TimestampScale` 1000000, `MuxingApp` and
   `WritingApp` `"rfx"`.
4. **Tracks** (ID `1654AE6B`) with `n + 2` TrackEntry elements (ID `AE`), each
   with `TrackNumber`, `TrackUID` (the same number), `TrackType` `0x21` (not a
   Matroska type: "private"), `CodecID` and `Name`:
   - track numbers `1..n` (the *data tracks*): `CodecID` `"D_RFX/PRED"`,
     `Name` the capture's file name;
   - number `n + 1`: `"D_RFX/META"`, name `"meta"`;
   - number `n + 2`: `"D_RFX/MAP"`, name `"map"`.
5. The **metadata cluster**, then the **map clusters** (section 4).
6. The **data clusters**: for each data track in order, one cluster per segment
   in increasing segment number (section 6).

Every **Cluster** (ID `1F43B675`) holds exactly two children: `Timestamp`
(ID `E7`) and one `SimpleBlock` (ID `A3`). The block data is the track number as
an EBML varint, two zero bytes (relative timestamp), the flag byte `0x80`, and
then the payload. The cluster's `Timestamp` is not a time: it is the segment
number (data tracks), the pair number (map track) or 0 (metadata). There is no
lacing, no Cues and no SeekHead. A reader locates a segment by scanning the
cluster headers once (each is a few bytes; the payloads need not be read) and
keeping, per `(track, timestamp)`, the offset and length of the block payload.

A generic Matroska tool lists the tracks and their blocks but cannot decode
them.

## 3. Metadata

The payload of the block in the cluster of track `n + 1` is one JSON object
(UTF-8, as `serde_json` writes it):

```
{ "format": "rfx", "version": 1, "segment": 262144, "sources": [ T0, T1, ... ] }
```

A reader refuses `format` other than `"rfx"` and `version` other than 1.
`segment` is the number of samples per segment (section 6). `sources` has one
object per data track, in track-number order (`sources[k]` is track `k + 1`;
the document and the commands count tracks from 0, with 0 the primary). Each
object has these fields; a field marked *default* may be missing and then has
the value given:

| field | type | meaning |
|-------|------|---------|
| `name` | string | file name of the capture (no directory) |
| `size` | integer | bytes of the original file |
| `sha256` | string | hash of the original file's bytes |
| `words` | integer | samples in the file |
| `tail` | array of 0-255 | the bytes past the last whole group |
| `fmt` | string | `s16`, `r16`, `s8`, `r8` or `lds` (default `s16`) |
| `sample_sha256` | string | hash of the canonical samples, each written as 4 bytes little-endian `i32` (default empty) |
| `shift`, `mean` | integer | section 1 |
| `own`, `solo`, `taps`, `ntop`, `ttaps`, `stride`, `every`, `lam_bits` | integer | predictor parameters, section 5 (`solo`, `ntop`, `ttaps` default 0) |
| `refs` | array of integers | the tracks this one is predicted from, each smaller than this track's own number; empty for a track predicted from its own past only |
| `weights` | array of numbers | one weight per entry of `refs` (they add up to 1), or empty for equal weights (default empty) |
| `position` | integer | where the capture stood on the command line when packed (default 0; informational) |
| `noise` | number or null | noise-against-signal estimate used to order the tracks (default null; informational) |
| `sidecars` | array | files that sat beside the capture, kept whole: `{ "name", "sha256", "data": [bytes] }` (default empty; the Duplicator's `.yaml`, at most 1 MiB) |
| `origin` | string or null | name of the compressed file (`.flac`, `.ldf`) the capture was decoded from, if it was (default null) |
| `rate_hz` | number | sample rate of the capture in Hz, from its `.yaml` (`"sample_rate_hz"`) or given when packing; 0 if unknown (default 0) |

Unknown fields are ignored. Tools that only want the metadata (`rfx info`,
`rfx info --yaml`, `rfx unpack --meta`) read this one block and nothing else.

## 4. The map

The map track holds the **lag tables**. Each map cluster (timestamp `0, 1, 2, ...`,
the pair's rank) carries `varint(k) varint(j)` (LEB128, track numbers from 0,
`j < k`) followed by the table of track `k` against reference `j` (section 7).
There is one pair for every `j` in `refs` of track `k`, in increasing order of
`(k, j)`. A track with no references has no table.

## 5. The predictor

The predictor estimates sample `i` of a segment from the samples before it and
from a **reference window**. Its weights come from a recursive weighted
least-squares fit that both the writer and the reader compute from the past
samples only: no weight is stored. The arithmetic is specified exactly, so
that every implementation gets the same bits.

### 5.1 Parameters

Per track (the metadata fields): `own`, `solo`, `taps`, `ntop`, `ttaps`,
`stride`, `every`, `lam_bits`. `stride` and `every` are powers of two. `lam =
f32::from_bits(lam_bits)`, the forgetting factor per covariance update. `solo`
0 means `solo = own`.

`use_ref` is true when `refs` is not empty. The own history length is `O = own`
with references and `O = solo` without. The regressor count is

```
d = O                                     without references
d = O + (2*taps + 1) + ntop*(2*ttaps + 1) with references
```

with `d <= 64`. `dp` is `d` rounded up to a power of two, at least 2.

The writer's levels, for reference (a reader does not need them):

| level | own | solo | plain | taps | ntop | ttaps | stride | every |
|-------|-----|------|-------|------|------|-------|--------|-------|
| 1 | 6 | 8 | 8 | 3 | 2 | 1 | 8 | 128 |
| 2 | 8 | 16 | 16 | 4 | 3 | 1 | 4 | 64 |
| 3 (default) | 8 | 32 | 16 | 6 | 3 | 1 | 1 | 16 |
| 4 | 12 | 48 | 24 | 6 | 4 | 1 | 1 | 16 |

with `lam = (1 - 1/1024)^stride`, computed in double precision and rounded to `f32`. A track predicted from the plain
mean of its references (no channels) stores `ntop` 0 and `own` = the `plain`
value; a track with channels stores `ntop` = min(level's `ntop`, number of
references) and `own` as in the table.

### 5.2 The reference window

For a segment starting at sample `s0` with `n` samples, the window covers the
target positions `p = s0 - taps ... s0 + n + taps - 1` (`len = n + 2*taps`
positions). It has `1 + ntop` **channels**, each `len` values, stored one after
the other with channel `c` at offset `c * l` (the layout is a convention of the
implementation: `l >= len`; the values are what matter).

For a reference track `j` of track `k`, and a target position `p`, the
**reference sample** is

```
x(j, p) = (S_j[p + lag] >> shift_j) - mean_j      if 0 <= p + lag < words_j
        = 0                                        otherwise
```

where `S_j` are the canonical samples of `j`, `shift_j` and `mean_j` are `j`'s,
and `lag` is the lag table's value at `p` (section 7; for `p < 0` the first
lag of the table). Reference samples are exact integers (below 2^24 in
magnitude), converted to `f32`.

- **Channel 0**, the weighted mean: `W[e]` for `e = 0..len-1` is the sum, over
  the entries `r = 0, 1, ...` of `refs` in order, of `w_r * x(refs[r], p)`,
  where `w_r` is `weights[r]` when `weights` has an entry per reference and
  `f32(1) / f32(number of refs)` otherwise. Each product is rounded to `f32`,
  then added to the running sum (starting from 0) in `f32`.
- **Channels `1..ntop`**: the `ntop` **top references**, taken from `refs`
  sorted by `weights`, heaviest first (a stable sort: equal weights keep their
  order in `refs`; with no `weights`, the first `ntop` of `refs`). Channel `c`
  is `x(top_c, p)` alone (weight 1).

### 5.3 Regressors and the prediction

At sample `i` of the segment (counted from 0 at the segment's first sample),
the regressor vector `r` is, in order:

1. the `O` previous samples of the segment, newest first (`y[i-1], y[i-2],
   ...`), the missing ones (before the segment's start) 0;
2. with references: the `2*taps + 1` values `W[i], W[i+1], ..., W[i + 2*taps]`
   of channel 0;
3. with references, for each top channel `c = 1..ntop`, the `2*ttaps + 1` values
   `C_c[i + taps - ttaps], ..., C_c[i + taps + ttaps]` (the middle of the same
   span).

then zeros up to `dp` entries. The weight vector `w` has `dp` entries, all 0 at
the start of a segment; entries past `d` stay 0. The prediction is

```
dot  = tree(r, w)            (below)
p_i  = floor(dot + 0.5)      in f32, then clamped to [-4194304, 4194304], as an i32
```

`tree` is a binary tree sum over `dp` leaves: the products `t[q] = r[q]*w[q]`
(each rounded) for `q = 0..dp-1`; then, with `h = dp/2`, `t[q] = t[q] + t[q+h]`
for `q < h`, then with `h/2`, and so on until `t[0]`. NaNs cannot arise from a
valid stream; a reader may treat them as a corrupt stream.

### 5.4 Learning

After sample `i` is known (`y`), with `r` the regressors that predicted it:

1. **Covariance update**, only when `i mod stride == 0` (`i & (stride-1) == 0`):
   for `p = 0..d-1`: `b[p] = lam*b[p] + r[p]*y`, and for `q = 0..p`:
   `a[p][q] = lam*a[p][q] + r[p]*r[q]`; every product is rounded to `f32`, then
   the sum. `a` (lower triangle, `d x d`) and `b` (`d` entries) start at 0 for
   each segment.
2. **Solve**, only when `i mod every == every - 1` and `i > 4*d`: solve
   `(a + ridge*I) x = b`, with `ridge = (trace(a)/d)*1e-6 + 1e-6` (in `f32`,
   in that order). If the matrix is not positive definite (a pivot
   `<= 0`), `w` keeps its old values; otherwise `w[0..d-1] = x`. Section 5.5.
3. **History**: the regressor vector's own part shifts by one (the old `r[0]`
   becomes `r[1]`, ...) and `r[0] = y`.

### 5.5 The solver

Cholesky factorisation `A = L L^T` of `A = a + ridge*I` (lower triangle), column
by column, then forward and backward substitution; every operation rounded to
`f32` on its own, in this order:

```
for j in 0..d-1:
    v = A[j][j]
    for k in 0..j-1:  v = v - L[j][k]*L[j][k]
    if v <= 0: fail
    L[j][j] = sqrt(v);  rd[j] = 1 / L[j][j]
    for i in j+1..d-1:
        x = A[i][j]
        for k in 0..j-1:  x = x - L[i][k]*L[j][k]
        L[i][j] = x * rd[j]
s = b
for i in 0..d-1:                    (forward)
    s[i] = s[i] * rd[i]
    for i2 in i+1..d-1:  s[i2] = s[i2] - L[i2][i]*s[i]
for i in d-1 down to 0:             (backward)
    s[i] = s[i] * rd[i]
    for i2 in 0..i-1:  s[i2] = s[i2] - L[i][i2]*s[i]
x = s
```

Each product is rounded before the subtraction.

## 6. The data tracks

Each data track is cut into **segments** of `segment` samples (the last one
shorter): segment `q` holds samples `q*segment ... min((q+1)*segment, words) - 1`.
There is one block, in a cluster with timestamp `q`, per segment. Segments are
independent: each starts from empty history, zero weights and a fresh
residual model, so they can be decoded in parallel and in any order, given the
reference tracks' samples.

The block payload is the range-coded residuals `e_i = y_i - p_i` of the
segment (section 8), one per sample, `n` of them. A decoder computes `p_i`,
reads `e_i`, forms `y_i = p_i + e_i`, learns `y_i` (section 5.4) and goes on.
Since the residual model (section 8) uses residuals only, a decoder can
parse all residuals of a segment before it computes a prediction.

To decode a segment of track `k`: decode its residuals; build the window from
the reference tracks (`refs`), which are tracks `< k` and so can be decoded
first; run the predictor; convert with `sample = (y + mean) << shift`. Writing
the samples in the track's format and appending `tail` gives the original file;
its SHA-256 is `sha256`.

## 7. Lag tables

A lag table is an integer **step function** of the target position: positions
`pos[0] = 0 < pos[1] < ... < pos[m-1]` and lags `lag[0..m-1]`; the lag at `p` is
`lag[i]` for the largest `i` with `pos[i] <= p` (`lag[0]` for `p < 0`).
Between two captures of one disc the lag drifts slowly (a sample every so many
thousand samples) and jumps where the capture skipped or repeated something.

Encoding (`m >= 1`):

```
varint(m)  varint(unit)  varint(pos[0] = 0)  varint(zigzag(lag[0]))  range-coded stream
```

`unit` is the greatest common divisor of the gaps `pos[i] - pos[i-1]` (1 when
`m = 1`). `zigzag(v) = (v << 1) ^ (v >> 63)` (64-bit). The range-coded stream
(section 8.1, a fresh encoder) holds, for `i = 1..m-1`, a **gap** and a **step**:
`gap = (pos[i] - pos[i-1]) / unit` and `step = lag[i] - lag[i-1]` (a decoder
adds `gap*unit` and `step`, with wrapping arithmetic). The model, a set of
adaptive bits (`P2`, section 8.1) that starts fresh with all probabilities at
one half, `prev_gap = 1`, `prev_mag = 0`, `prev_up = 0`, `prev_sign = 0`:

**Gap.** `d = gap - prev_gap` (signed), `mag = |d|`, `nb` = bit length of `mag`
(0 for 0), `c = min(bit length of prev_gap, 23)`.
- `nb` ones, each with `cls[c][j]` for `j = 0..nb-1`, then a zero with
  `cls[c][nb]` (omitted if `nb = 64`);
- if `nb >= 2`: the bit `(mag >> (nb-2)) & 1` with `mant[nb]`, then the
  `nb - 2` bits below it, raw (`direct`, high bit first; in pieces of at most
  32 bits);
- if `nb >= 1`: the sign (1 if `d < 0`) with `sign[prev_sign]`, and
  `prev_sign = 1 + sign`; if `nb = 0`, `prev_sign = 0`;
- `prev_gap = gap`.

**Step.** `mag = |step|`:
- if `1 <= mag <= 4`: bit 0 with `esc`; then `mag - 1` ones, each with
  `smag[prev_mag][j-1]` for `j = 1..mag-1`, then, if `mag < 4`, a zero with
  `smag[prev_mag][mag-1]`; then the direction (1 if `step > 0`) with
  `same[prev_up]`; `prev_up` = direction, `prev_mag = mag - 1`;
- otherwise: bit 1 with `esc`; `z = zigzag(step)`, then the bit length of `z` as
  7 raw bits and `z` itself in that many raw bits (high first).

Arrays: `cls` 24 x 65, `mant` 65, `sign` 3, `esc` 1, `same` 2, `smag` 4 x 3,
all of `P2`.

## 8. Range coder and residual model

### 8.1 Range coder

A carry-propagating range coder of the LZMA family, 32-bit range, `low` 64-bit.

Encoder: `low = 0`, `range = 0xFFFFFFFF`, `cache = 0`, `cache_size = 1`.
- `shift_low`: if `(low as u32) < 0xFF000000` or `low >> 32 != 0`: let `carry =
  low >> 32`; output `cache + carry` and then `cache_size - 1` bytes `0xFF +
  carry` (all mod 256); set `cache = (low >> 24) & 0xFF`, `cache_size = 0`.
  Then `cache_size += 1`, `low = (low & 0x00FFFFFF) << 8`.
- While `range < 2^24`: `range <<= 8`, `shift_low`.
- `bit(P2 p, b)`: `bound = (range >> 15) * p15(p)`; if `b = 0`: `range = bound`;
  else `low += bound`, `range -= bound`; then `p.learn(b)` and normalise.
- `direct(value, n)`: for each of the `n` bits, high first: `range >>= 1`; if the
  bit is 1, `low += range`; normalise.
- `finish`: five `shift_low`.

The first byte output is always 0. Decoder: skip that first byte, read the next
4 bytes into `code` (big endian), `range = 0xFFFFFFFF`; `bit`: `bound` as above;
if `code < bound`: `range = bound`, bit 0; else `code -= bound`, `range -=
bound`, bit 1; learn; normalise (`range <<= 8`, `code = (code << 8) | next byte`,
bytes past the end read as 0). `direct`: per bit `range >>= 1`; if `code >=
range`: `code -= range`, bit 1; normalise. A stream that needs more than 8
bytes past its end is corrupt.

**`P2`** is a pair of 16-bit estimates `(f, s)` of the probability of a zero
bit, both starting at 32768. `p15 = clamp((f + s) >> 2, 24, 32744)`. After a bit
`0`: `f += (65535 - f) >> 5`, `s += (65535 - s) >> 8`; after a bit `1`:
`f -= f >> 5`, `s -= s >> 8`.

### 8.2 Residual model

Every segment starts with a fresh model: all `P2` at their initial value and
`avg = 64`. For each residual `e` (an `i32`):

```
u = (e << 1) ^ (e >> 31)  (as u32);   v = u + 1;   nb = floor(log2(v))
bl = bit length of avg;   second = (avg >> (bl-2)) & 1 if bl >= 2 else 0
c  = min(2*bl + second, 47)
```

1. `nb` ones then a zero (`nb` is at most 31), the `j`-th with
   `unary[c][j]`;
2. if `nb >= 1`: `top = (v >> (nb-1)) & 1` with `m1[nb]`;
3. if `nb >= 2`: `(v >> (nb-2)) & 1` with `m2[nb][top]`;
4. if `nb >= 3`: the low `nb - 2` bits of `v`, raw (`direct`).

Then `avg = avg - (avg >> 4) + min(u, 2^20)`. Arrays: `unary` 48 x 33, `m1` 33,
`m2` 33 x 2, all `P2`. The decoder inverts this: the unary run gives `nb` (at
most 32), the bits rebuild `v`, `u = v - 1`, `e = (u >> 1) ^ -(u & 1)`.

The model reads past residuals only (through `avg`), never a prediction.

## 9. The FLAC variant

`rfx pack --flac` (off by default) replaces the predictor with FLAC, so that the
file is an ordinary Matroska audio file that any audio or video program opens.
There is no prediction between tracks, and the file is larger (about 28% of the
capture's size instead of 24% on the Duplicator's captures). The file is
written by ffmpeg's Matroska muxer; a reader needs only the following.

- **DocType** `matroska`. One audio track per capture, `CodecID` `A_FLAC`, in
  the order of the captures, mono, 16-bit, with the sample rate `rate_hz /
  1000` (an RF rate of 75 MHz is `75000` Hz; the FLAC header cannot say more),
  the track's `Name`/title the capture's name. The `CodecPrivate` is the FLAC
  stream header: `"fLaC"` and a STREAMINFO block (a reader accepts the header
  with or without the `fLaC` marker). Each block holds one FLAC frame. The
  frames of a stream have a fixed block size (STREAMINFO minimum = maximum);
  frame `f` holds samples `f*N` to `f*N + N - 1`, and the last frame is shorter.
- **Attachment** named `rfx.json` (MIME `application/json`): one JSON object
  `{ "format": "rfx", "codec": "flac", "version": 1, "tracks": [ ... ] }` with
  one entry per audio track, in order, in the schema of section 3 with the
  predictor fields (`own` ... `lam_bits`, `refs`, `weights`) zero or empty.
- **Samples.** The FLAC samples `s` (16-bit signed) relate to the canonical
  samples `v` of section 1 by: `s16`, `lds`: `s = v`; `s8`: `s = v * 256`; `r8`:
  `s = (v - 128) * 256`; `r16`: `s = v - 32768`. The original file is rebuilt
  as in section 6 (samples written in the track's format, then `tail`) and
  checked against `sha256`.

The predictor tools (`merge`, `extract`, `remove`, `add`) do not read this
variant; `unpack`, `info` and ld-decode do.

## 10. What a reader must check

- the DocType, `format`, `version`, and that `refs` of a track only name
  earlier tracks;
- that every segment of every track has a block, and that the lag tables
  needed exist (`(k, j)` for each `j` in `refs`);
- after decoding, the SHA-256 of the rebuilt file (or, when written in another
  format, `sample_sha256`).

`rfx pack` checks each segment by decoding it again (and the GPU path against
the CPU's bits) unless `--no-verify` is given.

## 11. Reference implementation

| part | file |
|------|------|
| container | `crates/rfx/src/ebml.rs` |
| metadata, packing, unpacking | `crates/rfx/src/pack.rs` |
| sample formats | `crates/rfx/src/fmt.rs` |
| predictor (CPU) | `crates/rfx/src/pred.rs` |
| predictor (OpenCL, same bits) | `crates/rfx/src/kernel.cl`, `gpu.rs`, `gpupipe.rs` |
| segments | `crates/rfx/src/seg.rs` |
| range coder, residual model | `crates/rfx/src/rc.rs`, `model.rs` |
| lag tables | `crates/rfx/src/align.rs` |
| random-access reader for ld-decode | `crates/rfx/src/reader.rs` |
| FLAC variant | `crates/rfx/src/flacmode.rs` |
