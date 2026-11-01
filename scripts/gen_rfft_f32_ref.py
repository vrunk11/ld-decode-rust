# Generates the single-precision real-FFT golden the PAL audio path needs
# (`crates/ld-decode/tests/data/rfft32768_f32_{in,out}.bin`).
#
# `core.py`'s `runfilter_audio_phase2` hands scipy the **float32** stage-1 audio
# buffer whenever the field has at least `blocklen` samples (every PAL field,
# no NTSC one), and `scipy.fft.fft` keeps its input's precision: the result is a
# complex64 half spectrum, i.e. a single-precision r2c plus the conjugate
# reflection. `ffi_ducc::rfft_f32` is that transform on the vendored ducc0.
#
# Run this with the bundled 7.4.0 interpreter, which carries the scipy the port
# targets (scipy 1.18.1, numpy 2.5.3):
#
#   cd <repo>
#   PYTHONHOME="$PWD/_ld-decode-7.4.0-release-ref/python" \
#   PYTHONPATH="$PWD/_ld-decode-7.4.0-release-ref" \
#     _ld-decode-7.4.0-release-ref/python/python.exe \
#     ld-decode-rust/scripts/gen_rfft_f32_ref.py
#
# Output: raw little-endian f32 input (32768), raw little-endian f32 complex64
# output (32768 bins, interleaved re, im).
import os

import numpy as np
import scipy.fft

N = 32768
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "crates", "ld-decode", "tests", "data")

i = np.arange(N, dtype=np.float64)
# The shape of a PAL stage-1 audio block: a few kHz of band-limited content plus
# a fractional term, so the f32 rounding of the transform is stressed rather
# than trivial.
x32 = (np.sin(i * 0.021) * 3.0e3 + np.cos(i * 0.0007) * 41.5).astype(np.float32)

h = scipy.fft.fft(x32)
assert h.dtype == np.complex64, h.dtype
assert h.shape == (N,), h.shape

x32.tofile(os.path.join(OUT, "rfft32768_f32_in.bin"))
h.view(np.float32).tofile(os.path.join(OUT, "rfft32768_f32_out.bin"))
print("wrote rfft32768_f32_in.bin / rfft32768_f32_out.bin", h.dtype, h.shape)
