//! OpenCL demodulator: [`demod_block_cpu`](super::demodblock::demod_block_cpu)
//! for a whole batch of blocks at once, in single precision.
//!
//! Same signal path and the same filters as the CPU kernel (see the header of
//! `demodblock.rs`); the FFTs are a radix-8 Stockham transform with twiddles
//! tabulated in double precision. It is not bit-identical to the reference
//! (nothing on a GPU is), so it is an option, checked against the CPU kernel
//! with `LD_GPU_CHECK`.
//!
//! Precision: the FM discriminator's output is a frequency near 8 MHz that
//! moves ~7 kHz per IRE. Carried as is in single precision, that constant
//! would leave only ~1 Hz of resolution and the post-demod FFTs would round
//! at that scale. So the discriminator writes `f - c` (c = the 50 IRE
//! frequency), the clamp and the post-demod filters run on that, and
//! `c * H(0)` is added back to each filtered channel: the filters are linear,
//! so it is the same quantity, now resolved to ~0.1 Hz.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use rustfft::num_complex::Complex64;

use super::demodblock::{audio_stage1, BlockDecode, DemodSpecRef, VideoChannels};
use crate::opencl::{Arg, Buffer, Gpu, Kernel, Queue};

/// Blocks per GPU submission (bounds the device memory: ~10 buffers of
/// `MAX_BATCH * blocklen` complex values).
const MAX_BATCH: usize = 64;

const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float2 cmul(float2 a, float2 b) { return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline float2 cmulf(float2 a, float2 b) { return cmul(a, b); }

// Real samples -> complex rows.
__kernel void load_real(__global const float* in, __global float2* out) {
    int i = get_global_id(0), b = get_global_id(1);
    if (i >= N) return;
    out[b * N + i] = (float2)(in[b * N + i], 0.0f);
}

// v * exp(s * i * pi/4 * k) for k = 1, 2, 3 (s = -1 forward, +1 inverse).
#define R2 0.70710678118654752f
inline float2 w8_1(float2 v, float s) { return (float2)(R2 * (v.x - s * v.y), R2 * (v.y + s * v.x)); }
inline float2 w8_2(float2 v, float s) { return (float2)(-s * v.y, s * v.x); }
inline float2 w8_3(float2 v, float s) { return (float2)(R2 * (-v.x - s * v.y), R2 * (-v.y + s * v.x)); }

// One radix-8 Stockham pass (Govindaraju et al.): sub-transform length Ns.
// tw[m] = exp(-2 pi i m / N), tabulated in double precision.
__kernel void fft_pass(__global const float2* in, __global float2* out, __global const float2* tw,
                       const int Ns, const float s) {
    int j = get_global_id(0), b = get_global_id(1);
    if (j >= N / 8) return;
    __global const float2* x = in + (size_t)b * N;
    __global float2* y = out + (size_t)b * N;
    int k = j % Ns;
    float2 v[8];
    for (int r = 0; r < 8; r++) v[r] = x[j + r * (N / 8)];
    if (Ns > 1) {
        int step = N / (Ns * 8);
        for (int r = 1; r < 8; r++) {
            float2 t = tw[(k * r * step) & (N - 1)];
            if (s > 0.0f) t.y = -t.y;
            v[r] = cmul(v[r], t);
        }
    }
    // DFT-8: radix-2 split into two DFT-4.
    float2 a0 = v[0] + v[4], a4 = v[0] - v[4];
    float2 a1 = v[1] + v[5], a5 = w8_1(v[1] - v[5], s);
    float2 a2 = v[2] + v[6], a6 = w8_2(v[2] - v[6], s);
    float2 a3 = v[3] + v[7], a7 = w8_3(v[3] - v[7], s);
    float2 c0 = a0 + a2, c2 = a0 - a2, c1 = a1 + a3, c3 = w8_2(a1 - a3, s);
    float2 d0 = a4 + a6, d2 = a4 - a6, d1 = a5 + a7, d3 = w8_2(a5 - a7, s);
    float2 o[8];
    o[0] = c0 + c1; o[4] = c0 - c1; o[2] = c2 + c3; o[6] = c2 - c3;
    o[1] = d0 + d1; o[5] = d0 - d1; o[3] = d2 + d3; o[7] = d2 - d3;
    int idx = (j / Ns) * Ns * 8 + k;
    for (int r = 0; r < 8; r++) y[idx + r * Ns] = o[r];
}

// Hermitian full spectrum of x[:N/2+1] * f (what irfft inverts: the imaginary
// parts of the DC and Nyquist bins are dropped).
inline float2 herm(__global const float2* x, __global const float2* f, int k) {
    if (k == 0 || k == N / 2) return (float2)(cmul(x[k], f[k]).x, 0.0f);
    if (k < N / 2) return cmul(x[k], f[k]);
    float2 v = cmul(x[N - k], f[N - k]);
    return (float2)(v.x, -v.y);
}

// The block spectrum's three products: RF video (* MTF power), EFM, RF highpass.
__kernel void rf_products(__global const float2* X, __global const float2* rfvideo, __global const float2* mtfpow,
                          const int use_pow, __global const float2* fefm, __global const float2* frfhpf,
                          __global float2* H, __global float2* E, __global float2* R) {
    int k = get_global_id(0), b = get_global_id(1);
    if (k >= N) return;
    __global const float2* x = X + (size_t)b * N;
    size_t o = (size_t)b * N + k;
    float2 h = cmul(x[k], rfvideo[k]);
    if (use_pow) h = cmul(h, mtfpow[k]);
    H[o] = h;
    E[o] = cmul(x[k], fefm[k]);
    R[o] = herm(x, frfhpf, k);
}

// FM discriminator, minus c; `raw` keeps the unclamped value, `D` the clamped
// one as a complex row for the post-demod FFT.
__kernel void discriminate(__global const float2* h, __global float* raw, __global float2* D,
                           const double scale, const double c, const float lo, const float hi) {
    int n = get_global_id(0), b = get_global_id(1);
    if (n >= N) return;
    size_t o = (size_t)b * N + n;
    float v;
    if (n == 0) {
        v = (float)(-c);
    } else {
        float2 z = h[o], w = h[o - 1];
        float re = z.x * w.x + z.y * w.y, im = z.y * w.x - z.x * w.y;
        double d = (double)atan2(im, re);
        if (d < 0.0) d += 6.283185307179586;
        v = (float)(d * scale - c);
    }
    raw[o] = v;
    D[o] = (float2)(clamp(v, lo, hi), 0.0f);
}

// The three post-demod filters as Hermitian spectra.
__kernel void video_products(__global const float2* Y, __global const float2* f0, __global const float2* f1,
                             __global const float2* f2, __global float2* V0, __global float2* V1, __global float2* V2) {
    int k = get_global_id(0), b = get_global_id(1);
    if (k >= N) return;
    __global const float2* y = Y + (size_t)b * N;
    size_t o = (size_t)b * N + k;
    V0[o] = herm(y, f0, k);
    V1[o] = herm(y, f1, k);
    V2[o] = herm(y, f2, k);
}

// Cut and pack every output channel of the batch.
__kernel void extract(__global const float2* V0, __global const float2* V1, __global const float2* V2,
                      __global const float* raw, __global const float2* R, __global const float2* E,
                      const int cstart, const int clen, const int rstart, const int rlen,
                      const float add0, const float add1, const float add2, const float c,
                      __global float* o_demod, __global float* o_05, __global float* o_burst,
                      __global float* o_raw, __global float* o_rfhpf, __global short* o_efm) {
    int j = get_global_id(0), b = get_global_id(1);
    const float inv = 1.0f / (float)N;
    size_t base = (size_t)b * N;
    if (j < clen) {
        size_t i = base + cstart + j, oo = (size_t)b * clen + j;
        o_demod[oo] = V0[i].x * inv + add0;
        o_05[oo] = V1[i].x * inv + add1;
        o_burst[oo] = V2[i].x * inv + add2;
        o_raw[oo] = raw[i] + c;
        o_efm[oo] = convert_short_rtz(clamp(E[i].x * inv, -32768.0f, 32767.0f));
    }
    if (j < rlen) {
        o_rfhpf[(size_t)b * rlen + j] = R[base + rstart + j].x * inv;
    }
}

// Spectrum bins the CPU's analog-audio stage reads: per block, `n` bins from
// each of the four starts in `starts`.
__kernel void gather_bins(__global const float2* X, __global const int* starts, const int n, __global float2* out) {
    int j = get_global_id(0), b = get_global_id(1);
    if (j >= 4 * n) return;
    int run = j / n, i = j % n;
    out[(size_t)b * 4 * n + j] = X[(size_t)b * N + min(starts[run] + i, N - 1)];
}
"#;

/// Per-decoder GPU state: its queue, kernels and buffers.
struct State {
    q: Queue,
    k_load: Kernel,
    k_fft: Kernel,
    k_rf: Kernel,
    k_disc: Kernel,
    k_video: Kernel,
    k_extract: Kernel,
    k_bins: Kernel,
    tw: Buffer,
    rfvideo: Buffer,
    mtfpow: Buffer,
    fefm: Buffer,
    frfhpf: Buffer,
    fvideo: [Buffer; 3],
    starts: Buffer,
    /// Work rows: input/scratch, spectrum, H, E, R, raw, D, V0..V2 are carved
    /// from these as the pipeline goes.
    rows: [Buffer; 6],
    raw: Buffer,
    out_f: [Buffer; 5],
    out_efm: Buffer,
    bins: Buffer,
    /// MTF power uploaded last (its level), to skip re-uploading.
    pow_level: Option<f64>,
}

pub(crate) struct GpuDemod {
    blocklen: usize,
    /// DC gain of the three post-demod filters (to add `c * H(0)` back).
    fvideo_dc: [f64; 3],
    state: Mutex<State>,
}

fn c32(v: &[Complex64]) -> Vec<[f32; 2]> {
    v.iter().map(|c| [c.re as f32, c.im as f32]).collect()
}

impl GpuDemod {
    pub(crate) fn new(spec: &DemodSpecRef) -> Result<Self> {
        let n = spec.blocklen;
        anyhow::ensure!(n.is_power_of_two() && n.trailing_zeros() % 3 == 0, "GPU demod needs blocklen = 8^k (got {n})");
        let gpu = Gpu::get()?;
        let mut ks = gpu
            .kernels(
                SOURCE,
                &format!("-DN={n}"),
                &["load_real", "fft_pass", "rf_products", "discriminate", "video_products", "extract", "gather_bins"],
            )?
            .into_iter();
        let mut k = || ks.next().unwrap();
        let (k_load, k_fft, k_rf, k_disc, k_video, k_extract, k_bins) = (k(), k(), k(), k(), k(), k(), k());
        let q = gpu.queue()?;
        let f = spec.filters;
        let upload = |data: &[[f32; 2]]| -> Result<Buffer> {
            let b = gpu.buffer(std::mem::size_of_val(data))?;
            q.write(&b, 0, data)?;
            Ok(b)
        };
        let tw: Vec<[f32; 2]> = (0..n)
            .map(|m| {
                let a = -std::f64::consts::TAU * m as f64 / n as f64;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        let state = State {
            tw: upload(&tw)?,
            rfvideo: upload(&c32(&f.rfvideo))?,
            mtfpow: gpu.buffer(n * 8)?,
            fefm: upload(&c32(&f.fefm))?,
            frfhpf: upload(&c32(&f.frfhpf_rfft))?,
            fvideo: [
                upload(&c32(&f.fvideo_rfft[0]))?,
                upload(&c32(&f.fvideo_rfft[1]))?,
                upload(&c32(&f.fvideo_rfft[2]))?,
            ],
            starts: gpu.buffer(16)?,
            rows: [gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?, gpu.buffer(8)?],
            raw: gpu.buffer(4)?,
            out_f: [gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?, gpu.buffer(4)?],
            out_efm: gpu.buffer(2)?,
            bins: gpu.buffer(8)?,
            pow_level: None,
            q,
            k_load,
            k_fft,
            k_rf,
            k_disc,
            k_video,
            k_extract,
            k_bins,
        };
        let fvideo_dc = [f.fvideo_rfft[0][0].re, f.fvideo_rfft[1][0].re, f.fvideo_rfft[2][0].re];
        Ok(Self { blocklen: n, fvideo_dc, state: Mutex::new(state) })
    }

    /// Demodulate `blocks` (each `blocklen` samples), cut as the CPU kernel
    /// does with `cut = true`.
    pub(crate) fn demod(
        &self,
        blocks: &[&[f32]],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
    ) -> Result<Vec<BlockDecode>> {
        let mut out = Vec::with_capacity(blocks.len());
        for chunk in blocks.chunks(MAX_BATCH) {
            out.extend(self.batch(chunk, mtf_level, mtf_pow, rotdelay, spec)?);
        }
        Ok(out)
    }

    fn batch(
        &self,
        blocks: &[&[f32]],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
    ) -> Result<Vec<BlockDecode>> {
        let n = self.blocklen;
        let nb = blocks.len();
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let row_bytes = nb * n * 8;
        for r in &mut st.rows {
            r.ensure(row_bytes)?;
        }
        st.raw.ensure(nb * n * 4)?;
        let q = &st.q;

        // Input, as complex rows (rows[0] = staging, then rows[1] = X).
        let mut flat = Vec::with_capacity(nb * n);
        for b in blocks {
            flat.extend_from_slice(&b[..n]);
        }
        q.write(&st.raw, 0, &flat)?;
        st.k_load.args(&[Arg::Buf(&st.raw), Arg::Buf(&st.rows[0])])?;
        q.launch(&st.k_load, &[n, nb], 256)?;
        // FFT rows[0] -> lands in rows[1] after 5 passes (odd count).
        let fft = |src: usize, tmp: usize, sign: f32| -> Result<usize> {
            let (mut a, mut b) = (src, tmp);
            let mut ns = 1usize;
            while ns < n {
                st.k_fft.args(&[Arg::Buf(&st.rows[a]), Arg::Buf(&st.rows[b]), Arg::Buf(&st.tw), Arg::I32(ns as i32), Arg::F32(sign)])?;
                q.launch(&st.k_fft, &[n / 8, nb], 256)?;
                std::mem::swap(&mut a, &mut b);
                ns *= 8;
            }
            Ok(a)
        };
        let x = fft(0, 1, -1.0)?;
        let free = if x == 1 { 0 } else { 1 };

        // MTF power, uploaded when its level changed.
        let use_pow = mtf_pow.is_some();
        if let Some(p) = mtf_pow {
            if st.pow_level != Some(mtf_level) {
                q.write(&st.mtfpow, 0, &c32(p))?;
                st.pow_level = Some(mtf_level);
            }
        }
        // H -> rows[2], E -> rows[3], R -> rows[4].
        st.k_rf.args(&[
            Arg::Buf(&st.rows[x]),
            Arg::Buf(&st.rfvideo),
            Arg::Buf(&st.mtfpow),
            Arg::I32(use_pow as i32),
            Arg::Buf(&st.fefm),
            Arg::Buf(&st.frfhpf),
            Arg::Buf(&st.rows[2]),
            Arg::Buf(&st.rows[3]),
            Arg::Buf(&st.rows[4]),
        ])?;
        q.launch(&st.k_rf, &[n, nb], 256)?;

        // Analog-audio bins, gathered from the block spectrum before it goes.
        let audio_starts: Vec<i32> = (0..2)
            .flat_map(|ch| {
                let af = &spec.filters.audio[ch];
                let half = af.nbins / 2;
                [af.lowbin as i32, (n - af.lowbin - half) as i32]
            })
            .collect();
        let half_bins: Vec<usize> = (0..2).map(|ch| spec.filters.audio[ch].nbins / 2).collect();
        let run = half_bins[0].max(half_bins[1]);
        st.bins.ensure(nb * 4 * run * 8)?;
        q.write(&st.starts, 0, &audio_starts)?;
        st.k_bins.args(&[Arg::Buf(&st.rows[x]), Arg::Buf(&st.starts), Arg::I32(run as i32), Arg::Buf(&st.bins)])?;
        q.launch(&st.k_bins, &[4 * run, nb], 64)?;

        // Inverse transforms: H (hilbert), E (EFM), R (RF highpass). Each lands
        // in the free row after 5 passes; move the roles accordingly.
        let h = fft(2, x, 1.0)?; // x's row is dead now
        let hilbert = h;
        let spare = if h == 2 { x } else { 2 };
        let e = fft(3, spare, 1.0)?;
        let spare2 = if e == 3 { spare } else { 3 };
        let r = fft(4, spare2, 1.0)?;
        let spare3 = if r == 4 { spare2 } else { 4 };
        let _ = free;

        // Discriminator: raw -> st.raw, clamped D -> rows[spare3].
        let c = spec.levels.iretohz(50.0);
        let scale = spec.freq_hz / std::f64::consts::TAU;
        let lo = (1_500_000.0 - c) as f32;
        let hi = (spec.freq_hz * 0.75 - c) as f32;
        st.k_disc.args(&[
            Arg::Buf(&st.rows[hilbert]),
            Arg::Buf(&st.raw),
            Arg::Buf(&st.rows[spare3]),
            Arg::F64(scale),
            Arg::F64(c),
            Arg::F32(lo),
            Arg::F32(hi),
        ])?;
        q.launch(&st.k_disc, &[n, nb], 256)?;
        // The hilbert row is free from here; the post-demod FFT uses it.
        let y = fft(spare3, hilbert, -1.0)?;
        let other = if y == spare3 { hilbert } else { spare3 };
        // Free rows for V0..V2: `other`, and the two of {0..5} not holding
        // E, R or Y.
        let used = [e, r, y, other];
        let mut vrows = vec![other];
        vrows.extend((0..6).filter(|i| !used.contains(i)).take(2));
        st.k_video.args(&[
            Arg::Buf(&st.rows[y]),
            Arg::Buf(&st.fvideo[0]),
            Arg::Buf(&st.fvideo[1]),
            Arg::Buf(&st.fvideo[2]),
            Arg::Buf(&st.rows[vrows[0]]),
            Arg::Buf(&st.rows[vrows[1]]),
            Arg::Buf(&st.rows[vrows[2]]),
        ])?;
        q.launch(&st.k_video, &[n, nb], 256)?;
        // Inverse of each video row, with Y's row as the ping-pong partner.
        let mut v_out = [0usize; 3];
        let mut tmp = y;
        for i in 0..3 {
            let o = fft(vrows[i], tmp, 1.0)?;
            v_out[i] = o;
            tmp = if o == vrows[i] { tmp } else { vrows[i] };
        }

        // Cut ranges, as the CPU kernel.
        let cstart = spec.blockcut.min(n);
        let cend = n.saturating_sub(spec.blockcut_end);
        let clen = cend.saturating_sub(cstart);
        let rstart = (spec.blockcut as i64 - rotdelay).clamp(0, n as i64) as usize;
        let rstop_raw = -((spec.blockcut_end as i64) + rotdelay);
        let rstop = if rstop_raw < 0 { (n as i64 + rstop_raw).clamp(0, n as i64) } else { rstop_raw.clamp(0, n as i64) } as usize;
        let rlen = rstop.saturating_sub(rstart);
        for (i, b) in st.out_f.iter_mut().enumerate() {
            b.ensure(nb * if i == 4 { rlen } else { clen } * 4)?;
        }
        st.out_efm.ensure(nb * clen * 2)?;
        let add: Vec<f32> = self.fvideo_dc.iter().map(|g| (c * g) as f32).collect();
        st.k_extract.args(&[
            Arg::Buf(&st.rows[v_out[0]]),
            Arg::Buf(&st.rows[v_out[1]]),
            Arg::Buf(&st.rows[v_out[2]]),
            Arg::Buf(&st.raw),
            Arg::Buf(&st.rows[r]),
            Arg::Buf(&st.rows[e]),
            Arg::I32(cstart as i32),
            Arg::I32(clen as i32),
            Arg::I32(rstart as i32),
            Arg::I32(rlen as i32),
            Arg::F32(add[0]),
            Arg::F32(add[1]),
            Arg::F32(add[2]),
            Arg::F32(c as f32),
            Arg::Buf(&st.out_f[0]),
            Arg::Buf(&st.out_f[1]),
            Arg::Buf(&st.out_f[2]),
            Arg::Buf(&st.out_f[3]),
            Arg::Buf(&st.out_f[4]),
            Arg::Buf(&st.out_efm),
        ])?;
        q.launch(&st.k_extract, &[clen.max(rlen), nb], 256)?;

        let mut ch: Vec<Vec<f32>> = (0..5).map(|i| vec![0f32; nb * if i == 4 { rlen } else { clen }]).collect();
        for (i, v) in ch.iter_mut().enumerate() {
            q.read(&st.out_f[i], 0, v)?;
        }
        let mut efm = vec![0i16; nb * clen];
        q.read(&st.out_efm, 0, &mut efm)?;
        let mut bins = vec![[0f32; 2]; nb * 4 * run];
        q.read(&st.bins, 0, &mut bins)?;
        drop(guard);

        // Assemble per block; the analog audio's stage 1 runs on the CPU from
        // the gathered bins.
        let to64 = |v: &[[f32; 2]]| -> Vec<Complex64> { v.iter().map(|c| Complex64::new(f64::from(c[0]), f64::from(c[1]))).collect() };
        let res = (0..nb)
            .map(|b| {
                let cl = |i: usize| ch[i][b * clen..(b + 1) * clen].to_vec();
                let audio = [0usize, 1].map(|a| {
                    let base = b * 4 * run + 2 * a * run;
                    let h = half_bins[a];
                    audio_stage1(&to64(&bins[base..base + h]), &to64(&bins[base + run..base + run + h]), a, spec, true)
                });
                BlockDecode {
                    video: VideoChannels {
                        demod: cl(0),
                        demod_raw: cl(3),
                        demod_05: cl(1),
                        demod_burst: cl(2),
                        audio,
                        efm: efm[b * clen..(b + 1) * clen].to_vec(),
                    },
                    rfhpf: ch[4][b * rlen..(b + 1) * rlen].to_vec(),
                }
            })
            .collect();
        Ok(res)
    }
}

/// Shared per-decoder handle.
pub(crate) type GpuDemodRef = Arc<GpuDemod>;
