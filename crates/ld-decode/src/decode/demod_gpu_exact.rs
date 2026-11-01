//! `--gpu`: the whole FM demodulator on an OpenCL device, computing **the
//! reference's arithmetic** and so producing the CPU kernel's bytes.
//!
//! This is the bit-exact GPU path. It is [`super::demodblock::demod_block_cpu`]
//! moved onto the device operation for operation:
//!
//! - every transform runs ducc's own 1-D plan (`ducc_gpu`, from `ducc_plan`),
//!   not a different FFT: `rfft` + reflection for `fft_real_full`, the full
//!   complex inverse for `hilbert`/EFM, and the half-spectrum `c2r` for `rfhpf`
//!   and the three `FVideo` channels;
//! - every filter product is numpy's complex multiply, i.e. two rounded
//!   products and one FMA each (`fmaddsub`), not a 4-product form;
//! - the discriminator calls a device port of `optimized::ucrt_atan2`, so the
//!   `atan2` bits are the ones the CPU's port produces;
//! - nothing is re-centred: the discriminator emits absolute Hz, as the
//!   reference does, so the post-demod filters see the values the reference
//!   filters (the fast path's `f - c` / `+ c*H(0)` trick changes rounding).
//!
//! **Fallback.** The `atan2` port declines NaN/inf/subnormal operands (it has
//! no code for the scalings UCRT uses there). Rather than approximate them,
//! the discriminator sets a per-block flag and the host recomputes that whole
//! block with [`demod_block_cpu`] — so a declined operand cannot change an
//! output byte, and `LD_GPU_EXACT_FALLBACK=1` reports how many blocks took the
//! path. On a real capture it is zero (the discriminand is a product of
//! spectrum bins: any block without signal gives exact zeros, which the port
//! covers).
//!
//! The faster but differently-rounded Stockham/`atan2` implementation is
//! `--gpu-fast` ([`super::demod_gpu`]).

use std::sync::Mutex;

use anyhow::Result;
use rustfft::num_complex::Complex64;

use super::demodblock::{audio_stage1, kept_range, BlockDecode, DemodSpecRef, VideoChannels};
use super::ducc_gpu::{self, DuccGpu, PlanDev, Scratch};
use super::ducc_plan::C2cPlan;
use crate::opencl::{Arg, Buffer, Gpu, Kernel, Pinned};
use crate::optimized::ucrt_atan2::{ATAN_TBL_HI, ATAN_TBL_LO};

/// Blocks per submission. Each block is ~10 buffers of `blocklen` complex
/// values, so this bounds the device memory (~250 MB at 32) as well as the
/// number of launches: one batch issues ~50 launches however many blocks it
/// holds.
const MAX_BATCH: usize = 32;

/// The demodulator's own kernels. The transforms come from `ducc_gpu`.
const SOURCE: &str = r#"
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// --- UCRT atan2 (port of src/optimized/ucrt_atan2.rs) ----------------------
// Bit-exact reconstruction of ucrtbase.dll's atan2: the argument reduction
// (signs, a swap so the ratio is <= 1) and the polynomial/table atan core, with
// the value carried as a pair of doubles that are added at the end so the
// pi/2 and pi fix-ups keep their low bits. Every op sequence below is the CPU
// port's, in the same order -- an equivalent regrouping is a different answer
// (the table read order was once swapped and cost 1 ulp on 25% of samples).
// `*ok` is cleared for the operand
// classes the port declines (NaN/inf, subnormal operands, and the
// subnormal-scaled ratio branch); the caller then reruns the block on the CPU.
#define A_TINY 1e-08
#define PI_LO 3.178650954705639e-08
#define INV_256 0.00390625
#define SIXTEENTH 0.0625
#define C090 0.09002981028544979
#define C111 0.11110736283514526
#define C142 0.1428571356180717
#define C199A 0.19999918038989142
#define C199B 0.19999999999393223
#define C333A 0.33333333333224097
#define C333B 0.3333333333333317
#define PI_HI 3.1415926218032837
#define SCALE_256 256.0
#define PI2_LO 6.123233995736766e-17
#define HALF 0.5
#define PI2_HI 1.5707963267948966
#define PI 3.141592653589793

inline double ucrt_atan2(double y, double x, int *ok,
                         __global const double *tbl_hi, __global const double *tbl_lo)
{
  ulong bx = as_ulong(x), by = as_ulong(y);
  int sx = (int)(bx >> 63) != 0;
  int sy = (int)(by >> 63) != 0;
  ulong abs_x = bx & 0x7fffffffffffffffUL;
  ulong abs_y = by & 0x7fffffffffffffffUL;
  if (abs_x >= 0x7ff0000000000000UL || abs_y >= 0x7ff0000000000000UL) { *ok = 0; return 0.0; }
  if (abs_y == 0) return sx ? (sy ? -PI : PI) : y;
  if (abs_x == 0) return sy ? -PI2_HI : PI2_HI;
  int exp_x = (int)((bx >> 52) & 0x7ff);
  int exp_y = (int)((by >> 52) & 0x7ff);
  if (exp_x == 0 || exp_y == 0) { *ok = 0; return 0.0; }
  double vx = x, vy = y;
  if (exp_x < 0x3fd && exp_y < 0x3fd) {
    vx = x * 2.0;
    vy = y * 2.0;
    exp_x = (int)((as_ulong(vx) >> 52) & 0x7ff);
    exp_y = (int)((as_ulong(vy) >> 52) & 0x7ff);
  }
  int esi = exp_y - exp_x;
  if (esi > 56) return sy ? -PI2_HI : PI2_HI;
  if (esi < -28) {
    if (!sx) {
      if (esi >= -1074) {
        if (esi >= -1022) return vy / vx;
        *ok = 0;
        return 0.0;
      }
      return sy ? -0.0 : 0.0;
    }
  }
  if (esi < -56 && sx) return sy ? -PI : PI;
  double lg = sx ? -vx : vx;
  double sm = sy ? -vy : vy;
  int swapped = sm > lg;
  if (swapped) { double t = lg; lg = sm; sm = t; }
  double z = sm / lg;
  double lo, hi;
  if (z <= SIXTEENTH) {
    if (A_TINY > z) {
      lo = 0.0;
      hi = z;
    } else {
      double z2 = z * z;
      double lg_hi = as_double(as_ulong(lg) & 0xffffffff00000000UL);
      double z_hi = as_double(as_ulong(z) & 0xffffffff00000000UL);
      double t1 = lg - lg_hi;
      double acc = sm;
      t1 *= z_hi;
      acc -= z_hi * lg_hi;
      acc -= t1;
      double c = C111;
      acc -= (z - z_hi) * lg;
      acc /= lg;
      double t = z2 * C090;
      c -= t;
      t = C142;
      c *= z2;
      t -= c;
      double c2 = C199B;
      t *= z2;
      c2 -= t;
      double t2 = C333B;
      c2 *= z2;
      double z3 = z2 * z;
      t2 -= c2;
      t2 *= z3;
      acc -= t2;
      acc += z;
      lo = 0.0;
      hi = acc;
    }
  } else {
    int n = (int)(z * SCALE_256 + HALF);
    int idx = n - 16;
    // The table arrays are named after the half of the value they hold, and
    // UCRT adds them the other way round: the high part becomes the pair's
    // `lo` and the low part is folded into the polynomial. Copying the CPU
    // port's reads exactly (`t_lo <- HI`, `t_hi <- LO`) is what keeps the
    // final `lo + hi` grouping -- and so its rounding -- identical.
    double t_lo = tbl_hi[idx];
    double t_hi = tbl_lo[idx];
    double m = ((double)n) * INV_256;
    int d = 0x3ff - (int)((as_ulong(lg) >> 52) & 0x7ff);
    int e = d / 2;
    int f = d - e;
    double two_e = as_double(((ulong)(e + 0x3ff)) << 52);
    double two_f = as_double(((ulong)(f + 0x3ff)) << 52);
    double big = two_e * lg * two_f;
    double small = two_e * sm * two_f;
    double x_hi = as_double(as_ulong(big) & 0xfffffffff8000000UL);
    double p = big - x_hi;
    double acc = small;
    acc -= x_hi * m;
    p *= m;
    double msmall = m * small;
    acc -= p;
    msmall += big;
    acc /= msmall;
    double r = acc;
    double r2 = r * r;
    double poly = r + t_hi;
    double c = C333A;
    c -= r2 * C199A;
    c *= r2;
    c *= r;
    poly -= c;
    lo = t_lo;
    hi = poly;
  }
  if (swapped) { lo = PI2_HI - lo; hi = PI2_LO - hi; }
  if (sx) { lo = PI_HI - lo; hi = PI_LO - hi; }
  double v = lo + hi;
  if (sy) v = -v;
  return v;
}

// numpy's complex multiply (`fmaddsub`): one rounded product plus one FMA per
// component. NOT the 4-product form -- this is a rounding decision.
inline double2 npcmul(double2 a, double2 b)
{ return (double2)(fma(a.x, b.x, -(a.y * b.y)), fma(a.x, b.y, a.y * b.x)); }

// --- kernels ---------------------------------------------------------------

// H = X * RFVideo (* MTF power), E = X * Fefm (full length), and the first
// `w` bins of the dropout-detection highpass product (half spectrum).
__kernel void products(__global const double2 *x, __global const double2 *rfvideo,
                       __global const double2 *mtfpow, const int use_pow,
                       __global const double2 *fefm, __global const double2 *frfhpf,
                       const int n, const int w,
                       __global double2 *h, __global double2 *e, __global double2 *r)
{
  int k = get_global_id(0);
  int b = get_global_id(1);
  if (k >= n) return;
  double2 v = x[(size_t)b * n + k];
  double2 p = npcmul(v, rfvideo[k]);
  if (use_pow) p = npcmul(p, mtfpow[k]);
  h[(size_t)b * n + k] = p;
  e[(size_t)b * n + k] = npcmul(v, fefm[k]);
  if (k < w) r[(size_t)b * w + k] = npcmul(v, frfhpf[k]);
}

// FM discriminator in absolute Hz, exactly as `unwrap_hilbert` computes it:
// plain 4-product conjugate multiply (numba, not numpy), UCRT atan2, `+2pi` on
// a negative phase, then `* freq_hz/2pi`. `demod[0]` stays 0.
__kernel void discriminate(__global const double2 *hilb, __global double *raw,
                           __global double *dclamp, __global const double *tbl_hi,
                           __global const double *tbl_lo, const double scale,
                           const double lo, const double hi, const int n,
                           __global int *flags)
{
  int i = get_global_id(0);
  int b = get_global_id(1);
  if (i >= n) return;
  size_t o = (size_t)b * n + i;
  double v;
  if (i == 0) {
    v = 0.0;
  } else {
    double2 z = hilb[o];
    double2 w = hilb[o - 1];
    double a = z.x, bb = z.y;
    double c = w.x, dd = -w.y;
    double pre = a * c - bb * dd;
    double pim = a * dd + bb * c;
    int ok = 1;
    double d = ucrt_atan2(pim, pre, &ok, tbl_hi, tbl_lo);
    if (!ok) flags[b] = 1;
    v = d < 0.0 ? (d + 6.283185307179586) * scale : d * scale;
  }
  raw[o] = v;
  dclamp[o] = clamp(v, lo, hi);
}

// The three post-demod half-spectrum products. `f` holds the three filters
// back to back (`w` complex each); `y` is the full-length row, whose first `w`
// bins are the half spectrum.
__kernel void video_products(__global const double2 *y, __global const double2 *f,
                             const int n, const int w, __global double2 *v)
{
  int k = get_global_id(0);
  int b = get_global_id(1);
  if (k >= w) return;
  double2 x = y[(size_t)b * n + k];
  size_t o = (size_t)b * 3 * w + k;
  v[o] = npcmul(x, f[k]);
  v[o + w] = npcmul(x, f[w + k]);
  v[o + 2 * w] = npcmul(x, f[2 * w + k]);
}

// Spectrum bins the CPU's analog-audio stage reads: per block, `n` bins from
// each of the four starts in `starts` (the same gather the fast path uses).
__kernel void gather_bins(__global const double2 *x, __global const int *starts,
                          const int n, const int nn, __global double2 *out)
{
  int j = get_global_id(0);
  int b = get_global_id(1);
  if (j >= 4 * n) return;
  int run = j / n;
  int i = j - run * n;
  out[(size_t)b * 4 * n + j] = x[(size_t)b * nn + min(starts[run] + i, nn - 1)];
}

// Cut and pack every output channel of the batch.
__kernel void extract(__global const double *vreal, __global const double *rf,
                      __global const double *raw, __global const double2 *efm,
                      const int n, const int cstart, const int clen,
                      const int rstart, const int rlen,
                      __global float *o_demod, __global float *o_05,
                      __global float *o_burst, __global float *o_raw,
                      __global float *o_rfhpf, __global short *o_efm)
{
  int j = get_global_id(0);
  int b = get_global_id(1);
  if (j < clen) {
    int i = cstart + j;
    size_t o = (size_t)b * clen + j;
    o_demod[o] = (float)vreal[(size_t)(b * 3) * n + i];
    o_05[o] = (float)vreal[(size_t)(b * 3 + 1) * n + i];
    o_burst[o] = (float)vreal[(size_t)(b * 3 + 2) * n + i];
    o_raw[o] = (float)raw[(size_t)b * n + i];
    o_efm[o] = convert_short_rtz(clamp(efm[(size_t)b * n + i].x, -32768.0, 32767.0));
  }
  if (j < rlen) {
    o_rfhpf[(size_t)b * rlen + j] = (float)rf[(size_t)b * n + rstart + j];
  }
}
"#;

/// Per-decoder device state: the plan, its buffers and the demod kernels.
struct State {
    /// The transforms' harness. Its queue is the *only* one this path uses:
    /// every upload, kernel and download must be ordered against the plan's
    /// launches, and two queues from the same context are not ordered against
    /// each other (the first cut uploaded on one and transformed on the other,
    /// so a batch could transform the previous batch's samples).
    dg: DuccGpu,
    plan_n: PlanDev,
    plan_h: PlanDev,
    sc: Scratch,
    // Persistent buffers, by index into `dg`.
    smp: usize,
    xs: usize,
    hs: usize,
    es: usize,
    rs: usize,
    ys: usize,
    vs: usize,
    hilb: usize,
    efmc: usize,
    dcl: usize,
    raw: usize,
    bins: usize,
    cch3: usize,
    cch1: usize,
    vreal: usize,
    rreal: usize,
    flags: usize,
    out_f: [usize; 5],
    out_efm: usize,
    // Filters and tables.
    f_rfvideo: Buffer,
    f_mtfpow: Buffer,
    f_fefm: Buffer,
    f_frfhpf: Buffer,
    f_fvideo: Buffer,
    starts: Buffer,
    tbl_hi: Buffer,
    tbl_lo: Buffer,
    // Demod kernels (their own program, launched on `q`).
    k_products: Kernel,
    k_bins: Kernel,
    k_disc: Kernel,
    k_video: Kernel,
    k_extract: Kernel,
    up: Pinned,
    down: Pinned,
    pow_level: Option<f64>,
}

pub(crate) struct GpuDemodExact {
    blocklen: usize,
    state: Mutex<State>,
}

/// Complex values as bytes, in the device's `double2` layout.
fn cplx_bytes(v: &[Complex64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 16);
    for c in v {
        out.extend_from_slice(&c.re.to_ne_bytes());
        out.extend_from_slice(&c.im.to_ne_bytes());
    }
    out
}

impl GpuDemodExact {
    pub(crate) fn new(spec: &DemodSpecRef) -> Result<Self> {
        let n = spec.blocklen;
        anyhow::ensure!(n % 16 == 0, "GPU demod needs blocklen divisible by 16 (got {n})");
        let nh = n / 2;
        let w = nh + 1;
        let gpu = Gpu::get()?;
        let names = ["products", "discriminate", "video_products", "gather_bins", "extract"];
        let ks = gpu
            .kernels(SOURCE, &format!("-DN={n}"), &names)?
            .into_iter()
            .collect::<Vec<_>>();
        let mut ks = ks.into_iter();
        let mut k = || ks.next().unwrap();
        let (k_products, k_disc, k_video, k_bins, k_extract) = (k(), k(), k(), k(), k());

        let mut dg = DuccGpu::new(&gpu)?;
        let plan_n = PlanDev::build(&mut dg, &C2cPlan::new(n))?;
        let plan_h = PlanDev::build(&mut dg, &C2cPlan::new(nh))?;
        let sc = Scratch::alloc(&mut dg, 3 * MAX_BATCH, nh)?;

        let bt = MAX_BATCH;
        let alloc = |dg: &mut DuccGpu, bytes: usize| dg.alloc(bytes);
        let smp = alloc(&mut dg, bt * n * 8)?;
        let xs = alloc(&mut dg, bt * n * 16)?;
        let hs = alloc(&mut dg, bt * n * 16)?;
        let es = alloc(&mut dg, bt * n * 16)?;
        let rs = alloc(&mut dg, bt * w * 16)?;
        let ys = alloc(&mut dg, bt * n * 16)?;
        let vs = alloc(&mut dg, bt * 3 * w * 16)?;
        let hilb = alloc(&mut dg, bt * n * 16)?;
        let efmc = alloc(&mut dg, bt * n * 16)?;
        let dcl = alloc(&mut dg, bt * n * 8)?;
        let raw = alloc(&mut dg, bt * n * 8)?;
        let cch3 = alloc(&mut dg, 3 * bt * nh * 16)?;
        let cch1 = alloc(&mut dg, bt * nh * 16)?;
        let vreal = alloc(&mut dg, 3 * bt * n * 8)?;
        let rreal = alloc(&mut dg, bt * n * 8)?;
        let flags = alloc(&mut dg, bt * 4)?;
        let out_f = [
            alloc(&mut dg, bt * n * 4)?,
            alloc(&mut dg, bt * n * 4)?,
            alloc(&mut dg, bt * n * 4)?,
            alloc(&mut dg, bt * n * 4)?,
            alloc(&mut dg, bt * n * 4)?,
        ];
        let out_efm = alloc(&mut dg, bt * n * 2)?;

        let q = gpu.queue()?;
        let f = spec.filters;
        let upload = |data: &[Complex64]| -> Result<Buffer> {
            let bytes = cplx_bytes(data);
            let b = gpu.buffer(bytes.len())?;
            q.write(&b, 0, &bytes)?;
            Ok(b)
        };
        let f_rfvideo = upload(&f.rfvideo)?;
        let f_fefm = upload(&f.fefm)?;
        let f_frfhpf = upload(&f.frfhpf_rfft)?;
        let mut fvideo_bins = Vec::with_capacity(3 * w);
        for ch in 0..3 {
            fvideo_bins.extend_from_slice(&f.fvideo_rfft[ch][..w]);
        }
        let f_fvideo = upload(&fvideo_bins)?;
        let f_mtfpow = gpu.buffer(n * 16)?;
        let starts = gpu.buffer(16)?;
        let tbl_hi = gpu.buffer(std::mem::size_of_val(&ATAN_TBL_HI))?;
        let tbl_lo = gpu.buffer(std::mem::size_of_val(&ATAN_TBL_LO))?;
        q.write(&tbl_hi, 0, &ATAN_TBL_HI[..])?;
        q.write(&tbl_lo, 0, &ATAN_TBL_LO[..])?;

        // The audio band's bin count decides the gather width.
        let run = (0..2)
            .map(|ch| spec.filters.audio[ch].nbins / 2)
            .max()
            .unwrap_or(0);
        let bins = alloc(&mut dg, bt * 4 * run * 16)?;

        let state = State {
            dg,
            plan_n,
            plan_h,
            sc,
            smp,
            xs,
            hs,
            es,
            rs,
            ys,
            vs,
            hilb,
            efmc,
            dcl,
            raw,
            bins,
            cch3,
            cch1,
            vreal,
            rreal,
            flags,
            out_f,
            out_efm,
            f_rfvideo,
            f_mtfpow,
            f_fefm,
            f_frfhpf,
            f_fvideo,
            starts,
            tbl_hi,
            tbl_lo,
            k_products,
            k_bins,
            k_disc,
            k_video,
            k_extract,
            up: gpu.pinned(4)?,
            down: gpu.pinned(4)?,
            pow_level: None,
        };
        Ok(Self {
            blocklen: n,
            state: Mutex::new(state),
        })
    }

    /// Demodulate `blocks` (each `blocklen` samples), cut as the CPU kernel
    /// does.
    pub(crate) fn demod(
        &self,
        blocks: &[(u64, &[f32])],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
    ) -> Result<Vec<BlockDecode>> {
        let mut out = Vec::with_capacity(blocks.len());
        for (i, chunk) in blocks.chunks(MAX_BATCH).enumerate() {
            out.extend(self.batch(chunk, mtf_level, mtf_pow, rotdelay, spec, i * MAX_BATCH)?);
        }
        Ok(out)
    }

    fn batch(
        &self,
        blocks: &[(u64, &[f32])],
        mtf_level: f64,
        mtf_pow: Option<&[Complex64]>,
        rotdelay: i64,
        spec: &DemodSpecRef,
        first: usize,
    ) -> Result<Vec<BlockDecode>> {
        let n = self.blocklen;
        let nh = n / 2;
        let w = nh + 1;
        let nb = blocks.len();
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let dg = &st.dg;
        let q = &dg.q;

        // The block samples go up as f64: f32 -> f64 is exact, which is the
        // conversion the CPU kernel makes.
        st.up.ensure(nb * n * 8)?;
        {
            let dst = st.up.slice_mut::<f64>(nb * n);
            for (row, b) in dst.chunks_exact_mut(n).zip(blocks) {
                for (d, &v) in row.iter_mut().zip(&b.1[..n]) {
                    *d = f64::from(v);
                }
            }
        }
        q.upload(dg.buf(st.smp), 0, &st.up, 0, nb * n * 8)?;

        // 1. X = fft_real_full(samples)
        ducc_gpu::r2c(dg, &st.plan_h, st.plan_n.roots(), &st.sc, nb, n, st.smp, st.xs, true)?;

        // 2. H / EFM / rfhpf products.
        let use_pow = mtf_pow.is_some();
        if let Some(p) = mtf_pow {
            if st.pow_level != Some(mtf_level) {
                q.write(&st.f_mtfpow, 0, &cplx_bytes(p))?;
                st.pow_level = Some(mtf_level);
            }
        }
        st.k_products.args(&[
            Arg::Buf(dg.buf(st.xs)),
            Arg::Buf(&st.f_rfvideo),
            Arg::Buf(&st.f_mtfpow),
            Arg::I32(use_pow as i32),
            Arg::Buf(&st.f_fefm),
            Arg::Buf(&st.f_frfhpf),
            Arg::I32(n as i32),
            Arg::I32(w as i32),
            Arg::Buf(dg.buf(st.hs)),
            Arg::Buf(dg.buf(st.es)),
            Arg::Buf(dg.buf(st.rs)),
        ])?;
        q.launch(&st.k_products, &[n, nb], 128)?;

        // Analog-audio bins, gathered from X while it is still live.
        let audio_starts: Vec<i32> = (0..2)
            .flat_map(|ch| {
                let af = &spec.filters.audio[ch];
                let half = af.nbins / 2;
                [af.lowbin as i32, (n - af.lowbin - half) as i32]
            })
            .collect();
        let half_bins: Vec<usize> = (0..2).map(|ch| spec.filters.audio[ch].nbins / 2).collect();
        let run = half_bins[0].max(half_bins[1]);
        q.write(&st.starts, 0, &audio_starts)?;
        st.k_bins.args(&[
            Arg::Buf(dg.buf(st.xs)),
            Arg::Buf(&st.starts),
            Arg::I32(run as i32),
            Arg::I32(n as i32),
            Arg::Buf(dg.buf(st.bins)),
        ])?;
        q.launch(&st.k_bins, &[4 * run, nb], 64)?;

        // 3. hilbert = ifft(H), efm_series = ifft(E): full complex inverses.
        ducc_gpu::c2c_inverse(dg, &st.plan_n, &st.sc, nb, n, st.hs, st.hilb)?;
        ducc_gpu::c2c_inverse(dg, &st.plan_n, &st.sc, nb, n, st.es, st.efmc)?;

        // 4. Discriminator (UCRT atan2) in absolute Hz, and the clamp.
        q.write(dg.buf(st.flags), 0, &vec![0i32; nb])?;
        let scale = spec.freq_hz / std::f64::consts::TAU;
        let lo = 1_500_000.0;
        let hi = spec.freq_hz * 0.75;
        st.k_disc.args(&[
            Arg::Buf(dg.buf(st.hilb)),
            Arg::Buf(dg.buf(st.raw)),
            Arg::Buf(dg.buf(st.dcl)),
            Arg::Buf(&st.tbl_hi),
            Arg::Buf(&st.tbl_lo),
            Arg::F64(scale),
            Arg::F64(lo),
            Arg::F64(hi),
            Arg::I32(n as i32),
            Arg::Buf(dg.buf(st.flags)),
        ])?;
        q.launch(&st.k_disc, &[n, nb], 128)?;

        // 5. Y = fft_real_full(clamp(demod)); the three video products. Only
        // the half spectrum is read, so the reflection is skipped here.
        ducc_gpu::r2c(dg, &st.plan_h, st.plan_n.roots(), &st.sc, nb, n, st.dcl, st.ys, false)?;
        st.k_video.args(&[
            Arg::Buf(dg.buf(st.ys)),
            Arg::Buf(&st.f_fvideo),
            Arg::I32(n as i32),
            Arg::I32(w as i32),
            Arg::Buf(dg.buf(st.vs)),
        ])?;
        q.launch(&st.k_video, &[w, nb], 128)?;

        // 6. The three real inverses, and the rfhpf one.
        ducc_gpu::c2r(dg, &st.plan_h, st.plan_n.roots(), &st.sc, st.cch3, 3 * nb, n, st.vs, st.vreal)?;
        ducc_gpu::c2r(dg, &st.plan_h, st.plan_n.roots(), &st.sc, st.cch1, nb, n, st.rs, st.rreal)?;

        // 7. Cut ranges, as the CPU kernel.
        let (cstart, cend) = kept_range(n, spec, true);
        let clen = cend.saturating_sub(cstart);
        let rstart = (spec.blockcut as i64 - rotdelay).clamp(0, n as i64) as usize;
        let rstop_raw = -((spec.blockcut_end as i64) + rotdelay);
        let rstop = if rstop_raw < 0 {
            (n as i64 + rstop_raw).clamp(0, n as i64)
        } else {
            rstop_raw.clamp(0, n as i64)
        } as usize;
        let rlen = rstop.saturating_sub(rstart);
        let out_sizes: [usize; 5] = [
            nb * clen * 4,
            nb * clen * 4,
            nb * clen * 4,
            nb * clen * 4,
            nb * rlen * 4,
        ];
        st.k_extract.args(&[
            Arg::Buf(dg.buf(st.vreal)),
            Arg::Buf(dg.buf(st.rreal)),
            Arg::Buf(dg.buf(st.raw)),
            Arg::Buf(dg.buf(st.efmc)),
            Arg::I32(n as i32),
            Arg::I32(cstart as i32),
            Arg::I32(clen as i32),
            Arg::I32(rstart as i32),
            Arg::I32(rlen as i32),
            Arg::Buf(dg.buf(st.out_f[0])),
            Arg::Buf(dg.buf(st.out_f[1])),
            Arg::Buf(dg.buf(st.out_f[2])),
            Arg::Buf(dg.buf(st.out_f[3])),
            Arg::Buf(dg.buf(st.out_f[4])),
            Arg::Buf(dg.buf(st.out_efm)),
        ])?;
        q.launch(&st.k_extract, &[clen.max(rlen), nb], 128)?;

        // Everything back through the pinned buffer, one wait.
        let efm_bytes = nb * clen * 2;
        let bin_bytes = nb * 4 * run * 16;
        let total: usize = out_sizes.iter().sum::<usize>() + efm_bytes + bin_bytes;
        st.down.ensure(total)?;
        let mut off = 0;
        for (i, &sz) in out_sizes.iter().enumerate() {
            q.download(dg.buf(st.out_f[i]), 0, &mut st.down, off, sz)?;
            off += sz;
        }
        q.download(dg.buf(st.out_efm), 0, &mut st.down, off, efm_bytes)?;
        off += efm_bytes;
        q.download(dg.buf(st.bins), 0, &mut st.down, off, bin_bytes)?;
        q.finish()?;

        // `LD_GPU_DUMP=<dir> LD_GPU_DUMP_BLOCK=<n>`: one block's stages, for a
        // stage-by-stage comparison with the CPU kernel's `LD_DUMP_PIPE`.
        if let (Some(dir), Some(want)) = (
            std::env::var_os("LD_GPU_DUMP"),
            std::env::var("LD_GPU_DUMP_BLOCK").ok().and_then(|v| v.parse::<u64>().ok()),
        ) {
            if let Some(i) = blocks.iter().position(|b| b.0 == want) {
                let dir = std::path::Path::new(&dir);
                let mut dump = |name: &str, buf: usize, count: usize| -> Result<()> {
                    let mut v = vec![0.0f64; count];
                    q.read(dg.buf(buf), i * count * 8, &mut v)?;
                    let mut bytes = Vec::with_capacity(count * 8);
                    for x in &v {
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                    std::fs::write(dir.join(name), &bytes)?;
                    Ok(())
                };
                dump("gpu_x.bin", st.xs, 2 * n)?;
                dump("gpu_h.bin", st.hs, 2 * n)?;
                dump("gpu_hilb.bin", st.hilb, 2 * n)?;
                dump("gpu_demod.bin", st.raw, n)?;
                dump("gpu_dcl.bin", st.dcl, n)?;
                dump("gpu_ys.bin", st.ys, 2 * n)?;
                dump("gpu_efm.bin", st.efmc, 2 * n)?;
                dump("gpu_rfhpf.bin", st.rreal, n)?;
                dump("gpu_vreal.bin", st.vreal, 3 * n)?;
                dump("gpu_vs.bin", st.vs, 3 * w * 2)?;
                crate::teeprintln!("GPUDUMP block {} -> {}", want, dir.display());
            }
        }

        // Which blocks took a declined `atan2` operand?
        let mut flags = vec![0i32; nb];
        dg.read(st.flags, &mut flags)?;
        let declined: Vec<usize> = flags
            .iter()
            .enumerate()
            .filter(|(_, &f)| f != 0)
            .map(|(i, _)| i)
            .collect();

        let bytes = st.down.slice::<u8>(total);
        let mut off = 0;
        let mut take = |len: usize| {
            let b = &bytes[off..off + len];
            off += len;
            b
        };
        let ch: Vec<Vec<f32>> = out_sizes
            .iter()
            .map(|&sz| {
                take(sz)
                    .chunks_exact(4)
                    .map(|w| f32::from_ne_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            })
            .collect();
        let efm: Vec<i16> = take(efm_bytes)
            .chunks_exact(2)
            .map(|w| i16::from_ne_bytes([w[0], w[1]]))
            .collect();
        let bins: Vec<Complex64> = take(bin_bytes)
            .chunks_exact(16)
            .map(|w| {
                Complex64::new(
                    f64::from_ne_bytes(w[..8].try_into().unwrap()),
                    f64::from_ne_bytes(w[8..].try_into().unwrap()),
                )
            })
            .collect();
        drop(guard);

        if !declined.is_empty() && std::env::var_os("LD_GPU_EXACT_FALLBACK").is_some() {
            crate::teeprintln!(
                "GPUFALLBACK blocks {:?} of batch at {}: an atan2 operand the port declines",
                declined.iter().map(|&i| first + i).collect::<Vec<_>>(),
                first
            );
        }

        let mut res: Vec<BlockDecode> = (0..nb)
            .map(|b| {
                let cl = |i: usize| ch[i][b * clen..(b + 1) * clen].to_vec();
                let audio = [0usize, 1].map(|a| {
                    let base = b * 4 * run + 2 * a * run;
                    let h = half_bins[a];
                    audio_stage1(&bins[base..base + h], &bins[base + run..base + run + h], a, spec, true)
                });
                BlockDecode {
                    video: VideoChannels {
                        demod: cl(0),
                        demod_05: cl(1),
                        demod_burst: cl(2),
                        demod_raw: cl(3),
                        audio,
                        efm: efm[b * clen..(b + 1) * clen].to_vec(),
                    },
                    rfhpf: ch[4][b * rlen..(b + 1) * rlen].to_vec(),
                }
            })
            .collect();
        // A declined operand is not approximated: rerun that block on the CPU
        // kernel, which is the reference itself.
        for &i in &declined {
            res[i] = super::demodblock::demod_block_cpu(
                blocks[i].1,
                mtf_level,
                spec,
                true,
                mtf_pow,
                rotdelay,
                blocks[i].0,
            );
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::ucrt_atan2;

    /// The device's discriminator must reproduce the CPU's `atan2` bits.
    ///
    /// This is the one stage the transform gates cannot see, and it has already
    /// cost bytes: reading UCRT's split table in the order its *names* suggest
    /// (`t_lo <- LO`, `t_hi <- HI`) regroups the pair's final `lo + hi` and
    /// moves 1 ulp on ~25% of samples -- enough that `LD_GPU_CHECK` reported
    /// 1-ulp deltas on `demod`/`raw` and a full sideA disc came out 3 bytes off.
    /// So the kernel is pinned against the scalar port here, on the same
    /// conjugate-product input shape the demodulator produces.
    ///
    /// `LD_GPU_PROBE=1` runs it (there is no GPU on CI).
    #[test]
    #[ignore]
    fn device_discriminator_matches_ucrt_atan2() {
        if std::env::var_os("LD_GPU_PROBE").is_none() {
            return;
        }
        let n = 32768usize;
        let gpu = Gpu::get().expect("OpenCL device");
        let mut dg = DuccGpu::new(&gpu).expect("harness");
        let ks = gpu
            .kernels(SOURCE, &format!("-DN={n}"), &["discriminate"])
            .expect("discriminate kernel");
        let k = &ks[0];

        // A row shaped like `ifft(H)`: bins whose magnitudes span the capture's
        // range, with consecutive samples sometimes decades apart (the ratio
        // `z` then covers both of UCRT's branches, and the exponent-gap
        // early-outs). Deterministic, so a failure is reproducible.
        let mut s = 0x0ddc_0ffe_e123_4567u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut row: Vec<Complex64> = Vec::with_capacity(n);
        for i in 0..n {
            // Every fourth sample sits ~2^40 below its neighbours.
            let mag = if i % 4 == 3 { 1e-7 } else { 1e-2 } * (0.5 + next());
            let phase = next() * std::f64::consts::TAU;
            row.push(Complex64::new(mag * phase.cos(), mag * phase.sin()));
        }

        let hilb = dg.alloc(n * 16).expect("hilb");
        let raw = dg.alloc(n * 8).expect("raw");
        let dcl = dg.alloc(n * 8).expect("dcl");
        let flags = dg.alloc(4).expect("flags");
        let tbl_hi = gpu.buffer(std::mem::size_of_val(&ATAN_TBL_HI)).expect("tbl_hi");
        let tbl_lo = gpu.buffer(std::mem::size_of_val(&ATAN_TBL_LO)).expect("tbl_lo");
        let q = &dg.q;
        q.write(&tbl_hi, 0, &ATAN_TBL_HI[..]).expect("write tbl_hi");
        q.write(&tbl_lo, 0, &ATAN_TBL_LO[..]).expect("write tbl_lo");
        dg.write(flags, &[0i32]).expect("write flags");
        dg.write(hilb, &row[..]).expect("write hilb");

        let scale = 40_000_000.0 / std::f64::consts::TAU;
        let (lo, hi) = (1_500_000.0, 40_000_000.0 * 0.75);
        k.args(&[
            Arg::Buf(dg.buf(hilb)),
            Arg::Buf(dg.buf(raw)),
            Arg::Buf(dg.buf(dcl)),
            Arg::Buf(&tbl_hi),
            Arg::Buf(&tbl_lo),
            Arg::F64(scale),
            Arg::F64(lo),
            Arg::F64(hi),
            Arg::I32(n as i32),
            Arg::Buf(dg.buf(flags)),
        ])
        .expect("args");
        q.launch(k, &[n, 1], 128).expect("launch");
        let mut got = vec![0.0f64; n];
        dg.read(raw, &mut got).expect("read raw");
        let mut got_dcl = vec![0.0f64; n];
        dg.read(dcl, &mut got_dcl).expect("read dcl");
        let mut fl = [0i32; 1];
        dg.read(flags, &mut fl[..]).expect("read flags");
        q.finish().expect("finish");
        assert_eq!(fl[0], 0, "the operand set left the port's covered class");

        // The CPU's discriminator, op for op (`unwrap_hilbert_into`).
        let mut bad = 0usize;
        let mut first_bad = None;
        for i in 1..n {
            let (a, b) = (row[i].re, row[i].im);
            let (c, d) = (row[i - 1].re, -row[i - 1].im);
            let pre = a * c - b * d;
            let pim = a * d + b * c;
            let r = ucrt_atan2::atan2(pim, pre).expect("declined on the CPU");
            let want = if r < 0.0 {
                (r + std::f64::consts::TAU) * scale
            } else {
                r * scale
            };
            if got[i].to_bits() != want.to_bits() || got_dcl[i].to_bits() != want.clamp(lo, hi).to_bits() {
                bad += 1;
                first_bad.get_or_insert(i);
            }
        }
        assert_eq!(got[0].to_bits(), 0.0f64.to_bits(), "demod[0] must stay 0");
        assert_eq!(bad, 0, "device discriminator differs at {bad} of {n} samples (first {first_bad:?})");
    }
}
