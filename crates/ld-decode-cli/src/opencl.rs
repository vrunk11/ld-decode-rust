//! OpenCL backend of the stack's virtual source (`--device opencl`), on the
//! decoder crate's shared OpenCL layer. The kernels evaluate exactly the plan
//! the CPU path evaluates (`stack::Plan`): the same positions (in double
//! precision), the same sinc phase, the same tables, in single precision for
//! the dot products.

use std::sync::Mutex;

use anyhow::Result;
use ld_decode::opencl::{Arg, Buffer, Gpu, Kernel, Pinned, Queue};

use crate::stack::{sinc_lut, Inner, Plan, HI_TAPS, LOCK_WIN, SINC_PHASES, SINC_TAPS};

const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

// One input read at its virtual samples: the full signal (sinc) and the
// analytic video band, as stack.rs compute_task does.
__kernel void read_input(
    __global const float* raw_all,
    __global const ushort* codes_all,
    const int use_codes,
    const float off,
    __global const uint* seg,
    __global const double* seg_t0,
    __global const double* a0_all,
    __global const double* slope_all,
    const double s0,
    const int len,
    const int base,
    const long raw_off,
    const long start,
    const int seg_off,
    __global const REAL* sinc,
    __global const REAL2* hil,
    __global REAL* full,
    __global REAL2* hi,
    __global uchar* cov)
{
    int j = get_global_id(0);
    if (j >= len) return;
    __global const float* raw = raw_all + raw_off;
    __global const ushort* codes = codes_all + raw_off;
    __global const double* a0 = a0_all + seg_off;
    __global const double* slope = slope_all + seg_off;
    uint k = seg[j];
    double a = a0[k];
    if (isnan(a)) {
        full[base + j] = (REAL)0;
        hi[base + j] = (REAL2)((REAL)0, (REAL)0);
        cov[base + j] = 0;
        return;
    }
    // Absolute position, then the span index (see `InputSpan::a0`).
    double p = a + ((s0 + (double)j) - seg_t0[k]) * slope[k];
    double kf = floor(p);
    int k0 = (int)((long)kf - start);
    int ph = (int)round((p - kf) * (double)PHASES);
    int xs = k0 - (SINC_TAPS / 2 - 1), xh = k0 - (HI_TAPS / 2 - 1);
    __global const REAL* hs = sinc + ph * SINC_TAPS;
    __global const REAL2* hh = hil + ph * HI_TAPS;
    REAL f = (REAL)0;
    REAL2 h = (REAL2)((REAL)0, (REAL)0);
    if (use_codes) {
        for (int d = 0; d < SINC_TAPS; d++) f += hs[d] * ((REAL)codes[xs + d] + (REAL)off);
        for (int d = 0; d < HI_TAPS; d++) h += hh[d] * ((REAL)codes[xh + d] + (REAL)off);
    } else {
        for (int d = 0; d < SINC_TAPS; d++) f += hs[d] * (REAL)raw[xs + d];
        for (int d = 0; d < HI_TAPS; d++) h += hh[d] * (REAL)raw[xh + d];
    }
    full[base + j] = f;
    hi[base + j] = h;
    cov[base + j] = 1;
}

// Phase-lock every input's video band onto the primary's over LOCK_WIN
// samples and sum with the weights (the rest of each input as read). A
// work-group of TILE outputs stages the window products of each input in
// local memory once, so the 128-sample sums read it instead of global memory.
#define TILE 256
__kernel void combine(
    const int ni,
    const int len,
    const int m,
    const int n,
    __global const REAL* full,
    __global const REAL2* hi,
    __global const uchar* cov,
    __global const REAL* weight,
    __global float* out)
{
    __local REAL2 prod[TILE + LOCK_WIN];
    int lid = get_local_id(0);
    int jo = get_global_id(0);
    int j = jo + m;
    int t0 = (int)get_group_id(0) * TILE + m - LOCK_WIN / 2;
    bool active = jo < n;
    REAL acc = (REAL)0, wsum = (REAL)0;
    if (active && cov[j]) { acc += full[j] * weight[0]; wsum += weight[0]; }
    for (int i = 1; i < ni; i++) {
        int b = i * len;
        for (int u = lid; u < TILE + LOCK_WIN; u += TILE) {
            int t = t0 + u;
            REAL2 v = (REAL2)((REAL)0, (REAL)0);
            if (t >= 0 && t < len && cov[t] && cov[b + t]) {
                REAL2 x = hi[t], y = hi[b + t];
                v = (REAL2)(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
            }
            prod[u] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (active && cov[b + j]) {
            REAL2 r = (REAL2)((REAL)0, (REAL)0);
            for (int u = 0; u < LOCK_WIN; u++) r += prod[lid + u];
            REAL a = sqrt(r.x * r.x + r.y * r.y);
            REAL2 rot = a > (REAL)0 ? r / a : (REAL2)((REAL)1, (REAL)0);
            REAL2 d = (REAL2)(rot.x - (REAL)1, rot.y);
            REAL2 z = hi[b + j];
            acc += (full[b + j] + (z.x * d.x - z.y * d.y)) * weight[i];
            wsum += weight[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) out[jo] = wsum > (REAL)0 ? (float)(acc / wsum) : 0.0f;
}
"#;

struct State {
    q: Queue,
    k_read: Kernel,
    k_combine: Kernel,
    sinc: Buffer,
    hil: Buffer,
    weight: Buffer,
    seg: Buffer,
    seg_t0: Buffer,
    raw: Buffer,
    codes: Buffer,
    /// Page-locked staging of the inputs' samples.
    up: Pinned,
    a0: Buffer,
    slope: Buffer,
    full: Buffer,
    hi: Buffer,
    cov: Buffer,
    out: Buffer,
}

pub struct StackGpu {
    ni: usize,
    /// Double precision (`Inner::exact`).
    f64: bool,
    state: Mutex<State>,
}

impl StackGpu {
    pub(crate) fn new(inner: &Inner) -> Result<Self> {
        let gpu = Gpu::get()?;
        tracing::info!("stack: OpenCL device {}", gpu.name);
        let mut ks = gpu
            .kernels(
                SOURCE,
                &format!(
                    "-DPHASES={SINC_PHASES} -DSINC_TAPS={SINC_TAPS} -DHI_TAPS={HI_TAPS} -DLOCK_WIN={LOCK_WIN} {}",
                    if inner.exact { "-DREAL=double -DREAL2=double2" } else { "-DREAL=float -DREAL2=float2" }
                ),
                &["read_input", "combine"],
            )?
            .into_iter();
        let (k_read, k_combine) = (ks.next().unwrap(), ks.next().unwrap());
        let q = gpu.queue()?;
        let f64 = inner.exact;
        // Tables and weights as the kernels' REAL.
        let constant = |data: &[f64]| -> Result<Buffer> {
            let bytes: Vec<u8> = if f64 {
                data.iter().flat_map(|v| v.to_ne_bytes()).collect()
            } else {
                data.iter().flat_map(|&v| (v as f32).to_ne_bytes()).collect()
            };
            let b = gpu.buffer(bytes.len())?;
            q.write(&b, 0, &bytes)?;
            Ok(b)
        };
        let sinc: Vec<f64> = sinc_lut().to_vec();
        let hil: Vec<f64> = inner.hi_lut.iter().flat_map(|c| [c.re, c.im]).collect();
        // Single precision: weights scaled to 1 at most (exact: as they are).
        let wmax = if f64 { 1.0 } else { inner.weight.iter().copied().fold(0.0, f64::max) };
        let weight: Vec<f64> = inner.weight.iter().map(|&w| w / wmax).collect();
        let state = State {
            sinc: constant(&sinc)?,
            hil: constant(&hil)?,
            weight: constant(&weight)?,
            seg: gpu.buffer(4)?,
            seg_t0: gpu.buffer(8)?,
            raw: gpu.buffer(4)?,
            codes: gpu.buffer(2)?,
            up: gpu.pinned(4)?,
            a0: gpu.buffer(8)?,
            slope: gpu.buffer(8)?,
            full: gpu.buffer(4)?,
            hi: gpu.buffer(8)?,
            cov: gpu.buffer(1)?,
            out: gpu.buffer(4)?,
            q,
            k_read,
            k_combine,
        };
        Ok(Self { ni: inner.weight.len(), f64, state: Mutex::new(state) })
    }

    /// Fill `out` with plan samples `m..m + out.len()`.
    pub(crate) fn run(&self, plan: &Plan, m: usize, out: &mut [f32]) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let (len, ni) = (plan.len, self.ni);
        st.seg.ensure(len * 4)?;
        st.seg_t0.ensure(plan.seg_t0.len() * 8)?;
        let rsz = if self.f64 { 8 } else { 4 };
        st.full.ensure(ni * len * rsz)?;
        st.hi.ensure(ni * len * 2 * rsz)?;
        st.cov.ensure(ni * len)?;
        st.out.ensure(out.len() * 4)?;
        st.q.write(&st.seg, 0, &plan.seg)?;
        st.q.write(&st.seg_t0, 0, &plan.seg_t0)?;
        // Every input in one upload: spans back to back, segment laws at
        // `i * nseg` (all NaN for an input this read does not cover).
        let nseg = plan.seg_t0.len();
        let codes = plan.inputs.iter().flatten().any(|sp| !sp.codes.is_empty());
        let total: usize = plan.inputs.iter().flatten().map(|sp| if codes { sp.codes.len() } else { sp.raw.len() }).sum();
        let elem = if codes { 2 } else { 4 };
        st.up.ensure(total * elem)?;
        let mut offs = Vec::with_capacity(ni);
        let mut a0 = vec![f64::NAN; ni * nseg];
        let mut slope = vec![0.0f64; ni * nseg];
        let mut at = 0usize;
        for (i, span) in plan.inputs.iter().enumerate() {
            offs.push((at as i64, span.as_ref().map_or(0.0, |sp| sp.off), span.as_ref().map_or(0, |sp| sp.start)));
            if let Some(span) = span {
                if codes {
                    st.up.slice_mut::<u16>(total)[at..at + span.codes.len()].copy_from_slice(&span.codes);
                    at += span.codes.len();
                } else {
                    st.up.slice_mut::<f32>(total)[at..at + span.raw.len()].copy_from_slice(&span.raw);
                    at += span.raw.len();
                }
                a0[i * nseg..(i + 1) * nseg].copy_from_slice(&span.a0);
                slope[i * nseg..(i + 1) * nseg].copy_from_slice(&span.slope);
            }
        }
        let t0 = std::time::Instant::now();
        let target = if codes { &mut st.codes } else { &mut st.raw };
        target.ensure(total * elem)?;
        let target = if codes { &st.codes } else { &st.raw };
        st.q.upload(target, 0, &st.up, 0, total * elem)?;
        st.a0.ensure(a0.len() * 8)?;
        st.slope.ensure(slope.len() * 8)?;
        st.q.write(&st.a0, 0, &a0)?;
        st.q.write(&st.slope, 0, &slope)?;
        for (i, &(off, sample_off, start)) in offs.iter().enumerate() {
            st.k_read.args(&[
                Arg::Buf(&st.raw),
                Arg::Buf(&st.codes),
                Arg::I32(codes as i32),
                Arg::F32(sample_off),
                Arg::Buf(&st.seg),
                Arg::Buf(&st.seg_t0),
                Arg::Buf(&st.a0),
                Arg::Buf(&st.slope),
                Arg::F64(plan.s0),
                Arg::I32(len as i32),
                Arg::I32((i * len) as i32),
                Arg::I64(off),
                Arg::I64(start),
                Arg::I32((i * nseg) as i32),
                Arg::Buf(&st.sinc),
                Arg::Buf(&st.hil),
                Arg::Buf(&st.full),
                Arg::Buf(&st.hi),
                Arg::Buf(&st.cov),
            ])?;
            st.q.launch(&st.k_read, &[len], 256)?;
        }
        let timing = std::env::var_os("LD_TIMING").is_some();
        let t1 = std::time::Instant::now();
        if timing {
            st.q.finish()?;
        }
        let t2 = std::time::Instant::now();
        st.k_combine.args(&[
            Arg::I32(ni as i32),
            Arg::I32(len as i32),
            Arg::I32(m as i32),
            Arg::I32(out.len() as i32),
            Arg::Buf(&st.full),
            Arg::Buf(&st.hi),
            Arg::Buf(&st.cov),
            Arg::Buf(&st.weight),
            Arg::Buf(&st.out),
        ])?;
        st.q.launch(&st.k_combine, &[out.len()], 256)?;
        st.q.read(&st.out, 0, out)?;
        if timing {
            ld_decode::teeprintln!(
                "STACKCL upload_us={} read_us={} combine_us={}",
                (t1 - t0).as_micros(),
                (t2 - t1).as_micros(),
                t2.elapsed().as_micros()
            );
        }
        Ok(())
    }
}
