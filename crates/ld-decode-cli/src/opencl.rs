//! OpenCL backend of the stack's virtual source (`--device opencl`), on the
//! decoder crate's shared OpenCL layer. The kernels evaluate exactly the plan
//! the CPU path evaluates (`stack::Plan`): the same positions (in double
//! precision), the same sinc phase, the same tables, in single precision for
//! the dot products.

use std::sync::Mutex;

use anyhow::Result;
use rayon::prelude::*;
use ld_decode::opencl::{Arg, Buffer, Gpu, Kernel, Pinned, Queue};
use rustfft::num_complex::Complex64;

use crate::stack::{
    sinc_lut, Inner, Plan, CARRIER_WIN, HI_TAPS, H_LO, H_SPAN, KAPPA_LO, KAPPA_SPAN, LOCK_PRIOR, LOCK_WIN, QN_CARRIER, QN_LOCK, Q_LO, Q_ROT,
    Q_MAX, Q_SPAN, SINC_PHASES, SINC_TAPS,
};

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

// stack.rs carrier, amp_weight and lock, in their order.
REAL carrier(REAL2 rl, REAL e, REAL n, REAL qn, REAL* q)
{
    REAL a = sqrt(rl.x * rl.x + rl.y * rl.y);
    *q = e > (REAL)0 ? fmin(fmax(a / e - qn, (REAL)0) / ((REAL)1 - qn), (REAL)Q_MAX) : (REAL)0;
    return fmax(a - qn * e, (REAL)0) / (((REAL)1 - qn) * n);
}

REAL amp_weight(REAL s, REAL afl, REAL w)
{
    return fmax(sqrt(s), afl) * w;
}

REAL2 lock(REAL2 r, REAL ea, REAL eb, REAL nj, REAL q0, REAL qi, REAL h0, REAL* g)
{
    REAL q = q0 * qi;
    REAL a = sqrt(r.x * r.x + r.y * r.y);
    *g = (REAL)1;
    if (q > (REAL)Q_LO) {
        REAL kappa = a / sqrt(ea * eb) / sqrt(q);
        REAL gk = clamp((kappa - (REAL)KAPPA_LO) / (REAL)KAPPA_SPAN, (REAL)0, (REAL)1);
        REAL hp = clamp((h0 - (REAL)H_LO) / (REAL)H_SPAN, (REAL)0, (REAL)1);
        *g = (REAL)1 - ((REAL)1 - gk) * fmin((q - (REAL)Q_LO) / (REAL)Q_SPAN, (REAL)1) * hp;
    }
    if (a <= (REAL)0) return (REAL2)((REAL)1, (REAL)0);
    REAL qr = fmax((q - (REAL)Q_ROT) / ((REAL)1 - (REAL)Q_ROT), (REAL)0);
    REAL wm = nj * qr / ((REAL)1 - qr);
    REAL vx = r.x / a * wm + (REAL)LOCK_PRIOR;
    REAL vy = r.y / a * wm;
    REAL b = sqrt(vx * vx + vy * vy);
    return (REAL2)(vx / b, vy / b);
}

// An input's carrier terms at t (stack.rs compute_task `lagged`): the lag
// product, |z[t]|^2, 1; zero where t or t - LAG is not covered.
void lagged(__global const REAL2* hi, __global const uchar* cov, int t, int len, REAL2* v, REAL* p, REAL* k)
{
    *v = (REAL2)((REAL)0, (REAL)0);
    *p = (REAL)0;
    *k = (REAL)0;
    if (t >= LAG && t < len && cov[t] && cov[t - LAG]) {
        REAL2 x = hi[t], y = hi[t - LAG];
        *v = (REAL2)(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
        *p = x.x * x.x + x.y * x.y;
        *k = (REAL)1;
    }
}

#define TILE 256
// XTALK: the combined video band as well (stack.rs combined_band), each
// input's turned by its lock rotation.
#ifdef XTALK
#define VACC(z, w) vacc += (z) * (w)
#define ROT(z, r) ((REAL2)((z).x * (r).x - (z).y * (r).y, (z).x * (r).y + (z).y * (r).x))
#define VOUT() vout[jo] = wsum > (REAL)0 ? vacc / wsum : (REAL2)((REAL)0, (REAL)0)
#else
#define VACC(z, w)
#define ROT(z, r) (z)
#define VOUT()
#endif
#ifdef CARRIER_COMBINE
// Weight every input by its carrier amplitude over LOCK_WIN samples,
// phase-lock its video band onto the primary's over the same window (the
// rest of each input as read), and sum. A work-group of TILE outputs stages
// the window terms of each input in local memory once, so the 128-sample
// sums read it instead of global memory.
__kernel void combine(
    const int ni,
    const int len,
    const int m,
    const int n,
    __global const REAL* full,
    __global const REAL2* hi,
    __global const uchar* cov,
    __global const REAL* weight,
    __global const REAL* afloor,
    __global float* out
#ifdef XTALK
    , __global REAL2* vout
#endif
    )
{
    // Correlation, both energies and the count over the samples the input
    // shares with the primary; its own carrier terms.
    __local REAL2 prod[TILE + LOCK_WIN], pr[TILE + LOCK_WIN];
    __local REAL pa[TILE + LOCK_WIN], pb[TILE + LOCK_WIN], pj[TILE + LOCK_WIN];
    __local REAL po[TILE + LOCK_WIN], pn[TILE + LOCK_WIN];
    int lid = get_local_id(0);
    int jo = get_global_id(0);
    int j = jo + m;
    int t0 = (int)get_group_id(0) * TILE + m - LOCK_WIN / 2;
    bool active = jo < n;
    REAL acc = (REAL)0, wsum = (REAL)0, q0 = (REAL)0, h0 = (REAL)0;
    REAL2 vacc = (REAL2)((REAL)0, (REAL)0);
    for (int u = lid; u < TILE + LOCK_WIN; u += TILE) {
        REAL2 v;
        REAL p, k;
        lagged(hi, cov, t0 + u, len, &v, &p, &k);
        pr[u] = v;
        po[u] = p;
        pn[u] = k;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (active) {
        REAL2 rl = (REAL2)((REAL)0, (REAL)0), rs = (REAL2)((REAL)0, (REAL)0);
        REAL e = (REAL)0, c = (REAL)0, es = (REAL)0, cs = (REAL)0, q;
        for (int u = 0; u < LOCK_WIN; u++) { rl += pr[lid + u]; e += po[lid + u]; c += pn[lid + u]; }
        for (int u = LOCK_WIN / 2 - CARRIER_WIN / 2; u < LOCK_WIN / 2 + CARRIER_WIN / 2; u++) {
            rs += pr[lid + u]; es += po[lid + u]; cs += pn[lid + u];
        }
        REAL s = cs > (REAL)0 ? carrier(rs, es, cs, (REAL)QN_CARRIER, &q) : (REAL)0;
        if (c > (REAL)0) {
            REAL sl = carrier(rl, e, c, (REAL)QN_LOCK, &q0);
            h0 = sl > (REAL)0 ? s / sl : (REAL)0;
        }
        if (cov[j]) {
            REAL w = amp_weight(s, afloor[0], weight[0]);
            acc += full[j] * w;
            wsum += w;
            VACC(hi[j], w);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 1; i < ni; i++) {
        int b = i * len;
        for (int u = lid; u < TILE + LOCK_WIN; u += TILE) {
            int t = t0 + u;
            REAL2 v = (REAL2)((REAL)0, (REAL)0);
            REAL ea = (REAL)0, eb = (REAL)0, cj = (REAL)0;
            if (t >= 0 && t < len && cov[t] && cov[b + t]) {
                REAL2 x = hi[t], y = hi[b + t];
                v = (REAL2)(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
                ea = x.x * x.x + x.y * x.y;
                eb = y.x * y.x + y.y * y.y;
                cj = (REAL)1;
            }
            prod[u] = v;
            pa[u] = ea;
            pb[u] = eb;
            pj[u] = cj;
            REAL2 lv;
            REAL lp, lk;
            lagged(hi + b, cov + b, t, len, &lv, &lp, &lk);
            pr[u] = lv;
            po[u] = lp;
            pn[u] = lk;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (active && cov[b + j]) {
            REAL2 r = (REAL2)((REAL)0, (REAL)0), rl = (REAL2)((REAL)0, (REAL)0);
            REAL ea = (REAL)0, eb = (REAL)0, nj = (REAL)0, eo = (REAL)0, no = (REAL)0;
            for (int u = 0; u < LOCK_WIN; u++) {
                r += prod[lid + u];
                ea += pa[lid + u];
                eb += pb[lid + u];
                nj += pj[lid + u];
                rl += pr[lid + u];
                eo += po[lid + u];
                no += pn[lid + u];
            }
            REAL2 rs = (REAL2)((REAL)0, (REAL)0);
            REAL es = (REAL)0, cs = (REAL)0;
            for (int u = LOCK_WIN / 2 - CARRIER_WIN / 2; u < LOCK_WIN / 2 + CARRIER_WIN / 2; u++) {
                rs += pr[lid + u]; es += po[lid + u]; cs += pn[lid + u];
            }
            REAL qi = (REAL)0, q, s = (REAL)0, g = (REAL)1;
            if (no > (REAL)0) carrier(rl, eo, no, (REAL)QN_LOCK, &qi);
            if (cs > (REAL)0) s = carrier(rs, es, cs, (REAL)QN_CARRIER, &q);
            REAL2 rot = nj > (REAL)0 ? lock(r, ea, eb, nj, q0, qi, h0, &g) : (REAL2)((REAL)1, (REAL)0);
            REAL w = amp_weight(s, afloor[i], weight[i]) * g;
            REAL2 d = (REAL2)(rot.x - (REAL)1, rot.y);
            REAL2 z = hi[b + j];
            acc += (full[b + j] + (z.x * d.x - z.y * d.y)) * w;
            wsum += w;
            VACC(ROT(z, rot), w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        out[jo] = wsum > (REAL)0 ? (float)(acc / wsum) : 0.0f;
        VOUT();
    }
}
#else
// --stack-weight fixed: phase-lock every input's video band onto the
// primary's over LOCK_WIN samples and sum with the fixed weights (the rest of
// each input as read; `afloor` unused).
__kernel void combine(
    const int ni,
    const int len,
    const int m,
    const int n,
    __global const REAL* full,
    __global const REAL2* hi,
    __global const uchar* cov,
    __global const REAL* weight,
    __global const REAL* afloor,
    __global float* out
#ifdef XTALK
    , __global REAL2* vout
#endif
    )
{
    __local REAL2 prod[TILE + LOCK_WIN];
    int lid = get_local_id(0);
    int jo = get_global_id(0);
    int j = jo + m;
    int t0 = (int)get_group_id(0) * TILE + m - LOCK_WIN / 2;
    bool active = jo < n;
    REAL acc = (REAL)0, wsum = (REAL)0;
    REAL2 vacc = (REAL2)((REAL)0, (REAL)0);
    if (active && cov[j]) { acc += full[j] * weight[0]; wsum += weight[0]; VACC(hi[j], weight[0]); }
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
            VACC(ROT(z, rot), weight[i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) {
        out[jo] = wsum > (REAL)0 ? (float)(acc / wsum) : 0.0f;
        VOUT();
    }
}
#endif
"#;

struct State {
    q: Queue,
    k_read: Kernel,
    k_combine: Kernel,
    /// Audio crosstalk cancellation: the combined video band read back.
    vout: Buffer,
    sinc: Buffer,
    hil: Buffer,
    weight: Buffer,
    afloor: Buffer,
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
                    "-DPHASES={SINC_PHASES} -DSINC_TAPS={SINC_TAPS} -DHI_TAPS={HI_TAPS} -DLOCK_WIN={LOCK_WIN} \
                     -DCARRIER_WIN={CARRIER_WIN} -DQN_LOCK={QN_LOCK:?} -DQN_CARRIER={QN_CARRIER:?} \
                     -DLOCK_PRIOR={LOCK_PRIOR:?} -DKAPPA_LO={KAPPA_LO:?} -DKAPPA_SPAN={KAPPA_SPAN:?} \
                     -DQ_LO={Q_LO:?} -DQ_SPAN={Q_SPAN:?} -DQ_MAX={Q_MAX:?} -DQ_ROT={Q_ROT:?} -DH_LO={H_LO:?} -DH_SPAN={H_SPAN:?} -DLAG={} {} {} {}",
                    inner.lag,
                    if inner.carrier { "-DCARRIER_COMBINE" } else { "" },
                    if inner.wants_band() { "-DXTALK" } else { "" },
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
            vout: gpu.buffer(8)?,
            sinc: constant(&sinc)?,
            hil: constant(&hil)?,
            weight: constant(&weight)?,
            afloor: constant(&inner.afloor)?,
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

    /// Fill `out` with plan samples `m..m + out.len()`, and `v` with the
    /// combined video band there (audio crosstalk cancellation).
    pub(crate) fn run(&self, plan: &Plan, m: usize, out: &mut [f32], v: Option<&mut [Complex64]>) -> Result<()> {
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
        if v.is_some() {
            st.vout.ensure(out.len() * 2 * rsz)?;
        }
        let mut args = vec![
            Arg::I32(ni as i32),
            Arg::I32(len as i32),
            Arg::I32(m as i32),
            Arg::I32(out.len() as i32),
            Arg::Buf(&st.full),
            Arg::Buf(&st.hi),
            Arg::Buf(&st.cov),
            Arg::Buf(&st.weight),
            Arg::Buf(&st.afloor),
            Arg::Buf(&st.out),
        ];
        if v.is_some() {
            args.push(Arg::Buf(&st.vout));
        }
        st.k_combine.args(&args)?;
        st.q.launch(&st.k_combine, &[out.len()], 256)?;
        st.q.read(&st.out, 0, out)?;
        if let Some(v) = v {
            if self.f64 {
                // A `Complex64` is two f64, as the buffer holds them.
                st.q.read(&st.vout, 0, &mut v[..out.len()])?;
            } else {
                let mut b = vec![0.0f32; 2 * out.len()];
                st.q.read(&st.vout, 0, &mut b)?;
                v.par_iter_mut().zip(b.par_chunks_exact(2)).for_each(|(o, c)| *o = Complex64::new(f64::from(c[0]), f64::from(c[1])));
            }
        }
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
