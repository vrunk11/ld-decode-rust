//! OpenCL backend of the stack's virtual source (`--device opencl`), on the
//! decoder crate's shared OpenCL layer. The kernels evaluate exactly the plan
//! the CPU path evaluates (`stack::Plan`): the same positions (in double
//! precision), the same sinc phase, the same tables, in single precision for
//! the dot products.

use std::sync::Mutex;

use anyhow::Result;
use ld_decode::opencl::{Arg, Buffer, Gpu, Kernel, Queue};

use crate::stack::{sinc_lut, Inner, Plan, HI_TAPS, LOCK_WIN, SINC_PHASES, SINC_TAPS};

const SOURCE: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

// One input read at its virtual samples: the full signal (sinc) and the
// analytic video band, as stack.rs compute_task does.
__kernel void read_input(
    __global const float* raw,
    __global const uint* seg,
    __global const double* seg_t0,
    __global const double* a0,
    __global const double* slope,
    const double s0,
    const int len,
    const int base,
    __global const float* sinc,
    __global const float2* hil,
    __global float* full,
    __global float2* hi,
    __global uchar* cov)
{
    int j = get_global_id(0);
    if (j >= len) return;
    uint k = seg[j];
    double a = a0[k];
    if (isnan(a)) {
        full[base + j] = 0.0f;
        hi[base + j] = (float2)(0.0f, 0.0f);
        cov[base + j] = 0;
        return;
    }
    double p = a + ((s0 + (double)j) - seg_t0[k]) * slope[k];
    double kf = floor(p);
    int k0 = (int)kf;
    int ph = (int)round((p - kf) * (double)PHASES);
    __global const float* xs = raw + (k0 - (SINC_TAPS / 2 - 1));
    __global const float* hs = sinc + ph * SINC_TAPS;
    float f = 0.0f;
    for (int d = 0; d < SINC_TAPS; d++) f += hs[d] * xs[d];
    __global const float* xh = raw + (k0 - (HI_TAPS / 2 - 1));
    __global const float2* hh = hil + ph * HI_TAPS;
    float2 h = (float2)(0.0f, 0.0f);
    for (int d = 0; d < HI_TAPS; d++) h += hh[d] * xh[d];
    full[base + j] = f;
    hi[base + j] = h;
    cov[base + j] = 1;
}

// Phase-lock every input's video band onto the primary's over LOCK_WIN
// samples and sum with the weights (the rest of each input as read).
__kernel void combine(
    const int ni,
    const int len,
    const int m,
    const int n,
    __global const float* full,
    __global const float2* hi,
    __global const uchar* cov,
    __global const float* weight,
    __global float* out)
{
    int jo = get_global_id(0);
    if (jo >= n) return;
    int j = jo + m;
    float acc = 0.0f, wsum = 0.0f;
    if (cov[j]) { acc += full[j] * weight[0]; wsum += weight[0]; }
    int lo = max(0, j - LOCK_WIN / 2), up = min(len, j + LOCK_WIN / 2);
    for (int i = 1; i < ni; i++) {
        int b = i * len;
        if (!cov[b + j]) continue;
        float2 r = (float2)(0.0f, 0.0f);
        for (int t = lo; t < up; t++) {
            if (cov[t] && cov[b + t]) {
                float2 x = hi[t], y = hi[b + t];
                r += (float2)(x.x * y.x + x.y * y.y, x.y * y.x - x.x * y.y);
            }
        }
        float a = length(r);
        float2 rot = a > 0.0f ? r / a : (float2)(1.0f, 0.0f);
        float2 d = rot - (float2)(1.0f, 0.0f);
        float2 z = hi[b + j];
        acc += (full[b + j] + (z.x * d.x - z.y * d.y)) * weight[i];
        wsum += weight[i];
    }
    out[jo] = wsum > 0.0f ? acc / wsum : 0.0f;
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
    a0: Buffer,
    slope: Buffer,
    full: Buffer,
    hi: Buffer,
    cov: Buffer,
    out: Buffer,
}

pub struct StackGpu {
    ni: usize,
    state: Mutex<State>,
}

impl StackGpu {
    pub(crate) fn new(inner: &Inner) -> Result<Self> {
        let gpu = Gpu::get()?;
        tracing::info!("stack: OpenCL device {}", gpu.name);
        let mut ks = gpu
            .kernels(
                SOURCE,
                &format!("-DPHASES={SINC_PHASES} -DSINC_TAPS={SINC_TAPS} -DHI_TAPS={HI_TAPS} -DLOCK_WIN={LOCK_WIN}"),
                &["read_input", "combine"],
            )?
            .into_iter();
        let (k_read, k_combine) = (ks.next().unwrap(), ks.next().unwrap());
        let q = gpu.queue()?;
        let constant = |data: &[f32]| -> Result<Buffer> {
            let b = gpu.buffer(std::mem::size_of_val(data))?;
            q.write(&b, 0, data)?;
            Ok(b)
        };
        let sinc: Vec<f32> = sinc_lut().iter().map(|&v| v as f32).collect();
        let hil: Vec<f32> = inner.hi_lut.iter().flat_map(|c| [c.re as f32, c.im as f32]).collect();
        let wmax = inner.weight.iter().copied().fold(0.0, f64::max);
        let weight: Vec<f32> = inner.weight.iter().map(|&w| (w / wmax) as f32).collect();
        let state = State {
            sinc: constant(&sinc)?,
            hil: constant(&hil)?,
            weight: constant(&weight)?,
            seg: gpu.buffer(4)?,
            seg_t0: gpu.buffer(8)?,
            raw: gpu.buffer(4)?,
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
        Ok(Self { ni: inner.weight.len(), state: Mutex::new(state) })
    }

    /// Fill `out` with plan samples `m..m + out.len()`.
    pub(crate) fn run(&self, plan: &Plan, m: usize, out: &mut [f32]) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        let (len, ni) = (plan.len, self.ni);
        st.seg.ensure(len * 4)?;
        st.seg_t0.ensure(plan.seg_t0.len() * 8)?;
        st.full.ensure(ni * len * 4)?;
        st.hi.ensure(ni * len * 8)?;
        st.cov.ensure(ni * len)?;
        st.out.ensure(out.len() * 4)?;
        st.q.write(&st.seg, 0, &plan.seg)?;
        st.q.write(&st.seg_t0, 0, &plan.seg_t0)?;
        for (i, span) in plan.inputs.iter().enumerate() {
            let Some(span) = span else {
                // Not covered anywhere in this read.
                st.q.write(&st.cov, i * len, &vec![0u8; len])?;
                continue;
            };
            st.raw.ensure(span.raw.len() * 4)?;
            st.a0.ensure(span.a0.len() * 8)?;
            st.slope.ensure(span.slope.len() * 8)?;
            st.q.write(&st.raw, 0, &span.raw)?;
            st.q.write(&st.a0, 0, &span.a0)?;
            st.q.write(&st.slope, 0, &span.slope)?;
            st.k_read.args(&[
                Arg::Buf(&st.raw),
                Arg::Buf(&st.seg),
                Arg::Buf(&st.seg_t0),
                Arg::Buf(&st.a0),
                Arg::Buf(&st.slope),
                Arg::F64(plan.s0),
                Arg::I32(len as i32),
                Arg::I32((i * len) as i32),
                Arg::Buf(&st.sinc),
                Arg::Buf(&st.hil),
                Arg::Buf(&st.full),
                Arg::Buf(&st.hi),
                Arg::Buf(&st.cov),
            ])?;
            st.q.launch(&st.k_read, &[len], 256)?;
            // The raw/a0/slope buffers are reused by the next input.
            st.q.finish()?;
        }
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
        st.q.read(&st.out, 0, out)
    }
}
