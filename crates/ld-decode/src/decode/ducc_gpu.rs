//! ducc0's 1-D transform plan, executed on the OpenCL device.
//!
//! The vendored ducc build the decoder calls (`crate::ffi_ducc`) is a baseline
//! SSE2 build whose 1-D entry points run the **scalar** pass type, so every
//! transform it performs is a fixed sequence of plain IEEE double `+`, `-` and
//! `*` — no fused multiply-add, no cross-lane packing. [`ducc_plan`] encodes
//! that sequence (pass order, twiddle tables, the `ip > 10000` two-packet
//! split, the `l1 == 1` in-place radix-8 quirk), and this module runs it on the
//! device: the same butterflies in the same order over the same twiddles give
//! the same bits, whatever the processor.
//!
//! That is what makes `--gpu` (the default GPU demodulator, `demod_gpu_exact`)
//! byte-identical to the CPU kernel rather than merely close. It is *not* what
//! `--gpu-fast` does: that path runs a Stockham radix-8 transform, which is a
//! different arithmetic and rounds differently (`demod_gpu`).
//!
//! Layout: one [`PlanDev`] per transform length, holding the node tree with
//! every pass's twiddle table resident on the device. A plan handles *any*
//! number of independent rows in one launch per pass — `get_global_id(0)` is
//! block-indexed by the row length — which is what makes batching a field's
//! blocks worthwhile (~5 launches per transform, whatever the batch size).
//!
//! The kernels are `#pragma OPENCL FP_CONTRACT OFF`: a `a*b + c` written as
//! two operations must stay two operations, exactly as in the reference.

use std::sync::Arc;

use anyhow::{bail, Result};
use super::ducc_plan::{Body, C2cPlan, Kind, Node, UnityRoots};
use crate::opencl::{Arg, Buffer, Gpu, Kernel, Queue};

const SOURCE: &str = r#"
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define HSQT2 0.707106781186547524400844362104849

inline double2 cadd(double2 a, double2 b) { return (double2)(a.x+b.x, a.y+b.y); }
inline double2 csub(double2 a, double2 b) { return (double2)(a.x-b.x, a.y-b.y); }
inline double2 cmx(double2 a, double2 b)
{ return (double2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }

inline void rot90(double2 *a, int fwd)
{ double t = a->x; if (fwd) { a->x = a->y; a->y = -t; } else { a->x = -a->y; a->y = t; } }

inline void rot45(double2 *a, int fwd)
{
  double t = a->x;
  if (fwd) { a->x = HSQT2*(a->x + a->y); a->y = HSQT2*(a->y - t); }
  else     { a->x = HSQT2*(a->x - a->y); a->y = HSQT2*(a->y + t); }
}

inline void rot135(double2 *a, int fwd)
{
  double t = a->x;
  if (fwd) { a->x = HSQT2*(a->y - a->x); a->y = HSQT2*(-t - a->y); }
  else     { a->x = HSQT2*(-a->x - a->y); a->y = HSQT2*(t - a->y); }
}

inline void pmi(double2 *a, double2 *b)
{ double2 t = *a; *a = cadd(t, *b); *b = csub(t, *b); }

inline double2 smul(double2 v, double2 w, int fwd)
{
  if (fwd) return (double2)(v.x*w.x + v.y*w.y, v.y*w.x - v.x*w.y);
  return (double2)(v.x*w.x - v.y*w.y, v.x*w.y + v.y*w.x);
}

inline double2 uroot(__global const double2 *v1, __global const double2 *v2,
                     int idx, int n, int mask, int shift)
{
  int lo = (idx <= (n >> 1));
  int i = lo ? idx : (n - idx);
  double2 m = cmx(v1[i & mask], v2[i >> shift]);
  return lo ? m : (double2)(m.x, -m.y);
}

// cfftp8::exec_, both halves. `wa` is the twiddle row for this `i`, or 0 for
// the untwiddled form.
inline void bf8(double2 *v, double2 *o, int fwd, __global const double2 *wa)
{
  double2 a0, a1, a2, a3, a4, a5, a6, a7;
  a1 = cadd(v[1], v[5]);
  a5 = csub(v[1], v[5]);
  a3 = cadd(v[3], v[7]);
  a7 = csub(v[3], v[7]);
  rot90(&a7, fwd);
  pmi(&a1, &a3);
  rot90(&a3, fwd);
  pmi(&a5, &a7);
  rot45(&a5, fwd);
  rot135(&a7, fwd);
  a0 = cadd(v[0], v[4]);
  a4 = csub(v[0], v[4]);
  a2 = cadd(v[2], v[6]);
  a6 = csub(v[2], v[6]);
  if (wa == 0)
  {
    o[0] = cadd(cadd(a0, a2), a1);
    o[4] = csub(cadd(a0, a2), a1);
    o[2] = cadd(csub(a0, a2), a3);
    o[6] = csub(csub(a0, a2), a3);
    rot90(&a6, fwd);
    o[1] = cadd(cadd(a4, a6), a5);
    o[5] = csub(cadd(a4, a6), a5);
    o[3] = cadd(csub(a4, a6), a7);
    o[7] = csub(csub(a4, a6), a7);
  }
  else
  {
    pmi(&a0, &a2);
    o[0] = cadd(a0, a1);
    o[4] = smul(csub(a0, a1), wa[3], fwd);
    o[2] = smul(cadd(a2, a3), wa[1], fwd);
    o[6] = smul(csub(a2, a3), wa[5], fwd);
    rot90(&a6, fwd);
    pmi(&a4, &a6);
    o[1] = smul(cadd(a4, a5), wa[0], fwd);
    o[5] = smul(csub(a4, a5), wa[4], fwd);
    o[3] = smul(cadd(a6, a7), wa[2], fwd);
    o[7] = smul(csub(a6, a7), wa[6], fwd);
  }
}

inline void bf4(double2 *v, double2 *o, int fwd, __global const double2 *wa)
{
  double2 t2 = cadd(v[0], v[2]);
  double2 t1 = csub(v[0], v[2]);
  double2 t3 = cadd(v[1], v[3]);
  double2 t4 = csub(v[1], v[3]);
  rot90(&t4, fwd);
  if (wa == 0)
  {
    o[0] = cadd(t2, t3);
    o[1] = cadd(t1, t4);
    o[2] = csub(t2, t3);
    o[3] = csub(t1, t4);
  }
  else
  {
    o[0] = cadd(t2, t3);
    o[1] = smul(cadd(t1, t4), wa[0], fwd);
    o[2] = smul(csub(t2, t3), wa[1], fwd);
    o[3] = smul(csub(t1, t4), wa[2], fwd);
  }
}

// One radix pass over `bt` independent blocks of `ip` points laid out
// contiguously. Item count is bt*l1*ido; the three index maps are
// cfftpN::exec_'s.
#define PASS_BODY(RADIX, BF)                                                  \
  int t = get_global_id(0);                                                   \
  int blk = t / per_block;                                                    \
  int u = t - blk * per_block;                                                \
  int base = blk * ip;                                                        \
  double2 v[RADIX];                                                           \
  double2 o[RADIX];                                                           \
  int b;                                                                      \
  if (l1 == 1) {                                                              \
    int i = u;                                                                \
    for (b = 0; b < RADIX; b++) v[b] = src[base + i + ido*b];                 \
    BF(v, o, fwd, i == 0 ? 0 : wa + (i-1)*(RADIX-1));                         \
    for (b = 0; b < RADIX; b++) dst[base + i + ido*b] = o[b];                 \
  } else if (ido == 1) {                                                      \
    int k = u;                                                                \
    for (b = 0; b < RADIX; b++) v[b] = src[base + b + RADIX*k];               \
    BF(v, o, fwd, 0);                                                         \
    for (b = 0; b < RADIX; b++) dst[base + k + l1*b] = o[b];                  \
  } else {                                                                    \
    int i = u % ido;                                                          \
    int k = u / ido;                                                          \
    for (b = 0; b < RADIX; b++) v[b] = src[base + i + ido*(b + RADIX*k)];     \
    BF(v, o, fwd, i == 0 ? 0 : wa + (i-1)*(RADIX-1));                         \
    for (b = 0; b < RADIX; b++) dst[base + i + ido*(k + l1*b)] = o[b];        \
  }

__kernel void fft_pass8(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{ PASS_BODY(8, bf8) }

__kernel void fft_pass4(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{ PASS_BODY(4, bf4) }

__kernel void fft_pass2(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{
  int t = get_global_id(0);
  int blk = t / per_block;
  int u = t - blk * per_block;
  int base = blk * ip;
  if (l1 == 1) {
    int i = u;
    double2 x0 = src[base + i];
    double2 x1 = src[base + i + ido];
    dst[base + i] = cadd(x0, x1);
    dst[base + i + ido] = (i == 0) ? csub(x0, x1) : smul(csub(x0, x1), wa[i-1], fwd);
  } else if (ido == 1) {
    int k = u;
    double2 x0 = src[base + 2*k];
    double2 x1 = src[base + 2*k + 1];
    dst[base + k] = cadd(x0, x1);
    dst[base + k + l1] = csub(x0, x1);
  } else {
    int i = u % ido;
    int k = u / ido;
    int at = base + i + ido*2*k;
    double2 x0 = src[at];
    double2 x1 = src[at + ido];
    int ot = base + i + ido*k;
    dst[ot] = cadd(x0, x1);
    dst[ot + ido*l1] = (i == 0) ? csub(x0, x1) : smul(csub(x0, x1), wa[i-1], fwd);
  }
}

// cfft_multipass, `l1 == 1 && ido > 1`: blocks gathered at stride `ido`.
__kernel void node_gather(__global const double2 *src, __global double2 *dst,
                          int ip, int ido, int stride)
{
  int per = ido * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int i = t / ip;
  int m = t - i * ip;
  dst[blk*stride + i*ip + m] = src[blk*stride + i + ido*m];
}

// ... and scattered back with the outer twiddles.
__kernel void node_scatter(__global const double2 *src, __global double2 *dst,
                           __global const double2 *v1, __global const double2 *v2,
                           int nroots, int mask, int shift, int rfct,
                           int ip, int ido, int fwd, int stride)
{
  int per = ido * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int i = t / ip;
  int m = t - i * ip;
  double2 val = src[blk*stride + i*ip + m];
  if (i != 0 && m != 0)
    val = smul(val, uroot(v1, v2, rfct*m*i, nroots, mask, shift), fwd);
  dst[blk*stride + i + ido*m] = val;
}

// cfft_multipass, `ido == 1`: contiguous blocks, transposed on the way out.
__kernel void node_transpose(__global const double2 *src, __global double2 *dst,
                             int ip, int l1, int stride)
{
  int per = l1 * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int n = t / ip;
  int m = t - n * ip;
  dst[blk*stride + n + m*l1] = src[blk*stride + n*ip + m];
}

// rfftp_complexify, forward=false: half spectrum -> half-length complex input.
__kernel void c2r_pre(__global const double2 *sp, __global double2 *cch,
                      __global const double2 *v1, __global const double2 *v2,
                      int nroots, int mask, int shift, int n)
{
  int i = get_global_id(0);
  int nh = n >> 1;
  if (i == 0)
  {
    cch[0] = (double2)(sp[0].x + sp[nh].x, sp[0].x - sp[nh].x);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 t1 = (double2)(sp[i].x, sp[i].y);
  double2 t2 = (double2)(sp[xi].x, -sp[xi].y);
  double2 xe = cadd(t1, t2);
  double2 xo = cmx(csub(t1, t2), uroot(v1, v2, i, nroots, mask, shift));
  double2 lo = (double2)(xe.x - xo.y, xe.y + xo.x);
  double2 hi = (double2)(xe.x + xo.y, -xe.y + xo.x);
  cch[i] = lo;
  cch[xi] = hi;
}

// The tail of c2r: `in[i] = res[i]*fct` over all n doubles.
__kernel void c2r_scale(__global const double2 *src, __global double2 *dst,
                        int nh, double fct)
{
  int t = get_global_id(0);
  if (t >= nh) return;
  double2 v = src[t];
  dst[t] = (double2)(v.x*fct, v.y*fct);
}

// The same pre-pack for a row-major batch: row `r` is already the spectrum row.
// `rows` rows of `w = nh + 1` bins in, `rows` rows of `nh` complex out.
__kernel void c2r_pre_r(__global const double2 *sp, __global double2 *cch,
                        __global const double2 *v1, __global const double2 *v2,
                        int nroots, int mask, int shift, int n)
{
  int nh = n >> 1;
  int w = nh + 1;
  int g = get_global_id(0);
  int blk = g / nh;
  int i = g - blk * nh;
  int sb = blk * w;
  int cb = blk * nh;
  if (i == 0)
  {
    cch[cb] = (double2)(sp[sb].x + sp[sb + nh].x, sp[sb].x - sp[sb + nh].x);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 t1 = sp[sb + i];
  double2 t2 = (double2)(sp[sb + xi].x, -sp[sb + xi].y);
  double2 xe = cadd(t1, t2);
  double2 xo = cmx(csub(t1, t2), uroot(v1, v2, i, nroots, mask, shift));
  cch[cb + i] = (double2)(xe.x - xo.y, xe.y + xo.x);
  cch[cb + xi] = (double2)(xe.x + xo.y, -xe.y + xo.x);
}

// rfftp_complexify, forward=true, batched: the half-length complex result ->
// `n/2 + 1` bins, written into full-length (`n`) rows so the caller can mirror
// the upper half in place. `general_r2c`'s packing is folded in: bin k is
// `(out[2k-1], out[2k])`.
__kernel void r2c_post_f(__global const double2 *res, __global double2 *sp,
                         __global const double2 *v1, __global const double2 *v2,
                         int nroots, int mask, int shift, int n)
{
  int nh = n >> 1;
  int w = nh + 1;
  int g = get_global_id(0);
  int blk = g / w;
  int i = g - blk * w;
  int rb = blk * nh;
  int sb = blk * n;
  if (i == 0)
  {
    sp[sb] = (double2)(res[rb].x + res[rb].y, 0.0);
    return;
  }
  if (i == nh)
  {
    sp[sb + nh] = (double2)(res[rb].x - res[rb].y, 0.0);
    return;
  }
  int xi = nh - i;
  // Each mirrored pair must be written by *one* work item, the lower index:
  // `get(nh-j)` and `-conj(get(j))` are equal in exact arithmetic but not
  // after the root table's two rounded factors, and both work items writing
  // would race.
  if (i > xi) return;
  double2 ri = res[rb + i];
  double2 rxi = res[rb + xi];
  double2 xe = (double2)(ri.x + rxi.x, ri.y - rxi.y);
  double2 tw = uroot(v1, v2, i, nroots, mask, shift);
  double2 t = (double2)(ri.y + rxi.y, rxi.x - ri.x);
  double2 xo = (double2)(t.x*tw.x - t.y*(-tw.y), t.x*(-tw.y) + t.y*tw.x);
  sp[sb + i] = (double2)(0.5*(xe.x + xo.x), 0.5*(xe.y + xo.y));
  sp[sb + xi] = (double2)(0.5*(xe.x - xo.x), 0.5*(xo.y - xe.y));
}

// `fft_real_full_into`'s reflection: `X[n-k] = conj(X[k])` for k in 1..n/2.
__kernel void reflect(__global double2 *sp, int n)
{
  int nh = n >> 1;
  int g = get_global_id(0);
  int blk = g / (nh - 1);
  int k = 1 + (g - blk * (nh - 1));
  double2 v = sp[blk*n + k];
  sp[blk*n + n - k] = (double2)(v.x, -v.y);
}
"#;

/// One kernel argument: a buffer (by index into the harness) or a scalar.
pub(super) enum A {
    Buf(usize),
    Int(i32),
    Dbl(f64),
}

/// The bit-exact device harness: one program, its kernels, and every buffer it
/// created. Buffers are addressed by index into `mems` so a plan can be
/// expressed as "buffer 7 in, buffer 3 out" without borrowing fights.
pub(super) struct DuccGpu {
    pub gpu: Arc<Gpu>,
    pub q: Queue,
    kernels: Vec<(&'static str, Kernel)>,
    mems: Vec<Buffer>,
}

impl DuccGpu {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Self> {
        let names = [
            "fft_pass8",
            "fft_pass4",
            "fft_pass2",
            "node_gather",
            "node_scatter",
            "node_transpose",
            "c2r_pre",
            "c2r_scale",
            "c2r_pre_r",
            "r2c_post_f",
            "reflect",
        ];
        let kernels = gpu
            .kernels(SOURCE, "", &names)?
            .into_iter()
            .zip(names)
            .map(|(k, n)| (n, k))
            .collect();
        Ok(Self {
            gpu: Arc::clone(gpu),
            q: gpu.queue()?,
            kernels,
            mems: Vec::new(),
        })
    }

    fn kernel(&self, name: &str) -> Result<&Kernel> {
        match self.kernels.iter().find(|(n, _)| *n == name) {
            Some((_, k)) => Ok(k),
            None => bail!("no kernel `{name}`"),
        }
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<usize> {
        let b = self.gpu.buffer(bytes)?;
        self.mems.push(b);
        Ok(self.mems.len() - 1)
    }

    pub fn buf(&self, idx: usize) -> &Buffer {
        &self.mems[idx]
    }

    pub fn write<T: Copy>(&self, idx: usize, data: &[T]) -> Result<()> {
        self.q.write(&self.mems[idx], 0, data)
    }

    pub fn read<T: Copy>(&self, idx: usize, out: &mut [T]) -> Result<()> {
        self.q.read(&self.mems[idx], 0, out)
    }

    /// Enqueue `name` over exactly `global` work items, binding `args` in order.
    ///
    /// Every argument goes into one `Kernel::args` call: it numbers the
    /// arguments from zero, so binding them one at a time would overwrite slot
    /// 0 each time.
    pub fn dispatch(&self, name: &str, global: usize, args: &[A]) -> Result<()> {
        let k = self.kernel(name)?;
        let mut bound = Vec::with_capacity(args.len());
        for a in args {
            bound.push(match a {
                A::Buf(i) => Arg::Buf(&self.mems[*i]),
                A::Int(v) => Arg::I32(*v),
                A::Dbl(v) => Arg::F64(*v),
            });
        }
        k.args(&bound)?;
        self.q.launch_exact(k, &[global])
    }
}

/// A `UnityRoots` table resident on the device.
pub(super) struct RootsDev {
    v1: usize,
    v2: usize,
    n: i32,
    mask: i32,
    shift: i32,
}

impl RootsDev {
    fn upload(dg: &mut DuccGpu, roots: &UnityRoots) -> Result<Self> {
        let v1 = dg.alloc(roots.v1().len() * 16)?;
        dg.write(v1, roots.v1())?;
        let v2 = dg.alloc(roots.v2().len() * 16)?;
        dg.write(v2, roots.v2())?;
        Ok(Self {
            v1,
            v2,
            n: roots.len() as i32,
            mask: roots.mask() as i32,
            shift: roots.shift() as i32,
        })
    }
}

struct PassDev {
    radix: usize,
    l1: usize,
    ido: usize,
    wa: usize,
}

/// A pre-built node of a plan: its passes with their twiddle tables resident.
struct NodeDev {
    ip: usize,
    l1: usize,
    ido: usize,
    rfct: usize,
    passes: Vec<PassDev>,
    /// Buffer indices of the two packet children (`l1 == ido == 1` only, for
    /// the shapes the decoder uses).
    children: Option<(usize, usize)>,
}

/// Build the device-side plan tree: every pass with its twiddle table resident.
fn build_dev(dg: &mut DuccGpu, nodes: &mut Vec<NodeDev>, node: &Node) -> Result<usize> {
    let slot = nodes.len();
    nodes.push(NodeDev {
        ip: node.ip,
        l1: node.l1,
        ido: node.ido,
        rfct: node.rfct,
        passes: Vec::new(),
        children: None,
    });
    match &node.body {
        Body::Chain(passes) => {
            let mut pd = Vec::with_capacity(passes.len());
            for p in passes {
                let radix = match p.kind {
                    Kind::P8 => 8,
                    Kind::P4 => 4,
                    Kind::P2 => 2,
                };
                // An empty table still needs a valid buffer to bind.
                let wa = dg.alloc(p.wa.len().max(1) * 16)?;
                if !p.wa.is_empty() {
                    dg.write(wa, &p.wa)?;
                }
                pd.push(PassDev {
                    radix,
                    l1: p.l1,
                    ido: p.ido,
                    wa,
                });
            }
            nodes[slot].passes = pd;
        }
        Body::Packets(a, b) => {
            let ia = build_dev(dg, nodes, a)?;
            let ib = build_dev(dg, nodes, b)?;
            nodes[slot].children = Some((ia, ib));
        }
    }
    Ok(slot)
}

/// A plan resident on the device: its node tree, its roots table and the slot
/// of the root node.
pub(super) struct PlanDev {
    nodes: Vec<NodeDev>,
    roots: RootsDev,
    root: usize,
    n: usize,
}

impl PlanDev {
    pub fn build(dg: &mut DuccGpu, plan: &C2cPlan) -> Result<Self> {
        let roots = RootsDev::upload(dg, plan.roots())?;
        let mut nodes = Vec::new();
        let root = build_dev(dg, &mut nodes, &plan.root)?;
        Ok(Self {
            nodes,
            roots,
            root,
            n: plan.n,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn roots(&self) -> &RootsDev {
        &self.roots
    }
}

/// Scratch buffers a transform runs between. The roles are fixed so no kernel
/// ever reads and writes the same element in one launch: `g` is the packet
/// gather, `h` the chain ping-pong partner, `i` the packet scatter and `o` the
/// packet output. Each is `rows_max` transforms wide.
pub(super) struct Scratch {
    pub g: usize,
    pub h: usize,
    pub i: usize,
    pub o: usize,
}

impl Scratch {
    pub fn alloc(dg: &mut DuccGpu, rows_max: usize, n: usize) -> Result<Self> {
        let bytes = rows_max * n * 16;
        Ok(Self {
            g: dg.alloc(bytes)?,
            h: dg.alloc(bytes)?,
            i: dg.alloc(bytes)?,
            o: dg.alloc(bytes)?,
        })
    }
}

/// One pass chain over `bt` contiguous `ip`-blocks, ping-ponging the way
/// `chain_run` does. Returns the buffer index holding the result.
fn dev_chain(
    dg: &DuccGpu,
    nodes: &[NodeDev],
    sc: &Scratch,
    slot: usize,
    bt: usize,
    fwd: bool,
    src: usize,
) -> Result<usize> {
    let node = &nodes[slot];
    let mut cur = src;
    let mut other = sc.h;
    for p in &node.passes {
        // `cfftp8` with `l1 == 1` works in place and does not swap; on the
        // device that is safe because each element belongs to exactly one work
        // item, which reads all eight of its inputs before writing any.
        let in_place = p.radix == 8 && p.l1 == 1;
        let out = if in_place { cur } else { other };
        let name = match p.radix {
            8 => "fft_pass8",
            4 => "fft_pass4",
            _ => "fft_pass2",
        };
        let global = bt * (node.ip / p.radix);
        dg.dispatch(
            name,
            global,
            &[
                A::Buf(cur),
                A::Buf(out),
                A::Buf(p.wa),
                A::Int(node.ip as i32),
                A::Int(p.l1 as i32),
                A::Int(p.ido as i32),
                A::Int(if fwd { 1 } else { 0 }),
                A::Int((p.l1 * p.ido) as i32),
            ],
        )?;
        if !in_place {
            std::mem::swap(&mut cur, &mut other);
        }
    }
    Ok(cur)
}

/// Run a whole plan on the device over `bt` contiguous `ip`-blocks, returning
/// the buffer index of the result.
pub(super) fn dev_node(
    dg: &DuccGpu,
    plan: &PlanDev,
    sc: &Scratch,
    bt: usize,
    fwd: bool,
    src: usize,
) -> Result<usize> {
    dev_node_at(dg, &plan.nodes, &plan.roots, sc, plan.root, bt, fwd, src)
}

fn dev_node_at(
    dg: &DuccGpu,
    nodes: &[NodeDev],
    roots: &RootsDev,
    sc: &Scratch,
    slot: usize,
    bt: usize,
    fwd: bool,
    src: usize,
) -> Result<usize> {
    let node = &nodes[slot];
    let Some((ia, ib)) = node.children else {
        return dev_chain(dg, nodes, sc, slot, bt, fwd, src);
    };
    let (a, b) = (&nodes[ia], &nodes[ib]);
    if a.l1 != 1 || b.ido != 1 {
        bail!("unexpected packet layout ({} / {})", a.l1, b.ido);
    }
    // a: gather the `ido` stride-`ido` blocks, transform them, twiddle back.
    // Every packet kernel block-indexes its global id by the block length, so
    // the same launch covers all `bt` transforms.
    dg.dispatch(
        "node_gather",
        bt * a.ido * a.ip,
        &[
            A::Buf(src),
            A::Buf(sc.g),
            A::Int(a.ip as i32),
            A::Int(a.ido as i32),
            A::Int(node.ip as i32),
        ],
    )?;
    let res_a = dev_chain(dg, nodes, sc, ia, bt * a.ido, fwd, sc.g)?;
    dg.dispatch(
        "node_scatter",
        bt * a.ido * a.ip,
        &[
            A::Buf(res_a),
            A::Buf(sc.i),
            A::Buf(roots.v1),
            A::Buf(roots.v2),
            A::Int(roots.n),
            A::Int(roots.mask),
            A::Int(roots.shift),
            A::Int(node.rfct as i32),
            A::Int(a.ip as i32),
            A::Int(a.ido as i32),
            A::Int(if fwd { 1 } else { 0 }),
            A::Int(node.ip as i32),
        ],
    )?;
    let res_b = dev_chain(dg, nodes, sc, ib, bt * b.l1, fwd, sc.i)?;
    // `b` writes transposed into the dedicated output buffer; `res_b` is only
    // read here, never written, so the transpose cannot alias its source.
    dg.dispatch(
        "node_transpose",
        bt * b.l1 * b.ip,
        &[
            A::Buf(res_b),
            A::Buf(sc.o),
            A::Int(b.ip as i32),
            A::Int(b.l1 as i32),
            A::Int(node.ip as i32),
        ],
    )?;
    Ok(sc.o)
}

/// `rfft` of `rows` real rows of `n` -> `rows` half spectra written into
/// `rows`-by-`n` complex rows. With `mirror`, the upper half is filled with
/// `conj(X[n-k])`, which is what `ffi_ducc::fft_real_full` produces; without
/// it only bins `0..=n/2` are written (its caller reads no more).
#[allow(clippy::too_many_arguments)]
pub(super) fn r2c(
    dg: &DuccGpu,
    plan_half: &PlanDev,
    outer: &RootsDev,
    sc: &Scratch,
    rows: usize,
    n: usize,
    src: usize,
    dst: usize,
    mirror: bool,
) -> Result<()> {
    let nh = n / 2;
    let w = nh + 1;
    if plan_half.n() != nh {
        bail!("r2c plan is for n={}, expected {nh}", plan_half.n());
    }
    let res = dev_node(dg, plan_half, sc, rows, true, src)?;
    dg.dispatch(
        "r2c_post_f",
        rows * w,
        &[
            A::Buf(res),
            A::Buf(dst),
            A::Buf(outer.v1),
            A::Buf(outer.v2),
            A::Int(outer.n),
            A::Int(outer.mask),
            A::Int(outer.shift),
            A::Int(n as i32),
        ],
    )?;
    if mirror {
        dg.dispatch(
            "reflect",
            rows * (nh - 1),
            &[A::Buf(dst), A::Int(n as i32)],
        )?;
    }
    Ok(())
}

/// `ifft` of `rows` full-length complex rows, normalized by `1/n`, into `dst`.
pub(super) fn c2c_inverse(
    dg: &DuccGpu,
    plan: &PlanDev,
    sc: &Scratch,
    rows: usize,
    n: usize,
    src: usize,
    dst: usize,
) -> Result<()> {
    if plan.n() != n {
        bail!("inverse plan is for n={}, expected {n}", plan.n());
    }
    let res = dev_node(dg, plan, sc, rows, false, src)?;
    // `c2r_scale` indexes `double2` elements, so a full complex row is `n` of
    // them (`n/2` doubles are `n/4` complex -- mixing the two counts scales
    // only the first quarter of the data, which is what a `rows*n/2` total did).
    let total = rows * n;
    dg.dispatch(
        "c2r_scale",
        total,
        &[
            A::Buf(res),
            A::Buf(dst),
            A::Int(total as i32),
            A::Dbl(1.0 / n as f64),
        ],
    )
}

/// `irfft` of `rows` half spectra (`rows`-by-`(n/2+1)` complex) into `rows`
/// real rows of `n`, normalized by `1/n`, into `dst`.
#[allow(clippy::too_many_arguments)]
pub(super) fn c2r(
    dg: &DuccGpu,
    plan_half: &PlanDev,
    outer: &RootsDev,
    sc: &Scratch,
    cch: usize,
    rows: usize,
    n: usize,
    sp: usize,
    dst: usize,
) -> Result<()> {
    let nh = n / 2;
    if plan_half.n() != nh {
        bail!("c2r plan is for n={}, expected {nh}", plan_half.n());
    }
    dg.dispatch(
        "c2r_pre_r",
        rows * nh,
        &[
            A::Buf(sp),
            A::Buf(cch),
            A::Buf(outer.v1),
            A::Buf(outer.v2),
            A::Int(outer.n),
            A::Int(outer.mask),
            A::Int(outer.shift),
            A::Int(n as i32),
        ],
    )?;
    let res = dev_node(dg, plan_half, sc, rows, false, cch)?;
    let total = rows * nh;
    dg.dispatch(
        "c2r_scale",
        total,
        &[
            A::Buf(res),
            A::Buf(dst),
            A::Int(total as i32),
            A::Dbl(1.0 / n as f64),
        ],
    )
}

#[cfg(test)]
mod probe {
    use super::*;
    use rustfft::num_complex::Complex64;

    fn read_f64(path: &str) -> Vec<f64> {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        b.chunks_exact(8).map(|w| f64::from_le_bytes(w.try_into().unwrap())).collect()
    }

    /// Offline probe against a `LD_GPU_DUMP` capture: does `ffi_ducc::irfft` of
    /// the device's own product row reproduce the CPU kernel's `out_video`?
    /// If yes, the product is right and the device's `c2r` is what moves; if no,
    /// the product does.
    #[test]
    #[ignore]
    fn probe_c2r_on_dumped_products() {
        let dir = std::env::var("LD_GPU_DUMP").unwrap_or_else(|_| "C:/Software/freebuff_rust_ld-decode/work/dumps".into());
        let n = 32768usize;
        let nh = n / 2;
        let w = nh + 1;
        let vs = read_f64(&format!("{dir}/gpu/gpu_vs.bin"));
        let vreal = read_f64(&format!("{dir}/gpu/gpu_vreal.bin"));
        let cpu0 = read_f64(&format!("{dir}/cpu/s0_out_video.bin"));
        for ch in 0..3 {
            let row: Vec<Complex64> = (0..w)
                .map(|k| Complex64::new(vs[(ch * w + k) * 2], vs[(ch * w + k) * 2 + 1]))
                .collect();
            let want = crate::ffi_ducc::irfft(&row[..], n);
            let gpu = &vreal[ch * n..(ch + 1) * n];
            let eq = want.iter().zip(gpu).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            eprintln!("row {ch}: ffi_ducc::irfft(vs) vs device c2r: {eq}/{n}");
            let refrow: Option<&[f64]> = match ch {
                0 => Some(&cpu0),
                1 => None,
                _ => None,
            };
            if let Some(r) = refrow {
                let eq = want.iter().zip(r).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                eprintln!("row {ch}: ffi_ducc::irfft(vs) vs CPU out_video: {eq}/{n}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::num_complex::Complex64;

    /// The device plan must reproduce `ffi_ducc`'s own 1-D entry points bit for
    /// bit -- all three the demod path uses: the full complex forward/inverse
    /// and the real pair. `LD_GPU_PROBE=1` runs it (there is no GPU on CI).
    #[test]
    #[ignore]
    fn device_plan_matches_ffi_ducc() {
        if std::env::var_os("LD_GPU_PROBE").is_none() {
            return;
        }
        let gpu = Gpu::get().expect("OpenCL device");
        let mut dg = DuccGpu::new(&gpu).expect("harness");
        let n = 32768usize;
        let nh = n / 2;
        let plan_n = PlanDev::build(&mut dg, &C2cPlan::new(n)).expect("plan n");
        let plan_h = PlanDev::build(&mut dg, &C2cPlan::new(nh)).expect("plan nh");
        let sc = Scratch::alloc(&mut dg, 3, n).expect("scratch");
        let cch = dg.alloc(3 * nh * 16).expect("cch");

        // A pseudo-random complex row, deterministic.
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let input: Vec<Complex64> = (0..n).map(|_| Complex64::new(next(), next())).collect();

        // Full complex inverse, through the plan + the 1/n tail.
        let src = dg.alloc(n * 16).expect("src");
        let dst = dg.alloc(n * 16).expect("dst");
        dg.write(src, &input).expect("write");
        c2c_inverse(&dg, &plan_n, &sc, 1, n, src, dst).expect("inverse");
        let mut got = vec![Complex64::new(0.0, 0.0); n];
        dg.read(dst, &mut got).expect("read");
        dg.q.finish().expect("finish");
        let want = crate::ffi_ducc::ifft(&input);
        let bad = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.re.to_bits() != b.re.to_bits() || a.im.to_bits() != b.im.to_bits())
            .count();
        assert_eq!(bad, 0, "complex inverse: {bad} of {n} bins differ");

        // Real forward: `rfft` + reflection, read back as the full spectrum.
        let real: Vec<f64> = (0..n).map(|_| next()).collect();
        let rsrc = dg.alloc(n * 8).expect("rsrc");
        dg.write(rsrc, &real).expect("write");
        r2c(&dg, &plan_h, plan_n.roots(), &sc, 1, n, rsrc, src, true).expect("r2c");
        let mut full = vec![Complex64::new(0.0, 0.0); n];
        dg.read(src, &mut full).expect("read");
        dg.q.finish().expect("finish");
        let want = crate::ffi_ducc::fft_real_full(&real);
        let bad = want
            .iter()
            .zip(&full)
            .filter(|(a, b)| a.re.to_bits() != b.re.to_bits() || a.im.to_bits() != b.im.to_bits())
            .count();
        assert_eq!(bad, 0, "r2c: {bad} of {n} bins differ");

        // Real inverse from the half spectrum, `irfft`.
        let half = crate::ffi_ducc::rfft(&real);
        dg.write(src, &half).expect("write half");
        c2r(&dg, &plan_h, plan_n.roots(), &sc, cch, 1, n, src, dst).expect("c2r");
        let mut out = vec![0.0f64; n];
        dg.read(dst, &mut out).expect("read");
        dg.q.finish().expect("finish");
        let want = crate::ffi_ducc::irfft(&half, n);
        let bad = want
            .iter()
            .zip(&out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(bad, 0, "c2r: {bad} of {n} samples differ");
    }
}
