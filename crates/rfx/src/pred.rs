//! The predictor: exponentially weighted least squares on the past only, so
//! an encoder and a decoder compute the same weights and none are stored.
//!
//! Regressors of sample `i`: the `own` previous samples of the segment; `2*taps+1`
//! values of the reference window centred on `i` (the weighted mean of the
//! already coded sources, read at the lag tables' positions); and for each of
//! the `ntop` best references `2*ttaps+1` values of that reference alone, so
//! the fit can weigh the sources itself. The covariance is updated every
//! `stride` samples and the weights re-solved every `every` samples.
//!
//! Arithmetic is fixed so that any implementation (this one, the OpenCL
//! kernel) gets the same bits:
//! - everything is f32 and every multiply, add, subtract, divide and square
//!   root is rounded on its own: no fused multiply-add, no reassociation;
//! - the prediction is the dot product of regressors and weights summed as a
//!   binary tree over `dp` leaves (`dp` = the number of regressors rounded up to a power of two; the
//!   leaves past the last regressor are 0): leaf `q` plus leaf `q + dp/2`,
//!   then `dp/4`, down to 1; it is rounded as `floor(dot + 0.5)`;
//! - the covariance is `a = lam * a + r_p * r_q` (lower triangle), the right
//!   hand side `b = lam * b + r_p * y`;
//! - the weights come from a Cholesky factorisation of `a + ridge * I`
//!   (`ridge = trace / d * 1e-6 + 1e-6`), column by column (the pivot is
//!   inverted once and the column and the substitutions multiply by it), with forward and
//!   backward substitution that update the remaining unknowns one column at
//!   a time. A matrix that is not positive definite keeps the old weights.

#[derive(Clone, Copy, Debug)]
pub struct Preset {
    /// Own past samples used with references.
    pub own: usize,
    /// Own past samples used when there are none (a primary source).
    pub solo: usize,
    /// Own past samples with the plain mean of the references (no channels).
    pub plain: usize,
    /// The mean of the references has `2 * taps + 1` values around the sample.
    pub taps: usize,
    /// References that also get a channel of their own, with `2 * ttaps + 1`
    /// values each.
    pub ntop: usize,
    pub ttaps: usize,
    pub stride: usize,
    pub every: usize,
    /// Forgetting factor per covariance update, as f32 bits.
    pub lam_bits: u32,
}

impl Preset {
    /// Levels 1 (fast) to 4 (strongest). Memory is 1024 samples throughout.
    pub fn level(level: u32) -> Option<Self> {
        let (own, solo, plain, taps, ntop, stride, every) = match level {
            1 => (6, 8, 8, 3, 2, 8, 128),
            2 => (8, 16, 16, 4, 3, 4, 64),
            3 => (8, 32, 16, 6, 3, 1, 16),
            4 => (12, 48, 24, 6, 4, 1, 16),
            _ => return None,
        };
        let lam = (1.0f64 - 1.0 / 1024.0).powi(stride as i32) as f32;
        Some(Self { own, solo, plain, taps, ntop, ttaps: 1, stride, every, lam_bits: lam.to_bits() })
    }

    /// Own past samples in use.
    pub fn own_for(&self, use_ref: bool) -> usize {
        if use_ref {
            self.own
        } else {
            self.solo
        }
    }

    /// Channels of reference window: the mean, then one per top reference.
    pub fn chans(&self) -> usize {
        1 + self.ntop
    }

    /// Values of a reference window for `n` samples, all channels: each is
    /// `n + 2 * taps` long, or the stride `l` between channels when given.
    pub fn window_len(&self, n: usize) -> usize {
        self.chans() * (n + 2 * self.taps)
    }

    /// `stride` and `every` are powers of two (tested with a mask).
    pub fn dim(&self, use_ref: bool) -> usize {
        self.own_for(use_ref) + if use_ref { 2 * self.taps + 1 + self.ntop * (2 * self.ttaps + 1) } else { 0 }
    }

    /// Leaves of the dot product tree: the next power of two.
    pub fn leaves(&self, use_ref: bool) -> usize {
        self.dim(use_ref).next_power_of_two().max(2)
    }
}

pub struct Pred {
    pre: Preset,
    use_ref: bool,
    /// Own past samples in use.
    own: usize,
    d: usize,
    dp: usize,
    /// Row length of the matrices (`d`).
    ds: usize,
    lam: f32,
    a: Vec<f32>,
    b: Vec<f32>,
    /// Weights and regressors: the part past `d` stays 0, which makes the
    /// padded leaves of the dot product 0 without a test.
    w: Vec<f32>,
    r: Vec<f32>,
    lc: Vec<f32>,
    mr: Vec<f32>,
    rdiag: Vec<f32>,

    s: Vec<f32>,
}

impl Pred {
    pub fn new(pre: Preset, use_ref: bool) -> Self {
        let own = pre.own_for(use_ref);
        let d = pre.dim(use_ref);
        let dp = pre.leaves(use_ref);
        assert!(d <= 64, "at most 64 regressors");
        let ds = d;
        Self {
            pre,
            use_ref,
            own,
            d,
            dp,
            ds,
            lam: f32::from_bits(pre.lam_bits),
            a: vec![0.0; ds * ds],
            b: vec![0.0; d],
            w: vec![0.0; dp.max(ds)],
            r: vec![0.0; dp.max(ds)],
            lc: vec![0.0; ds * ds],
            mr: vec![0.0; ds * ds],
            rdiag: vec![0.0; d],
            s: vec![0.0; ds],
        }
    }

    /// Prediction of sample `i` from the samples learnt so far (`update`) and
    /// the reference window: channel `c` starts at `c * ls`; the mean (channel
    /// 0) is read at `[i..i + 2*taps + 1]`, a top reference (channels 1..)
    /// at the middle `2*ttaps + 1` of the same span.
    pub fn predict(&mut self, refwin: &[f32], ls: usize, i: usize) -> i32 {
        let own = self.own;
        if self.use_ref {
            let n = 2 * self.pre.taps + 1;
            self.r[own..own + n].copy_from_slice(&refwin[i..i + n]);
            let (m, tt) = (2 * self.pre.ttaps + 1, self.pre.taps - self.pre.ttaps);
            for c in 1..=self.pre.ntop {
                let at = own + n + (c - 1) * m;
                let from = c * ls + i + tt;
                self.r[at..at + m].copy_from_slice(&refwin[from..from + m]);
            }
        }
        let dot = match self.dp {
            2 => tree::<1>(&self.r, &self.w),
            4 => tree::<2>(&self.r, &self.w),
            8 => tree::<4>(&self.r, &self.w),
            16 => tree::<8>(&self.r, &self.w),
            32 => tree::<16>(&self.r, &self.w),
            _ => tree::<32>(&self.r, &self.w),
        };
        let p = (dot + 0.5).floor();
        p.clamp(-4_194_304.0, 4_194_304.0) as i32
    }

    /// Learn sample `i` (value `y`) against the regressors of the last `predict`.
    pub fn update(&mut self, i: usize, y: i32) {
        let (d, ds) = (self.d, self.ds);
        let yf = y as f32;
        if i & (self.pre.stride - 1) == 0 {
            let lam = self.lam;
            for p in 0..d {
                let rp = self.r[p];
                let t = rp * yf;
                let lb = lam * self.b[p];
                self.b[p] = lb + t;
let row = &mut self.a[p * ds..p * ds + p + 1];                for (v, &rq) in row.iter_mut().zip(&self.r[..=p]) {                    let t = rp * rq;                    let la = lam * *v;                    *v = la + t;                }
            }
        }
        if i & (self.pre.every - 1) == self.pre.every - 1 && i > 4 * d {
            self.s[..d].copy_from_slice(&self.b);
            if solve(&self.a, &mut self.lc, &mut self.mr, &mut self.rdiag, &mut self.s, d) {
                self.w[..d].copy_from_slice(&self.s[..d]);
            }
        }
        // The newest sample becomes the first of the own history.
        let own = self.own;
        self.r.copy_within(0..own - 1, 1);
        self.r[0] = yf;
    }
}

/// Solve `(a + ridge I) x = s` for the symmetric positive definite `a`
/// (d x d, row major, the lower triangle is read); `s` receives `x`. False if
/// the matrix is not positive definite.
///
/// The factor is kept by column in `lc` (`lc[k * d + i]` is row `i` of column
/// `k`) so that the work below a pivot runs along contiguous memory, and by
/// row in `mr` for the backward substitution. Each element still sees the
/// same operations in the same order as the textbook row loops, so the bits
/// are the reference's.
fn solve(a: &[f32], lc: &mut [f32], mr: &mut [f32], rdiag: &mut [f32], s: &mut [f32], d: usize) -> bool {
    let mut tr = 0f32;
    for i in 0..d {
        tr += a[i * d + i];
    }
    let ridge = tr / d as f32 * 1e-6f32 + 1e-6f32;
    for p in 0..d {
        for q in 0..=p {
            lc[q * d + p] = a[p * d + q];
        }
        lc[p * d + p] += ridge;
    }
    for j in 0..d {
        let (head, tail) = lc.split_at_mut(j * d);
        let mut v = tail[j];
        for k in 0..j {
            let p = head[k * d + j] * head[k * d + j];
            v -= p;
        }
        if v <= 0.0 {
            return false;
        }
        let dj = v.sqrt();
        let rd = 1.0f32 / dj;
        tail[j] = dj;
        rdiag[j] = rd;
        let col = &mut tail[j + 1..d];
        for k in 0..j {
            let ljk = head[k * d + j];
            for (x, &c) in col.iter_mut().zip(&head[k * d + j + 1..k * d + d]) {
                let p = c * ljk;
                *x -= p;
            }
        }
        for x in col.iter_mut() {
            *x *= rd;
        }
        mr[j * d + j] = dj;
        for (i, &x) in col.iter().enumerate() {
            mr[(j + 1 + i) * d + j] = x;
        }
    }
    for i in 0..d {
        s[i] *= rdiag[i];
        let si = s[i];
        for (x, &c) in s[i + 1..d].iter_mut().zip(&lc[i * d + i + 1..i * d + d]) {
            let p = c * si;
            *x -= p;
        }
    }
    for i in (0..d).rev() {
        s[i] *= rdiag[i];
        let si = s[i];
        for (x, &c) in s[..i].iter_mut().zip(&mr[i * d..i * d + i]) {
            let p = c * si;
            *x -= p;
        }
    }
    true
}

/// Dot product of `r` and `w` (2 * H leaves) as the specified binary tree:
/// leaf `q` plus leaf `q + H`, then the halves of the result, down to one.
/// The sizes are constants so the partial sums stay in registers.
#[inline(always)]
fn tree<const H: usize>(r: &[f32], w: &[f32]) -> f32 {
    let (r0, r1) = r[..2 * H].split_at(H);
    let (w0, w1) = w[..2 * H].split_at(H);
    let mut t = [0f32; H];
    for q in 0..H {
        let x = r0[q] * w0[q];
        let y = r1[q] * w1[q];
        t[q] = x + y;
    }
    let mut n = H / 2;
    while n >= 1 {
        for q in 0..n {
            let v = t[q + n];
            t[q] += v;
        }
        n /= 2;
    }
    t[0]
}
