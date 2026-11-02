//! The predictor: exponentially weighted least squares on the past only, so
//! an encoder and a decoder compute the same weights and none are stored.
//!
//! Regressors of sample `i`: the `own` previous samples of the segment, and
//! `2*taps+1` values of the reference window (the mean of the already coded
//! sources, read at the lag tables' positions) centred on `i`. The
//! covariance is updated every `stride` samples and the weights re-solved
//! every `every` samples.
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
//!   (`ridge = trace / d * 1e-6 + 1e-6`), column by column, with forward and
//!   backward substitution that update the remaining unknowns one column at
//!   a time. A matrix that is not positive definite keeps the old weights.

#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub own: usize,
    pub taps: usize,
    pub stride: usize,
    pub every: usize,
    /// Forgetting factor per covariance update, as f32 bits.
    pub lam_bits: u32,
}

impl Preset {
    /// Levels 1 (fast) to 4 (strongest). Memory is 1024 samples throughout.
    pub fn level(level: u32) -> Option<Self> {
        let (own, taps, stride, every) = match level {
            1 => (8, 3, 8, 128),
            2 => (16, 4, 8, 128),
            3 => (16, 6, 4, 64),
            4 => (32, 6, 1, 32),
            _ => return None,
        };
        let lam = (1.0f64 - 1.0 / 1024.0).powi(stride as i32) as f32;
        Some(Self { own, taps, stride, every, lam_bits: lam.to_bits() })
    }

    pub fn dim(&self, use_ref: bool) -> usize {
        self.own + if use_ref { 2 * self.taps + 1 } else { 0 }
    }

    /// Leaves of the dot product tree: the next power of two.
    pub fn leaves(&self, use_ref: bool) -> usize {
        self.dim(use_ref).next_power_of_two().max(2)
    }
}

pub struct Pred {
    pre: Preset,
    use_ref: bool,
    d: usize,
    dp: usize,
    lam: f32,
    a: Vec<f32>,
    b: Vec<f32>,
    w: Vec<f32>,
    r: Vec<f32>,
    t: Vec<f32>,
    m: Vec<f32>,
    s: Vec<f32>,
}

impl Pred {
    pub fn new(pre: Preset, use_ref: bool) -> Self {
        let d = pre.dim(use_ref);
        let dp = pre.leaves(use_ref);
        assert!(d <= 64, "at most 64 regressors");
        Self {
            pre,
            use_ref,
            d,
            dp,
            lam: f32::from_bits(pre.lam_bits),
            a: vec![0.0; d * d],
            b: vec![0.0; d],
            w: vec![0.0; d],
            r: vec![0.0; d],
            t: vec![0.0; dp],
            m: vec![0.0; d * d],
            s: vec![0.0; d],
        }
    }

    /// Prediction of `ys[i]` from `ys[..i]` and `refwin[i..i + 2*taps + 1]`.
    pub fn predict(&mut self, ys: &[i32], i: usize, refwin: &[f32]) -> i32 {
        let own = self.pre.own;
        for p in 0..own {
            self.r[p] = if i > p { ys[i - 1 - p] as f32 } else { 0.0 };
        }
        if self.use_ref {
            let n = 2 * self.pre.taps + 1;
            self.r[own..own + n].copy_from_slice(&refwin[i..i + n]);
        }
        for q in 0..self.dp {
            self.t[q] = if q < self.d { self.r[q] * self.w[q] } else { 0.0 };
        }
        let mut stride = self.dp / 2;
        while stride >= 1 {
            for q in 0..stride {
                let v = self.t[q + stride];
                self.t[q] += v;
            }
            stride /= 2;
        }
        let p = (self.t[0] + 0.5).floor();
        p.clamp(-4_194_304.0, 4_194_304.0) as i32
    }

    /// Learn sample `i` (value `y`) against the regressors of the last `predict`.
    pub fn update(&mut self, i: usize, y: i32) {
        let d = self.d;
        let yf = y as f32;
        if i % self.pre.stride == 0 {
            let lam = self.lam;
            for p in 0..d {
                let rp = self.r[p];
                let t = rp * yf;
                self.b[p] = lam * self.b[p] + t;
                let row = &mut self.a[p * d..p * d + p + 1];
                for (v, &rq) in row.iter_mut().zip(&self.r[..=p]) {
                    let t = rp * rq;
                    *v = lam * *v + t;
                }
            }
        }
        if i % self.pre.every == self.pre.every - 1 && i > 4 * d {
            self.m.copy_from_slice(&self.a);
            self.s.copy_from_slice(&self.b);
            if solve(&mut self.m, &mut self.s, d) {
                self.w.copy_from_slice(&self.s);
            }
        }
    }
}

/// Solve `m x = s` for symmetric positive definite `m` (d x d, row major, the
/// lower triangle is read); `s` receives `x`. False if `m` is not positive
/// definite.
fn solve(m: &mut [f32], s: &mut [f32], d: usize) -> bool {
    let mut tr = 0f32;
    for i in 0..d {
        tr += m[i * d + i];
    }
    let ridge = tr / d as f32 * 1e-6f32 + 1e-6f32;
    for i in 0..d {
        m[i * d + i] += ridge;
    }
    for j in 0..d {
        let mut v = m[j * d + j];
        for k in 0..j {
            let p = m[j * d + k] * m[j * d + k];
            v -= p;
        }
        if v <= 0.0 {
            return false;
        }
        m[j * d + j] = v.sqrt();
        for i in j + 1..d {
            let mut v = m[i * d + j];
            for k in 0..j {
                let p = m[i * d + k] * m[j * d + k];
                v -= p;
            }
            m[i * d + j] = v / m[j * d + j];
        }
    }
    for i in 0..d {
        s[i] /= m[i * d + i];
        for r in i + 1..d {
            let p = m[r * d + i] * s[i];
            s[r] -= p;
        }
    }
    for i in (0..d).rev() {
        s[i] /= m[i * d + i];
        for r in 0..i {
            let p = m[i * d + r] * s[i];
            s[r] -= p;
        }
    }
    true
}
