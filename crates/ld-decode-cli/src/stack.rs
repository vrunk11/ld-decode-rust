//! Time-base correction at read time (`--tbc read`), and multi-source stacking
//! built on it.
//!
//! Pass 0 decodes every input once and keeps, per field, the VBI frame code,
//! the parity and the absolute position of each line. Fields of the same frame
//! are paired across inputs, and the primary's (first input's) lines define a
//! timeline of nominal-length lines. [`VirtualSource`] serves that timeline as
//! an RF stream: every output sample reads each input at its own position for
//! that instant, the inputs are combined with 1/sigma^2 weights, and the
//! ordinary decoder then runs on it unchanged, so its sync, VBI, dropout and
//! output resampling stages are the same code.
//!
//! Each input is read with two filters evaluated at the fractional position:
//! a Kaiser-windowed sinc (the full signal) and an analytic video-band filter
//! (above ~3 MHz). The video band is phase-locked to the primary's over 128
//! samples before the sum, as in the numpy prototype; the rest (full minus
//! video band: EFM and the analog audio carriers) is summed as read, since a
//! phase lock tuned on the video carrier would be wrong there. Both filters are
//! plain dot products, so the same plan runs on the CPU or, with
//! `--stack-device opencl`, on a GPU (`opencl.rs`).

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context as _, Result};
use ld_decode::{DecoderMetadata, WriteableField};
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

use crate::reader::{SampleFormat, SampleSource};

/// Virtual samples produced per parallel CPU task. Small enough that one
/// decoder read (~2M samples) splits into more tasks than there are cores.
const TASK: usize = 1 << 16;
/// Largest input span one read may cover; an input that jumps further within
/// one read (a skip in that capture) is left out of that read.
const MAX_SPAN: usize = 1 << 25;
/// FFT length of the noise-floor estimate.
const NOISE_FFT: usize = 1 << 17;
/// Virtual samples computed beyond each end of a task for the phase lock.
const LOCK_MARGIN: usize = 128;
/// Phase-lock window (samples), as in the prototype.
pub(crate) const LOCK_WIN: usize = 128;
/// Kaiser-windowed sinc: taps, beta, fractional phases tabulated.
pub(crate) const SINC_TAPS: usize = 32;
const SINC_BETA: f64 = 8.0;
pub(crate) const SINC_PHASES: usize = 1024;
/// Crossover between the summed low band and the phase-locked video band
/// (raised cosine from 2.5 to 3.5 MHz, as the prototype's analytic band edge).
pub(crate) const SPLIT_LO_HZ: f64 = 2.5e6;
pub(crate) const SPLIT_HI_HZ: f64 = 3.5e6;
/// Video-band filter: taps and Kaiser beta. Only the crossover's shape depends
/// on it (the low band is the full signal minus this band), and 128 taps put
/// the leakage across the crossover near -37 dB.
pub(crate) const HI_TAPS: usize = 128;
const HI_BETA: f64 = 8.0;
/// Band used to estimate each input's noise floor (no signal there).
const NOISE_BAND_HZ: (f64, f64) = (16.0e6, 19.5e6);
/// A paired line whose length differs from the primary's by more than this
/// ratio is treated as a skip in that input (not covered).
const MAX_LINE_RATIO_DEV: f64 = 0.02;

/// Per-field record kept from pass 0.
#[derive(Clone)]
pub struct FieldRec {
    pub first: bool,
    /// Frame code words (VBI) of the frame this field belongs to.
    pub code: Option<Vec<i64>>,
    /// Absolute position of each line (input samples).
    pub lines: Vec<f64>,
}

/// Pass-0 sink: keeps the field records, writes nothing.
#[derive(Default)]
pub struct Pass0Collector {
    pub fields: Vec<FieldRec>,
}

impl Pass0Collector {
    pub fn put(&mut self, field: &WriteableField) {
        let info = field.info();
        let code = if info.is_first_field {
            info.vbi.as_ref().map(|v| frame_code(&v.vbi_data)).filter(|c| !c.is_empty())
        } else {
            // A second field belongs to the frame its first field opened.
            self.fields.last().filter(|p| p.first).and_then(|p| p.code.clone())
        };
        let base = info.file_loc as f64;
        self.fields.push(FieldRec {
            first: info.is_first_field,
            code,
            lines: field.linelocs.iter().map(|l| base + l).collect(),
        });
    }

    pub fn finish(&mut self, _metadata: Option<DecoderMetadata>) {}
}

/// The VBI words that identify a frame: CAV picture numbers and CLV time codes
/// (top nibble 0xF or 0x8), sorted. Two captures of the same frame give the
/// same set; a misread gives a different one, so the field is just left
/// unpaired.
fn frame_code(words: &[i64]) -> Vec<i64> {
    let mut v: Vec<i64> = words
        .iter()
        .copied()
        .filter(|&w| w > 0 && matches!((w >> 20) & 0xF, 0x8 | 0xF))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// One input: its pass-0 fields and a random-access reader.
pub struct StackInput {
    pub path: String,
    pub fields: Vec<FieldRec>,
    reader: RawFile,
}

impl StackInput {
    pub fn new(path: &str, format: SampleFormat, fields: Vec<FieldRec>) -> Result<Self> {
        Ok(Self { path: path.to_string(), fields, reader: RawFile::open(path, format)? })
    }
}

/// Fixed-width raw capture read by sample index (positioned reads, so the
/// parallel tasks share one handle).
struct RawFile {
    file: File,
    bytes: usize,
    widen: fn(&[u8], &mut [f32]),
    len: u64,
}

impl RawFile {
    fn open(path: &str, format: SampleFormat) -> Result<Self> {
        let Some((bytes, widen)) = crate::reader::raw_layout(format) else {
            bail!("--tbc read needs a fixed-width raw input (s16, r16, r8, s8, rf); {path} is {format:?}");
        };
        let file = File::open(path).with_context(|| format!("opening {path}"))?;
        let len = file.metadata()?.len() / bytes as u64;
        Ok(Self { file, bytes, widen, len })
    }

    /// Samples `[start, start + out.len())`; zeros outside the file.
    fn read_at(&self, start: i64, out: &mut [f32]) -> Result<()> {
        out.fill(0.0);
        let lo = start.max(0) as u64;
        let hi = ((start + out.len() as i64).max(0) as u64).min(self.len);
        if hi <= lo {
            return Ok(());
        }
        let n = (hi - lo) as usize;
        let mut buf = vec![0u8; n * self.bytes];
        let mut filled = 0usize;
        while filled < buf.len() {
            let got = read_at(&self.file, &mut buf[filled..], lo * self.bytes as u64 + filled as u64)?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        let off = (lo as i64 - start) as usize;
        (self.widen)(&buf[..filled / self.bytes * self.bytes], &mut out[off..off + filled / self.bytes]);
        Ok(())
    }
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

/// Line anchors of the virtual timeline: ideal time of each anchor and, per
/// input, its absolute position there (NaN = not covered).
struct Timeline {
    t: Vec<f64>,
    pos: Vec<Vec<f64>>,
}

impl Timeline {
    fn build(inputs: &[StackInput], linelen: f64) -> Result<Self> {
        // Pair every input's fields with the primary's by (frame code, parity);
        // a code seen twice in one input is ambiguous and left out.
        let index = |fields: &[FieldRec]| {
            let mut m: HashMap<(Vec<i64>, bool), Option<usize>> = HashMap::new();
            for (j, f) in fields.iter().enumerate() {
                if let Some(c) = &f.code {
                    m.entry((c.clone(), f.first)).and_modify(|e| *e = None).or_insert(Some(j));
                }
            }
            m
        };
        let maps: Vec<_> = inputs.iter().map(|s| index(&s.fields)).collect();
        let prim = &inputs[0].fields;

        let mut t = Vec::new();
        let mut pos: Vec<Vec<f64>> = vec![Vec::new(); inputs.len()];
        let mut last = f64::NEG_INFINITY;
        let mut paired = vec![0usize; inputs.len()];
        for f in prim {
            let matched: Vec<Option<&FieldRec>> = inputs
                .iter()
                .zip(&maps)
                .map(|(s, m)| {
                    let j = *f.code.as_ref().and_then(|c| m.get(&(c.clone(), f.first)))?;
                    j.map(|j| &s.fields[j])
                })
                .collect();
            for (i, m) in matched.iter().enumerate() {
                paired[i] += usize::from(m.is_some());
            }
            for (l, &a) in f.lines.iter().enumerate() {
                // Fields overlap by a few lines; keep each line once, in order.
                if a <= last + 0.25 * linelen {
                    continue;
                }
                let tk = match t.last() {
                    None => 0.0,
                    Some(&tp) => {
                        let lines = ((2.0 * (a - last) / linelen).round() / 2.0).max(0.5);
                        tp + lines * linelen
                    }
                };
                t.push(tk);
                pos[0].push(a);
                for i in 1..inputs.len() {
                    pos[i].push(matched[i].and_then(|g| g.lines.get(l).copied()).unwrap_or(f64::NAN));
                }
                last = a;
            }
        }
        if t.len() < 2 {
            bail!("pass 0 found no usable lines in the primary input");
        }
        for (i, s) in inputs.iter().enumerate().skip(1) {
            tracing::info!(
                "stack: {} paired {} of the primary's {} fields",
                s.path,
                paired[i],
                prim.len()
            );
        }
        Ok(Self { t, pos })
    }

    /// Anchor segment containing ideal time `s` (last segment past the end).
    fn segment(&self, s: f64) -> usize {
        match self.t.partition_point(|&x| x <= s) {
            0 => 0,
            k => (k - 1).min(self.t.len() - 2),
        }
    }

    /// Position of input `i` at ideal time `s` within segment `k`, or NaN.
    fn position(&self, i: usize, k: usize, s: f64) -> f64 {
        let (a0, a1) = (self.pos[i][k], self.pos[i][k + 1]);
        if !(a0.is_finite() && a1.is_finite()) {
            return f64::NAN;
        }
        if i > 0 {
            let (p0, p1) = (self.pos[0][k], self.pos[0][k + 1]);
            let ratio = (a1 - a0) / (p1 - p0);
            if (ratio - 1.0).abs() > MAX_LINE_RATIO_DEV {
                return f64::NAN;
            }
        }
        let u = (s - self.t[k]) / (self.t[k + 1] - self.t[k]);
        a0 + u * (a1 - a0)
    }
}

/// Windowed-sinc table: `SINC_PHASES + 1` rows of `SINC_TAPS` taps for taps
/// at offsets `-(SINC_TAPS/2 - 1) ..= SINC_TAPS/2`, each row normalized to a
/// unit sum.
pub(crate) fn sinc_lut() -> &'static [f64] {
    static LUT: OnceLock<Vec<f64>> = OnceLock::new();
    LUT.get_or_init(|| {
        let h = (SINC_TAPS / 2) as f64;
        let i0b = bessel_i0(SINC_BETA);
        let mut lut = Vec::with_capacity((SINC_PHASES + 1) * SINC_TAPS);
        for p in 0..=SINC_PHASES {
            let a = p as f64 / SINC_PHASES as f64;
            let row: Vec<f64> = (0..SINC_TAPS)
                .map(|j| {
                    let x = (j as f64 - (h - 1.0)) - a;
                    let sinc = if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                    let r = (1.0 - (x / h).powi(2)).max(0.0);
                    sinc * bessel_i0(SINC_BETA * r.sqrt()) / i0b
                })
                .collect();
            let sum: f64 = row.iter().sum();
            lut.extend(row.iter().map(|v| v / sum));
        }
        lut
    })
}

fn bessel_i0(x: f64) -> f64 {
    let x2 = x * x / 4.0;
    let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
    loop {
        term *= x2 / (k * k);
        if term < sum * 1e-17 {
            return sum;
        }
        sum += term;
        k += 1.0;
    }
}

/// `∫_a^b exp(j c f) df`.
fn int_exp(c: f64, a: f64, b: f64) -> Complex64 {
    if (c * (b - a)).abs() < 1e-9 {
        return Complex64::from_polar(b - a, c * 0.5 * (a + b));
    }
    (Complex64::from_polar(1.0, c * b) - Complex64::from_polar(1.0, c * a)) / Complex64::new(0.0, c)
}

/// Impulse response, at `tau` input samples, of the analytic video band:
/// `2 g(f)` for f > 0 and 0 below, `g` rising as a raised cosine over
/// `SPLIT_LO_HZ..SPLIT_HI_HZ`, flat, then falling over `top` (a raised cosine
/// too, so the response stays short). Closed form, so the table is exact.
fn video_band_response(tau: f64, fs: f64, top: (f64, f64)) -> Complex64 {
    let pi = std::f64::consts::PI;
    let c = 2.0 * pi * tau / fs;
    // Raised-cosine edge over [f0, f1]: 0.5 + s * 0.5 cos(pi (f - f0) / B),
    // s = -1 rising, +1 falling.
    let edge = |f0: f64, f1: f64, s: f64| {
        let k = pi / (f1 - f0);
        let cosine = Complex64::from_polar(0.5, -k * f0) * int_exp(c + k, f0, f1)
            + Complex64::from_polar(0.5, k * f0) * int_exp(c - k, f0, f1);
        int_exp(c, f0, f1) * 0.5 + cosine * (0.5 * s)
    };
    let sum = edge(SPLIT_LO_HZ, SPLIT_HI_HZ, -1.0) + int_exp(c, SPLIT_HI_HZ, top.0) + edge(top.0, top.1, 1.0);
    sum * (2.0 / fs)
}

/// The video-band filter at the sinc's fractional phases: `SINC_PHASES + 1`
/// rows of `HI_TAPS` taps, tap `d` weighting input sample
/// `k0 - (HI_TAPS/2 - 1) + d` for a position `k0 + p/SINC_PHASES`, Kaiser
/// windowed.
fn video_band_lut(fs: f64) -> Vec<Complex64> {
    let top = ((18.0e6f64).min(0.4 * fs), (22.0e6f64).min(0.5 * fs));
    let h = (HI_TAPS / 2) as f64;
    let i0b = bessel_i0(HI_BETA);
    let mut lut = Vec::with_capacity((SINC_PHASES + 1) * HI_TAPS);
    for p in 0..=SINC_PHASES {
        let a = p as f64 / SINC_PHASES as f64;
        for d in 0..HI_TAPS {
            // q - n for n = k0 - (h - 1) + d and q = k0 + a.
            let tau = a + (h - 1.0) - d as f64;
            let r = (1.0 - (tau / h).powi(2)).max(0.0);
            lut.push(video_band_response(tau, fs, top) * (bessel_i0(HI_BETA * r.sqrt()) / i0b));
        }
    }
    lut
}

/// Integer sample and sinc phase of a position.
#[inline]
fn split_pos(p: f64) -> (i64, usize) {
    let k0 = p.floor();
    (k0 as i64, ((p - k0) * SINC_PHASES as f64).round() as usize)
}

pub(crate) struct Inner {
    inputs: Vec<StackInput>,
    timeline: Timeline,
    pub(crate) weight: Vec<f64>,
    len: u64,
    /// Video-band filter table (see [`video_band_lut`]).
    pub(crate) hi_lut: Vec<Complex64>,
}

/// What one span of virtual samples needs from the inputs: the anchor segment
/// of each sample and, per input, a raw span with the linear position law on
/// each segment. The CPU and OpenCL paths both work from it.
pub(crate) struct Plan {
    /// Ideal time of sample 0.
    pub(crate) s0: f64,
    pub(crate) len: usize,
    /// Per sample, its segment (index into `seg_t0`).
    pub(crate) seg: Vec<u32>,
    /// Ideal time of each segment's first anchor.
    pub(crate) seg_t0: Vec<f64>,
    pub(crate) inputs: Vec<Option<InputSpan>>,
}

pub(crate) struct InputSpan {
    pub(crate) raw: Vec<f32>,
    /// Per segment: position (relative to the span) at `seg_t0`, NaN where the
    /// input does not cover it, and input samples per virtual sample.
    pub(crate) a0: Vec<f64>,
    pub(crate) slope: Vec<f64>,
}

impl InputSpan {
    #[inline]
    fn position(&self, plan: &Plan, j: usize) -> f64 {
        let k = plan.seg[j] as usize;
        self.a0[k] + (plan.s0 + j as f64 - plan.seg_t0[k]) * self.slope[k]
    }
}

impl Inner {
    fn plan(&self, s0: f64, len: usize) -> Result<Plan> {
        let tl = &self.timeline;
        let k_first = tl.segment(s0);
        let mut seg = Vec::with_capacity(len);
        let mut k = k_first;
        for j in 0..len {
            let s = s0 + j as f64;
            while k + 2 < tl.t.len() && tl.t[k + 1] <= s {
                k += 1;
            }
            seg.push((k - k_first) as u32);
        }
        let nseg = k - k_first + 1;
        let seg_t0 = tl.t[k_first..k_first + nseg].to_vec();
        // Sample range of each segment, for the span bounds.
        let mut range = vec![(usize::MAX, 0usize); nseg];
        for (j, &g) in seg.iter().enumerate() {
            let r = &mut range[g as usize];
            r.0 = r.0.min(j);
            r.1 = j;
        }
        let inputs = (0..self.inputs.len())
            .into_par_iter()
            .map(|i| -> Result<Option<InputSpan>> {
                let mut a0 = Vec::with_capacity(nseg);
                let mut slope = Vec::with_capacity(nseg);
                let (mut pmin, mut pmax) = (f64::INFINITY, f64::NEG_INFINITY);
                for g in 0..nseg {
                    let kk = k_first + g;
                    let (pa, pb) = (tl.position(i, kk, tl.t[kk]), tl.position(i, kk, tl.t[kk + 1]));
                    let sl = (pb - pa) / (tl.t[kk + 1] - tl.t[kk]);
                    a0.push(pa);
                    slope.push(sl);
                    if pa.is_finite() && range[g].0 != usize::MAX {
                        for j in [range[g].0, range[g].1] {
                            let p = pa + (s0 + j as f64 - tl.t[kk]) * sl;
                            pmin = pmin.min(p);
                            pmax = pmax.max(p);
                        }
                    }
                }
                if !pmin.is_finite() {
                    return Ok(None);
                }
                let margin = (HI_TAPS / 2 + 2) as i64;
                let start = pmin.floor() as i64 - margin;
                let end = pmax.floor() as i64 + margin;
                if (end - start) as usize > MAX_SPAN {
                    tracing::warn!(
                        "stack: {} jumps {:.0} samples within one read near virtual sample {s0:.0}; left out there",
                        self.inputs[i].path,
                        pmax - pmin
                    );
                    return Ok(None);
                }
                let mut raw = vec![0f32; (end - start) as usize];
                self.inputs[i].reader.read_at(start, &mut raw)?;
                for v in &mut a0 {
                    *v -= start as f64;
                }
                Ok(Some(InputSpan { raw, a0, slope }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Plan { s0, len, seg, seg_t0, inputs })
    }
}

/// The read-time corrected (and, with several inputs, combined) RF stream.
pub struct VirtualSource {
    inner: Arc<Inner>,
    pos: u64,
    /// Its own pool on every core: the global rayon pool is the decoder's
    /// small tail pool (a quarter of the demod budget).
    pool: rayon::ThreadPool,
    gpu: Option<crate::opencl::StackGpu>,
}

impl VirtualSource {
    pub fn new(inputs: Vec<StackInput>, fs: f64, linelen: f64, opencl: bool) -> Result<Self> {
        let timeline = Timeline::build(&inputs, linelen)?;
        let len = *timeline.t.last().unwrap() as u64;
        // 1/sigma^2 from each input's noise floor, measured mid-way through
        // the part of it that the timeline uses.
        let mut weight = Vec::new();
        for (i, s) in inputs.iter().enumerate() {
            let used: Vec<f64> = timeline.pos[i].iter().copied().filter(|p| p.is_finite()).collect();
            let mid = used.get(used.len() / 2).copied().unwrap_or(0.0) as i64;
            let sig2 = noise_floor(&s.reader, mid - (NOISE_FFT / 2) as i64, fs)?;
            tracing::info!("stack: {} noise floor {:.4e} (weight {:.4e})", s.path, sig2, 1.0 / sig2);
            weight.push(1.0 / sig2);
        }
        tracing::info!(
            "stack: {} input(s), read-time TBC over {} lines ({:.2} s)",
            inputs.len(),
            timeline.t.len(),
            len as f64 / fs
        );
        let hi_lut = video_band_lut(fs);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let inner = Inner { inputs, timeline, weight, len, hi_lut };
        let gpu = if opencl { Some(crate::opencl::StackGpu::new(&inner)?) } else { None };
        Ok(Self { inner: Arc::new(inner), pos: 0, pool, gpu })
    }

    fn compute(&self, s0: u64, out: &mut [f32]) -> Result<()> {
        let inner = &self.inner;
        self.pool.install(|| match &self.gpu {
            Some(gpu) => {
                let m = LOCK_MARGIN;
                let t0 = std::time::Instant::now();
                let plan = inner.plan(s0 as f64 - m as f64, out.len() + 2 * m)?;
                let t1 = std::time::Instant::now();
                gpu.run(&plan, m, out)?;
                if std::env::var_os("LD_TIMING").is_some() {
                    ld_decode::teeprintln!("STACKGPU plan_us={} gpu_us={}", (t1 - t0).as_micros(), t1.elapsed().as_micros());
                }
                Ok(())
            }
            None => out
                .par_chunks_mut(TASK)
                .enumerate()
                .try_for_each(|(c, chunk)| compute_task(inner, s0 + (c * TASK) as u64, chunk)),
        })
    }
}

fn noise_floor(file: &RawFile, start: i64, fs: f64) -> Result<f64> {
    let mut raw = vec![0f32; NOISE_FFT];
    file.read_at(start, &mut raw)?;
    let fwd = FftPlanner::new().plan_fft_forward(NOISE_FFT);
    let mut x: Vec<Complex64> = raw.iter().map(|&v| Complex64::new(f64::from(v), 0.0)).collect();
    fwd.process(&mut x);
    let (lo, hi) = NOISE_BAND_HZ;
    let (mut acc, mut n) = (0.0, 0usize);
    for (b, v) in x.iter().enumerate().take(NOISE_FFT / 2) {
        let f = b as f64 * fs / NOISE_FFT as f64;
        if f > lo && f < hi {
            acc += v.norm_sqr();
            n += 1;
        }
    }
    if n == 0 {
        bail!("input rate {fs} Hz too low for the noise-floor band");
    }
    Ok((acc / n as f64 / NOISE_FFT as f64).max(1e-12))
}

fn compute_task(inner: &Inner, s0: u64, out: &mut [f32]) -> Result<()> {
    let n = out.len();
    let m = LOCK_MARGIN;
    let len = n + 2 * m;
    let plan = inner.plan(s0 as f64 - m as f64, len)?;
    let sinc = sinc_lut();

    // Each input read at its own positions: the full signal (sinc) and the
    // analytic video band (the low band is their difference).
    type Read = (Vec<f64>, Vec<Complex64>, Vec<bool>);
    let reads: Vec<Read> = plan
        .inputs
        .par_iter()
        .map(|span| {
            let mut full = vec![0.0; len];
            let mut hi = vec![Complex64::new(0.0, 0.0); len];
            let mut cov = vec![false; len];
            let Some(span) = span else { return (full, hi, cov) };
            for j in 0..len {
                let p = span.position(&plan, j);
                if !p.is_finite() {
                    continue;
                }
                let (k0, ph) = split_pos(p);
                let xs = &span.raw[(k0 - (SINC_TAPS / 2 - 1) as i64) as usize..][..SINC_TAPS];
                full[j] = sinc[ph * SINC_TAPS..][..SINC_TAPS].iter().zip(xs).map(|(h, &x)| h * f64::from(x)).sum();
                let xs = &span.raw[(k0 - (HI_TAPS / 2 - 1) as i64) as usize..][..HI_TAPS];
                let mut acc = Complex64::new(0.0, 0.0);
                for (h, &x) in inner.hi_lut[ph * HI_TAPS..][..HI_TAPS].iter().zip(xs) {
                    acc += h * f64::from(x);
                }
                hi[j] = acc;
                cov[j] = true;
            }
            (full, hi, cov)
        })
        .collect();

    // Phase lock of the video band onto the primary: the correlation with the
    // primary summed over a LOCK_WIN window (running sums), normalized. The
    // rest of each input (full minus video band) is summed as read.
    let half = LOCK_WIN / 2;
    let (full0, hi0, cov0) = &reads[0];
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    for j in m..m + n {
        if cov0[j] {
            acc[j - m] += full0[j] * inner.weight[0];
            wsum[j - m] += inner.weight[0];
        }
    }
    for (i, (full, hi, cov)) in reads.iter().enumerate().skip(1) {
        let mut cs = Vec::with_capacity(len + 1);
        let mut run = Complex64::new(0.0, 0.0);
        cs.push(run);
        for t in 0..len {
            if cov0[t] && cov[t] {
                run += hi0[t] * hi[t].conj();
            }
            cs.push(run);
        }
        for j in m..m + n {
            if !cov[j] {
                continue;
            }
            let r = cs[(j + half).min(len)] - cs[j.saturating_sub(half)];
            let a = r.norm();
            let rot = if a > 0.0 { r / a } else { Complex64::new(1.0, 0.0) };
            acc[j - m] += (full[j] + (hi[j] * (rot - 1.0)).re) * inner.weight[i];
            wsum[j - m] += inner.weight[i];
        }
    }
    for (o, (a, w)) in out.iter_mut().zip(acc.iter().zip(&wsum)) {
        *o = if *w > 0.0 { (a / w) as f32 } else { 0.0 };
    }
    Ok(())
}

impl SampleSource for VirtualSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let n = (self.inner.len.saturating_sub(self.pos) as usize).min(out.len());
        if n > 0 {
            let t0 = std::time::Instant::now();
            self.compute(self.pos, &mut out[..n])?;
            if std::env::var_os("LD_TIMING").is_some() {
                ld_decode::teeprintln!("STACK read at={} n={} us={}", self.pos, n, t0.elapsed().as_micros());
            }
            self.pos += n as u64;
        }
        Ok(n)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        self.pos = sample.min(self.inner.len);
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        true
    }
}
