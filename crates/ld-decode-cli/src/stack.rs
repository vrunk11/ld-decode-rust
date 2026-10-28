//! Time-base correction at read time (`--tbc read`), and multi-source stacking
//! built on it.
//!
//! Pass 0 decodes every input once and keeps, per field, the VBI frame code,
//! the parity and the absolute position of each line. Fields of the same frame
//! are paired across inputs, and the primary's (first input's) lines define a
//! timeline of nominal-length lines. [`VirtualSource`] serves that timeline as
//! an RF stream: every output sample reads each input at its own position for
//! that instant (analytic signal, Kaiser-windowed sinc on the carrier-demixed
//! signal), the inputs are combined coherently with 1/sigma^2 weights, and the
//! real part is returned. The ordinary decoder then runs on it unchanged, so
//! its sync, VBI, dropout and output resampling stages are the same code.
//!
//! The combination splits each input at 3 MHz. Above (video carrier band) the
//! inputs are phase-locked to the primary over 128 samples before the sum, as
//! in the numpy prototype; below (EFM and the analog audio carriers) they are
//! summed as read, since a phase lock tuned on the video carrier would be wrong
//! there.

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context as _, Result};
use ld_decode::{DecoderMetadata, WriteableField};
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::reader::{SampleFormat, SampleSource};

/// FFT length of one read of an input (the span its positions cover in one
/// task, plus `PAD` on each side).
const NFFT: usize = 1 << 17;
/// Virtual samples produced per parallel task. Small enough that one decoder
/// read (~2M samples) splits into more tasks than there are cores.
const TASK: usize = 1 << 16;
/// Input samples kept on each side of a task's span, so the circular FFT's
/// edge effects stay out of the samples that are read.
const PAD: usize = 8192;
/// Virtual samples computed beyond each end of a task for the phase lock.
const LOCK_MARGIN: usize = 128;
/// Phase-lock window (samples), as in the prototype.
const LOCK_WIN: usize = 128;
/// Kaiser-windowed sinc: taps, beta, fractional phases tabulated.
const SINC_TAPS: usize = 32;
const SINC_BETA: f64 = 8.0;
const SINC_PHASES: usize = 1024;
/// Crossover between the summed low band and the phase-locked video band
/// (raised cosine from 2.5 to 3.5 MHz, as the prototype's analytic band edge).
const SPLIT_LO_HZ: f64 = 2.5e6;
const SPLIT_HI_HZ: f64 = 3.5e6;
/// Demix frequencies of the two bands (the interpolation runs on the band
/// shifted near 0 Hz, where the windowed sinc is flattest).
const DEMIX_LO_HZ: f64 = 1.75e6;
const DEMIX_HI_HZ: f64 = 8.5e6;
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
fn sinc_lut() -> &'static [f64] {
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

fn fft_plans() -> &'static (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>) {
    static PLANS: OnceLock<(Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>)> = OnceLock::new();
    PLANS.get_or_init(|| {
        let mut p = FftPlanner::new();
        (p.plan_fft_forward(NFFT), p.plan_fft_inverse(NFFT))
    })
}

/// One input's two bands over a read span, carrier-demixed for interpolation.
struct Bands {
    start: i64,
    lo: Vec<Complex64>,
    hi: Vec<Complex64>,
}

/// Carrier phasors of one band, tabulated so that neither the demix nor the
/// remix calls sin/cos per sample: `down[n] = exp(-j w n)` over a read span,
/// `frac[p] = exp(j w p / SINC_PHASES)` for the sinc's fractional phases.
struct Mixer {
    down: Vec<Complex64>,
    frac: Vec<Complex64>,
}

impl Mixer {
    fn new(hz: f64, fs: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * hz / fs;
        let down = (0..NFFT).map(|n| Complex64::from_polar(1.0, -w * n as f64)).collect();
        let frac = (0..=SINC_PHASES).map(|p| Complex64::from_polar(1.0, w * p as f64 / SINC_PHASES as f64)).collect();
        Self { down, frac }
    }
}

impl Bands {
    fn read(file: &RawFile, start: i64, fs: f64, mix: &[Mixer; 2]) -> Result<Self> {
        let mut raw = vec![0f32; NFFT];
        file.read_at(start, &mut raw)?;
        let (fwd, inv) = fft_plans();
        let mut x: Vec<Complex64> = raw.iter().map(|&v| Complex64::new(f64::from(v), 0.0)).collect();
        fwd.process(&mut x);
        let mut lo = vec![Complex64::new(0.0, 0.0); NFFT];
        let mut hi = vec![Complex64::new(0.0, 0.0); NFFT];
        lo[0] = x[0];
        hi[NFFT / 2] = x[NFFT / 2];
        for b in 1..NFFT / 2 {
            let f = b as f64 * fs / NFFT as f64;
            let g = if f <= SPLIT_LO_HZ {
                0.0
            } else if f >= SPLIT_HI_HZ {
                1.0
            } else {
                0.5 * (1.0 - (std::f64::consts::PI * (f - SPLIT_LO_HZ) / (SPLIT_HI_HZ - SPLIT_LO_HZ)).cos())
            };
            hi[b] = x[b] * (2.0 * g);
            lo[b] = x[b] * (2.0 * (1.0 - g));
        }
        inv.process(&mut lo);
        inv.process(&mut hi);
        let scale = 1.0 / NFFT as f64;
        demix(&mut lo, &mix[0], scale);
        demix(&mut hi, &mix[1], scale);
        Ok(Self { start, lo, hi })
    }
}

/// `z[n] *= scale * exp(-j w n)` (n local to the span).
fn demix(z: &mut [Complex64], mix: &Mixer, scale: f64) {
    for (v, d) in z.iter_mut().zip(&mix.down) {
        *v *= *d * scale;
    }
}

/// Band-limited value of a demixed band at local position `q`, remixed. The
/// remix uses the same 1/SINC_PHASES-quantized position as the sinc row, so
/// the carrier phase and the interpolated envelope refer to one instant.
fn interp(zb: &[Complex64], q: f64, mix: &Mixer) -> Complex64 {
    let lut = sinc_lut();
    let k0 = q.floor();
    let p = ((q - k0) * SINC_PHASES as f64).round() as usize;
    let k = k0 as i64 - (SINC_TAPS / 2 - 1) as i64;
    // Row SINC_PHASES is the integer shift by one: the table holds it.
    let row = &lut[p * SINC_TAPS..(p + 1) * SINC_TAPS];
    let mut acc = Complex64::new(0.0, 0.0);
    for (j, h) in row.iter().enumerate() {
        acc += zb[(k + j as i64) as usize] * *h;
    }
    // exp(j w (k0 + p/P)) = conj(down[k0]) * frac[p].
    acc * mix.down[k0 as usize].conj() * mix.frac[p]
}

struct Inner {
    inputs: Vec<StackInput>,
    timeline: Timeline,
    weight: Vec<f64>,
    fs: f64,
    len: u64,
    /// Low band, video band.
    mix: [Mixer; 2],
}

/// The read-time corrected (and, with several inputs, combined) RF stream.
pub struct VirtualSource {
    inner: Arc<Inner>,
    pos: u64,
    /// Its own pool on every core: the global rayon pool is the decoder's
    /// small tail pool (a quarter of the demod budget).
    pool: rayon::ThreadPool,
}

impl VirtualSource {
    pub fn new(inputs: Vec<StackInput>, fs: f64, linelen: f64) -> Result<Self> {
        let timeline = Timeline::build(&inputs, linelen)?;
        let len = *timeline.t.last().unwrap() as u64;
        // 1/sigma^2 from each input's noise floor, measured mid-way through
        // the part of it that the timeline uses.
        let mut weight = Vec::new();
        for (i, s) in inputs.iter().enumerate() {
            let used: Vec<f64> = timeline.pos[i].iter().copied().filter(|p| p.is_finite()).collect();
            let mid = used.get(used.len() / 2).copied().unwrap_or(0.0) as i64;
            let sig2 = noise_floor(&s.reader, mid - (NFFT / 2) as i64, fs)?;
            tracing::info!("stack: {} noise floor {:.4e} (weight {:.4e})", s.path, sig2, 1.0 / sig2);
            weight.push(1.0 / sig2);
        }
        tracing::info!(
            "stack: {} input(s), read-time TBC over {} lines ({:.2} s)",
            inputs.len(),
            timeline.t.len(),
            len as f64 / fs
        );
        let mix = [Mixer::new(DEMIX_LO_HZ, fs), Mixer::new(DEMIX_HI_HZ, fs)];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        Ok(Self { inner: Arc::new(Inner { inputs, timeline, weight, fs, len, mix }), pos: 0, pool })
    }

    fn compute(&self, s0: u64, out: &mut [f32]) -> Result<()> {
        let inner = &self.inner;
        self.pool.install(|| {
            out.par_chunks_mut(TASK)
                .enumerate()
                .try_for_each(|(c, chunk)| compute_task(inner, s0 + (c * TASK) as u64, chunk))
        })
    }
}

fn noise_floor(file: &RawFile, start: i64, fs: f64) -> Result<f64> {
    let mut raw = vec![0f32; NFFT];
    file.read_at(start, &mut raw)?;
    let (fwd, _) = fft_plans();
    let mut x: Vec<Complex64> = raw.iter().map(|&v| Complex64::new(f64::from(v), 0.0)).collect();
    fwd.process(&mut x);
    let (lo, hi) = NOISE_BAND_HZ;
    let (mut acc, mut n) = (0.0, 0usize);
    for (b, v) in x.iter().enumerate().take(NFFT / 2) {
        let f = b as f64 * fs / NFFT as f64;
        if f > lo && f < hi {
            acc += v.norm_sqr();
            n += 1;
        }
    }
    if n == 0 {
        bail!("input rate {fs} Hz too low for the noise-floor band");
    }
    Ok((acc / n as f64 / NFFT as f64).max(1e-12))
}

fn compute_task(inner: &Inner, s0: u64, out: &mut [f32]) -> Result<()> {
    let n = out.len();
    let m = LOCK_MARGIN;
    let len = n + 2 * m;
    let tl = &inner.timeline;
    let ni = inner.inputs.len();

    // Ideal time of each sample and its anchor segment.
    let times: Vec<f64> = (0..len).map(|j| s0 as f64 - m as f64 + j as f64).collect();
    let mut segs = Vec::with_capacity(len);
    let mut k = tl.segment(times[0]);
    for &s in &times {
        while k + 2 < tl.t.len() && tl.t[k + 1] <= s {
            k += 1;
        }
        segs.push(k);
    }

    // Each input read at its own positions: low band and video band. The
    // inputs are independent, so they run in parallel too.
    type Span = (Vec<Complex64>, Vec<Complex64>, Vec<bool>);
    let reads: Vec<Span> = inner
        .inputs
        .par_iter()
        .enumerate()
        .map(|(i, input)| -> Result<Span> {
            let p: Vec<f64> = times.iter().zip(&segs).map(|(&s, &k)| tl.position(i, k, s)).collect();
            let (mut pmin, mut pmax) = (f64::INFINITY, f64::NEG_INFINITY);
            for &v in p.iter().filter(|v| v.is_finite()) {
                pmin = pmin.min(v);
                pmax = pmax.max(v);
            }
            let mut zl = vec![Complex64::new(0.0, 0.0); len];
            let mut zh = vec![Complex64::new(0.0, 0.0); len];
            let mut c = vec![false; len];
            if pmin.is_finite() {
                let start = pmin.floor() as i64 - PAD as i64;
                if pmax.ceil() as i64 + PAD as i64 > start + NFFT as i64 {
                    bail!("stack: {} moves too fast for one read span at sample {s0}", input.path);
                }
                let b = Bands::read(&input.reader, start, inner.fs, &inner.mix)?;
                for j in 0..len {
                    if p[j].is_finite() {
                        let q = p[j] - b.start as f64;
                        zl[j] = interp(&b.lo, q, &inner.mix[0]);
                        zh[j] = interp(&b.hi, q, &inner.mix[1]);
                        c[j] = true;
                    }
                }
            }
            Ok((zl, zh, c))
        })
        .collect::<Result<_>>()?;
    let mut lo: Vec<Vec<Complex64>> = Vec::with_capacity(ni);
    let mut hi: Vec<Vec<Complex64>> = Vec::with_capacity(ni);
    let mut cov: Vec<Vec<bool>> = Vec::with_capacity(ni);
    for (zl, zh, c) in reads {
        lo.push(zl);
        hi.push(zh);
        cov.push(c);
    }

    // Phase lock of the video band onto the primary: the correlation with the
    // primary summed over a LOCK_WIN window (running sums), normalized.
    let half = LOCK_WIN / 2;
    let mut rot: Vec<Vec<Complex64>> = vec![Vec::new(); ni];
    for i in 1..ni {
        let mut cs = Vec::with_capacity(len + 1);
        let mut run = Complex64::new(0.0, 0.0);
        cs.push(run);
        for t in 0..len {
            if cov[0][t] && cov[i][t] {
                run += hi[0][t] * hi[i][t].conj();
            }
            cs.push(run);
        }
        rot[i] = (0..len)
            .map(|j| {
                let r = cs[(j + half).min(len)] - cs[j.saturating_sub(half)];
                let a = r.norm();
                if a > 0.0 { r / a } else { Complex64::new(1.0, 0.0) }
            })
            .collect();
    }

    for j in m..m + n {
        let (mut acc, mut wsum) = (Complex64::new(0.0, 0.0), 0.0);
        for i in 0..ni {
            if !cov[i][j] {
                continue;
            }
            let z = if i == 0 { lo[0][j] + hi[0][j] } else { lo[i][j] + hi[i][j] * rot[i][j] };
            acc += z * inner.weight[i];
            wsum += inner.weight[i];
        }
        out[j - m] = if wsum > 0.0 { (acc.re / wsum) as f32 } else { 0.0 };
    }
    Ok(())
}

impl SampleSource for VirtualSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let n = (self.inner.len.saturating_sub(self.pos) as usize).min(out.len());
        if n > 0 {
            self.compute(self.pos, &mut out[..n])?;
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
