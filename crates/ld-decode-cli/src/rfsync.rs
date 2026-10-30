//! Pass 0 straight from the RF: the lines of an input are
//! located without demodulating it.
//!
//! The carrier sits at its lowest frequency during a sync pulse (-40 IRE,
//! ~7.6 MHz on NTSC, against ~8.1 MHz at blanking), so a coarse frequency
//! estimate is enough to find every pulse: a short complex band-pass FIR
//! evaluated only every `COARSE_STEP` samples (two neighbouring outputs give
//! the phase step). Pulse lengths tell hsyncs from the equalizing and broad
//! pulses of the vertical interval, and where that interval starts against
//! the line grid gives the field parity.
//!
//! Each hsync is then refined as the decoder refines it, on the instantaneous
//! frequency of a longer band-pass FIR computed only around the pulse: the
//! 50% crossing between the sync tip and the back porch, then the colour
//! burst's phase (least squares at fsc), which fixes the line to a fraction of
//! a sample. The burst's half-cycle ambiguity is resolved by the edge, which
//! is far closer than a quarter cycle; the edge's own noise cancels out.
//! Philips codes are read from the few lines after each vertical interval.
//!
//! Every input, the primary included, goes through the same rules, so a given
//! line of the disc gets the same index and the same convention in each.

use std::f64::consts::PI;
use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::stack::{bessel_i0, frame_code, FieldRec, Live, StackInput};

/// NTSC colour subcarrier (Hz).
const FSC: f64 = 315.0e6 / 88.0;
/// Carrier frequency step per IRE (Hz).
const HZ_IRE: f64 = 1.7e6 / 140.0;
/// Nominal blanking and sync tip carrier frequencies (Hz).
const BLANK_HZ: f64 = 8.1e6;
const SYNC_HZ: f64 = BLANK_HZ - 40.0 * HZ_IRE;

/// Coarse detector: taps, decimation, pass band.
const COARSE_TAPS: usize = 48;
const COARSE_STEP: usize = 32;
const COARSE_BAND: (f64, f64) = (4.5e6, 12.5e6);
/// Fine stage: taps and pass band (keeps the burst sidebands around the
/// blanking carrier, 4.5 to 11.7 MHz, and rejects the 2.8 MHz audio carrier).
const FINE_TAPS: usize = 191;
const FINE_BAND: (f64, f64) = (3.8e6, 13.0e6);
/// Samples processed per step, and the context read on each side of them.
const CHUNK: usize = 1 << 22;
const MARGIN: usize = 1 << 17;
/// Lines each field record carries past the next field's first line, as the
/// decoder's do.
const OVERLAP_LINES: usize = 10;
/// Lines (after the field's first) where Philips codes are looked for.
const VBI_LINES: (usize, usize) = (12, 20);

/// A complex band-pass FIR (Kaiser-windowed), as separate re/im taps.
struct Fir {
    /// Taps, zero-padded to a multiple of 8 (the dot product's lanes).
    re: Vec<f32>,
    im: Vec<f32>,
    half: usize,
}

impl Fir {
    fn new(taps: usize, (lo, hi): (f64, f64), beta: f64, fs: f64) -> Self {
        let half = (taps - 1) as f64 / 2.0;
        let (bw, f0) = ((hi - lo) / fs, (hi + lo) / 2.0 / fs);
        let i0b = bessel_i0(beta);
        let (mut re, mut im) = (Vec::with_capacity(taps), Vec::with_capacity(taps));
        for k in 0..taps {
            let m = k as f64 - half;
            let x = PI * bw * m;
            let sinc = if m == 0.0 { 1.0 } else { x.sin() / x };
            let r = m / half;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            let a = bw * sinc * w;
            // Reversed, so the output is a plain dot product with the input.
            let ph = -2.0 * PI * f0 * m;
            re.push((a * ph.cos()) as f32);
            im.push((a * ph.sin()) as f32);
        }
        let padded = taps.next_multiple_of(8);
        re.resize(padded, 0.0);
        im.resize(padded, 0.0);
        Self { re, im, half: (taps - 1) / 2 }
    }

    fn half(&self) -> usize {
        self.half
    }

    /// Output centred on `x[c]` (`c >= half`, `c + half + 8 < x.len()`).
    #[inline]
    fn at(&self, x: &[f32], c: usize) -> (f32, f32) {
        let w = &x[c - self.half..c - self.half + self.re.len()];
        // Eight independent lanes, so the sums vectorize.
        let (mut a, mut b) = ([0.0f32; 8], [0.0f32; 8]);
        for ((v, r), i) in w.chunks_exact(8).zip(self.re.chunks_exact(8)).zip(self.im.chunks_exact(8)) {
            for l in 0..8 {
                a[l] += v[l] * r[l];
                b[l] += v[l] * i[l];
            }
        }
        (a.iter().sum(), b.iter().sum())
    }

    /// Instantaneous frequency (Hz) at `c0..c1` (from the phase step between
    /// neighbouring outputs).
    fn freq(&self, x: &[f32], c0: usize, c1: usize, fs: f64) -> Vec<f32> {
        let k = (fs / (2.0 * PI)) as f32;
        let mut prev = self.at(x, c0 - 1);
        (c0..c1)
            .map(|c| {
                let z = self.at(x, c);
                let (re, im) = (z.0 * prev.0 + z.1 * prev.1, z.1 * prev.0 - z.0 * prev.1);
                prev = z;
                im.atan2(re) * k
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// Horizontal sync (~4.7 us).
    H,
    /// Equalizing pulse (~2.3 us).
    E,
    /// Broad pulse of the vertical sync (~27 us).
    V,
}

/// A sync pulse: absolute start (coarse, samples) and kind.
#[derive(Clone, Copy, Debug)]
struct Pulse {
    start: f64,
    kind: Kind,
}

/// The fine measurement of one hsync.
#[derive(Clone, Copy, Debug, Default)]
struct Fine {
    /// 50% crossing of the sync's leading edge (absolute samples).
    edge: Option<f64>,
    /// Burst phase relative to the edge (radians at fsc).
    psi: Option<f64>,
    /// Philips code read on this line, if it is one of the VBI lines looked at.
    code: Option<i64>,
}

/// The tracking state: the line grid so far and the fields found on it.
struct Tracker {
    fs: f64,
    hlen: f64,
    /// Absolute index of `lines[0]`.
    base: usize,
    /// The grid's lines; a line with no hsync (vertical interval, dropout)
    /// has no edge and is interpolated.
    lines: Vec<Fine>,
    /// Absolute index and coarse position of the last line with an hsync.
    last_h: Option<(usize, f64)>,
    rejects: usize,
    in_vsync: bool,
    /// Fields found (absolute first-line index, first field) and not yet sent.
    fields: Vec<(usize, bool)>,
    /// Frame code of the last first field sent (a second field inherits it).
    last_code: Option<(bool, Option<Vec<i64>>)>,
}

impl Tracker {
    fn new(fs: f64) -> Self {
        Self {
            fs,
            hlen: fs * 227.5 / FSC,
            base: 0,
            lines: Vec::new(),
            last_h: None,
            rejects: 0,
            in_vsync: false,
            fields: Vec::new(),
            last_code: None,
        }
    }

    fn end(&self) -> usize {
        self.base + self.lines.len()
    }

    /// Drop the grid (a jump in the capture): fields found so far that can no
    /// longer be completed are lost.
    fn reset(&mut self) {
        self.base = self.end();
        self.lines.clear();
        self.last_h = None;
        self.fields.clear();
        self.in_vsync = false;
    }

    fn pulse(&mut self, p: Pulse, fine: Fine) {
        match p.kind {
            Kind::H => {
                if let Some((_, ls)) = self.last_h {
                    let kr = ((p.start - ls) / self.hlen).round();
                    // Off the line grid by more than 0.4 us (plus slack for
                    // the line length's drift across a gap, ~0.1 us per line
                    // seen): a false pulse (dropout).
                    let off = (p.start - ls - kr * self.hlen).abs();
                    if kr < 1.0 || off > (0.4 + 0.15 * (kr - 1.0)) * self.fs / 1e6 {
                        self.rejects += 1;
                        if self.rejects >= 4 {
                            self.reset();
                        } else {
                            return;
                        }
                    } else if kr > 40.0 {
                        self.reset();
                    } else {
                        let kr = kr as usize;
                        if kr == 1 {
                            // Follow the line length (spindle speed).
                            self.hlen += (p.start - ls - self.hlen) / 8.0;
                        }
                        for _ in 1..kr {
                            self.lines.push(Fine::default());
                        }
                    }
                }
                self.rejects = 0;
                self.lines.push(fine);
                self.last_h = Some((self.end() - 1, p.start));
                self.in_vsync = false;
            }
            Kind::E | Kind::V => {
                let Some((li, ls)) = self.last_h else { return };
                if self.in_vsync {
                    return;
                }
                self.in_vsync = true;
                if p.kind != Kind::E {
                    return;
                }
                // The interval opens a full line after the last hsync on one
                // field, half a line after it on the other.
                let frac = (p.start - ls) / self.hlen;
                let first = if (0.85..1.15).contains(&frac) {
                    true
                } else if (0.35..0.65).contains(&frac) {
                    false
                } else {
                    return;
                };
                self.fields.push((li + 1, first));
            }
        }
    }

    /// Positions of the lines `a..b` (absolute indices, within the grid):
    /// fine where measured, interpolated elsewhere. `None` when a line at
    /// either end cannot be interpolated.
    fn positions(&self, a: usize, b: usize, psi0: Option<f64>) -> Option<Vec<f64>> {
        let w = 2.0 * PI * FSC / self.fs;
        let mut pos: Vec<f64> = self
            .lines
            .iter()
            .map(|l| match (l.edge, l.psi, psi0) {
                (Some(e), Some(p), Some(p0)) => {
                    // The burst's phase, with its half-cycle ambiguity
                    // resolved next to the edge. The correction is the edge's
                    // own error, a fraction of a sample: a large one is a
                    // damaged burst, and the edge is kept instead.
                    let mut d = p - p0;
                    d -= PI * (d / PI).round();
                    if (d / w).abs() < 2.5 { e - d / w } else { e }
                }
                (Some(e), _, _) => e,
                _ => f64::NAN,
            })
            .collect();
        // A line off the straight path of its neighbours by more than 3
        // samples is a damaged one (the time base moves far less than that
        // from one line to the next): interpolated instead.
        let dev = |pos: &[f64], i: usize| (pos[i] - (pos[i - 1] + pos[i + 1]) / 2.0).abs();
        let bad: Vec<usize> = (1..pos.len().saturating_sub(1))
            .filter(|&i| {
                let d = dev(&pos, i);
                d > 3.0
                    && (i < 2 || !(dev(&pos, i - 1) > d))
                    && (i + 2 >= pos.len() || !(dev(&pos, i + 1) > d))
            })
            .collect();
        for i in bad {
            pos[i] = f64::NAN;
        }
        let (a, b) = (a - self.base, b - self.base);
        let mut out = Vec::with_capacity(b - a);
        for i in a..b {
            if pos[i].is_finite() {
                out.push(pos[i]);
                continue;
            }
            let lo = (0..i).rev().find(|&j| pos[j].is_finite())?;
            let hi = (i + 1..pos.len()).find(|&j| pos[j].is_finite())?;
            let t = (i - lo) as f64 / (hi - lo) as f64;
            out.push(pos[lo] + t * (pos[hi] - pos[lo]));
        }
        Some(out)
    }

    /// Send every field whose lines are all placed.
    fn flush(&mut self, live: &Live, input: usize) {
        while self.fields.len() >= 2 {
            let (a, first) = self.fields[0];
            let next = self.fields[1].0;
            let b = next + OVERLAP_LINES;
            // A measured line past the end is needed to interpolate up to it.
            let placed = self.lines[b.saturating_sub(self.base).min(self.lines.len())..]
                .iter()
                .any(|l| l.edge.is_some());
            if a < self.base || !placed {
                if a < self.base {
                    self.fields.remove(0);
                    continue;
                }
                return;
            }
            // The field's burst phase reference: circular mean over its lines,
            // modulo half a cycle (consecutive lines alternate by pi).
            let (mut c, mut s) = (0.0f64, 0.0f64);
            for l in &self.lines[a - self.base..next - self.base] {
                if let Some(p) = l.psi {
                    c += (2.0 * p).cos();
                    s += (2.0 * p).sin();
                }
            }
            let psi0 = (c != 0.0 || s != 0.0).then(|| s.atan2(c) / 2.0);
            if let Some(lines) = self.positions(a, b, psi0) {
                let code = if first {
                    let words: Vec<i64> = self.lines[a - self.base + VBI_LINES.0..a - self.base + VBI_LINES.1]
                        .iter()
                        .filter_map(|l| l.code)
                        .collect();
                    Some(frame_code(&words)).filter(|c| !c.is_empty())
                } else {
                    self.last_code.as_ref().filter(|p| p.0).and_then(|p| p.1.clone())
                };
                self.last_code = Some((first, code.clone()));
                live.push(input, FieldRec { first, code, lines });
            }
            self.fields.remove(0);
            // Keep a few lines before the next field for interpolation.
            let keep_from = next.saturating_sub(64).max(self.base);
            self.lines.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }
}

/// Pulses in `freq` (the coarse frequency at `t0 + i * COARSE_STEP`): runs
/// below `thr`, timed by linear interpolation of the crossings.
fn find_pulses(freq: &[f32], t0: f64, thr: f32, fs: f64) -> Vec<Pulse> {
    let step = COARSE_STEP as f64;
    let us = fs / 1e6;
    let mut out = Vec::new();
    let mut start: Option<f64> = None;
    for i in 1..freq.len() {
        let (a, b) = (freq[i - 1], freq[i]);
        let t = t0 + (i - 1) as f64 * step + step * f64::from((a - thr) / (a - b));
        if a >= thr && b < thr {
            start = Some(t);
        } else if a < thr && b >= thr {
            if let Some(s) = start.take() {
                let d = (t - s) / us;
                let kind = if (3.6..6.2).contains(&d) {
                    Kind::H
                } else if (1.6..3.2).contains(&d) {
                    Kind::E
                } else if d >= 15.0 && d < 40.0 {
                    Kind::V
                } else {
                    continue;
                };
                out.push(Pulse { start: s, kind });
            }
        }
    }
    out
}

/// Median of a small slice (sorted in place).
fn median(v: &mut [f32]) -> f32 {
    v.sort_unstable_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Refine one hsync whose coarse start is `s` (index into `x`).
fn refine(x: &[f32], s: f64, fir: &Fir, fs: f64, vbi: bool) -> Fine {
    let us = fs / 1e6;
    let half = fir.half();
    let c0 = (s - 1.2 * us) as isize;
    let span = if vbi { 64.0 } else { 10.5 };
    let c1 = (s + span * us) as isize;
    if c0 < half as isize + 1 || c1 as usize + half + 8 >= x.len() {
        return Fine::default();
    }
    let (c0, c1) = (c0 as usize, c1 as usize);
    let f = fir.freq(x, c0, c1, fs);
    // Centred 9-sample boxcar for the edge and the levels.
    let sm: Vec<f32> = (0..f.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(4), (i + 5).min(f.len()));
            f[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect();
    let at = |t: f64| (t - c0 as f64) as usize;
    let sync = median(&mut sm[at(s + 1.0 * us)..at(s + 2.5 * us)].to_vec());
    let porch = median(&mut sm[at(s + 8.0 * us)..at(s + 9.0 * us)].to_vec());
    let depth = f64::from(porch - sync) / HZ_IRE;
    if !(25.0..55.0).contains(&depth) {
        return Fine::default();
    }
    let mid = (porch + sync) / 2.0;
    let (from, to) = (at(s - 1.0 * us), at(s + 1.0 * us));
    let Some(i) = (from.max(1)..to).find(|&i| sm[i - 1] >= mid && sm[i] < mid) else {
        return Fine::default();
    };
    let edge = c0 as f64 + (i - 1) as f64 + f64::from((sm[i - 1] - mid) / (sm[i - 1] - sm[i]));
    // Dropout check: the pulse and the porch stay within -90..50 IRE. Wider
    // than the decoder's -55..30, since on the RF the video is still
    // pre-emphasized (the edges overshoot); a dropout sends the carrier much
    // further off. The burst is left out (pre-emphasized as well).
    let (lo, hi) = (porch - 90.0 * HZ_IRE as f32, porch + 50.0 * HZ_IRE as f32);
    let ok = |a: f64, b: f64| sm[at(a)..at(b)].iter().all(|&v| v >= lo && v <= hi);
    if !ok(edge - 0.75 * us, edge + 4.0 * us) || !ok(edge + 8.0 * us, edge + 9.0 * us) {
        return Fine::default();
    }
    // Burst: least-squares phase at fsc over cycles 21..28 after the edge.
    let w = 2.0 * PI * FSC / fs;
    let (b0, b1) = (edge + 21.0 * fs / FSC, edge + 28.0 * fs / FSC);
    let (i0, i1) = (at(b0).max(0), at(b1).min(f.len()));
    let mean = f[i0..i1].iter().map(|&v| f64::from(v)).sum::<f64>() / (i1 - i0) as f64;
    let (mut pr, mut pi) = (0.0f64, 0.0f64);
    for (k, &v) in f[i0..i1].iter().enumerate() {
        let ph = w * ((c0 + i0 + k) as f64 - edge);
        let v = f64::from(v) - mean;
        pr += v * ph.cos();
        pi -= v * ph.sin();
    }
    let amp = 2.0 * (pr * pr + pi * pi).sqrt() / (i1 - i0) as f64 / HZ_IRE;
    let psi = (amp > 5.0).then(|| pi.atan2(pr));
    let code = if vbi { philips(&sm, c0, edge, porch, fs) } else { None };
    Fine { edge: Some(edge), psi, code }
}

/// A Philips code (24 bits, biphase, 2 us per bit) on the line starting at
/// `edge`, read as the decoder reads it: transitions through 50 IRE.
fn philips(sm: &[f32], c0: usize, edge: f64, porch: f32, fs: f64) -> Option<i64> {
    let us = fs / 1e6;
    let thr = porch + (50.0 * HZ_IRE) as f32;
    let at = |t: f64| (t - c0 as f64) as usize;
    let cross = |from: f64, len: f64| -> Option<f64> {
        let (a, b) = (at(from).max(1), at(from + len).min(sm.len()));
        (a..b)
            .find(|&i| (sm[i - 1] < thr) != (sm[i] < thr))
            .map(|i| c0 as f64 + (i - 1) as f64 + f64::from((sm[i - 1] - thr) / (sm[i - 1] - sm[i])))
    };
    let mut zc = Vec::with_capacity(24);
    let mut cur = cross(edge + 2.0 * us, 12.0 * us);
    while let Some(c) = cur {
        let i = at(c - 0.5 * us);
        zc.push((c, i < sm.len() && sm[i] < thr));
        if zc.len() > 24 {
            return None;
        }
        cur = cross(c + 1.9 * us, 0.2 * us);
    }
    if zc.len() != 24 || zc.windows(2).any(|w| !(1.85..2.15).contains(&((w[1].0 - w[0].0) / us))) {
        return None;
    }
    Some(zc.iter().fold(0i64, |v, b| v << 1 | i64::from(b.1)))
}

/// Locate every line of `input` from its RF and feed the fields to `live`.
pub fn locate(src: &StackInput, fs: f64, live: &Live, input: usize) -> Result<()> {
    let coarse = Fir::new(COARSE_TAPS, COARSE_BAND, 6.0, fs);
    let fine = Fir::new(FINE_TAPS, FINE_BAND, 7.0, fs);
    let thr = ((SYNC_HZ + BLANK_HZ) / 2.0) as f32;
    let hlen = fs * 227.5 / FSC;
    let len = src.len();
    let mut tr = Tracker::new(fs);
    let mut buf = vec![0.0f32; CHUNK + 2 * MARGIN];
    let mut own0 = 0u64;
    while own0 < len {
        if live.stopped() {
            bail!("stack: pass 0 no longer needed");
        }
        let b0 = own0 as i64 - MARGIN as i64;
        src.read_at(b0, &mut buf)?;
        // Coarse frequency on a grid aligned to absolute multiples of the
        // step, so chunk boundaries do not move it.
        let h = coarse.half();
        let g0 = ((b0 + h as i64 + 1).max(0) as u64).div_ceil(COARSE_STEP as u64) * COARSE_STEP as u64;
        let n = ((b0 + buf.len() as i64 - h as i64 - 10) as u64).saturating_sub(g0) as usize / COARSE_STEP;
        let k = (fs / (2.0 * PI)) as f32;
        let prods: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|j| {
                let c = (g0 as i64 + (j * COARSE_STEP) as i64 - b0) as usize;
                let (a, b) = (coarse.at(&buf, c), coarse.at(&buf, c + 1));
                (b.0 * a.0 + b.1 * a.1, b.1 * a.0 - b.0 * a.1)
            })
            .collect();
        let freq: Vec<f32> = (0..prods.len().saturating_sub(1))
            .map(|j| {
                let (re, im) = (prods[j].0 + prods[j + 1].0, prods[j].1 + prods[j + 1].1);
                im.atan2(re) * k
            })
            .collect();
        let t0 = g0 as f64 + COARSE_STEP as f64 / 2.0;
        let pulses = find_pulses(&freq, t0, thr, fs);
        let own1 = (own0 + CHUNK as u64).min(len);
        // VBI candidates: hsyncs 6.5 to 13.5 lines after a broad pulse.
        let mut last_v = f64::NEG_INFINITY;
        let jobs: Vec<(Pulse, bool)> = pulses
            .iter()
            .filter_map(|p| {
                if p.kind == Kind::V {
                    last_v = p.start;
                }
                let owned = p.start >= own0 as f64 && p.start < own1 as f64;
                let vbi = p.kind == Kind::H && (6.0..14.0).contains(&((p.start - last_v) / hlen));
                owned.then_some((*p, vbi))
            })
            .collect();
        let fines: Vec<Fine> = jobs
            .par_iter()
            .map(|&(p, vbi)| {
                if p.kind != Kind::H {
                    return Fine::default();
                }
                let mut f = refine(&buf, p.start - b0 as f64, &fine, fs, vbi);
                f.edge = f.edge.map(|e| e + b0 as f64);
                f
            })
            .collect();
        for (&(p, _), &f) in jobs.iter().zip(&fines) {
            tr.pulse(p, f);
        }
        tr.flush(live, input);
        own0 = own1;
    }
    Ok(())
}
