//! Time-base correction at read time (`--tbc read`), and multi-source stacking
//! built on it.
//!
//! Pass 0 locates every input (one per input, running alongside the stacked
//! decode, straight from the RF: `rfsync.rs`) and keeps, per field, the VBI frame code, the parity and
//! the absolute position of each line. Fields of the same frame are paired
//! across inputs, and the primary's (first input's) lines define a timeline of
//! nominal-length lines, laid down as far as every input is located ([`Live`]);
//! pass 0 thus stops where the stacked decode stops. [`VirtualSource`] serves that timeline as
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
//! `--device opencl`, on a GPU (`opencl.rs`).

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{bail, Context as _, Result};
use rayon::prelude::*;
use rustfft::num_complex::{Complex32, Complex64};
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
/// PAL: the video FM's lower chroma sideband reaches down to ~2.7 MHz (the
/// pilot's to 3.0), while EFM stays below 1.75 MHz and the audio carriers at
/// 0.68 / 1.07 MHz.
pub(crate) const PAL_SPLIT_HZ: (f64, f64) = (1.9e6, 2.4e6);
/// Video-band filter: taps and Kaiser beta. Only the crossover's shape depends
/// on it (the low band is the full signal minus this band), and 128 taps put
/// the leakage across the crossover near -37 dB.
pub(crate) const HI_TAPS: usize = 128;
const HI_BETA: f64 = 8.0;
/// Audio crosstalk cancellation: the analog audio carriers (Hz), NTSC and PAL.
const AUDIO_CARRIERS_HZ: [f64; 2] = [2.301136e6, 2.8125e6];
const PAL_AUDIO_CARRIERS_HZ: [f64; 2] = [0.68359375e6, 1.06640625e6];
/// Length of each of the three cascaded boxcars that take a carrier out of
/// the RF (seconds): ~150 kHz around it, wide enough for its deviation.
const XTALK_BOX_S: f64 = 320.0 / 75.0e6;
/// The correction fades in and out over this much at each chunk's edges
/// (seconds), where the boxcars run out of signal: so it is continuous from
/// one chunk to the next.
const XTALK_TAPER_S: f64 = 1024.0 / 75.0e6;
/// The correction is kept above this raised-cosine edge (Hz, NTSC; PAL: its
/// video band split): the video is demodulated from above it, the audio
/// carriers and the EFM sit below.
const XTALK_KEEP_HZ: (f64, f64) = (3.0e6, 3.6e6);
/// The carriers, once at 0 Hz, are kept every this many samples.
const XTALK_DEC: usize = 16;
/// The crosstalk gains are fitted on every this many samples of a chunk.
const XTALK_FIT_STRIDE: usize = 20;
/// FFT length of the correction's high-pass, and the context it takes on
/// each side of each block (seconds).
const XTALK_FFT_LEN: usize = 1 << 15;
const XTALK_OV_S: f64 = 2048.0 / 75.0e6;
/// Kept samples per span of running sums of the full-rate boxcar.
const XTALK_BOX_SPAN: usize = 1024;

/// Chroma leak cancellation (NTSC): the band the leak is fitted in (rising
/// and falling raised-cosine edges, Hz: the chroma, 3.58 MHz +- 0.6), the
/// lags of its fit (`LEAK_TAPS` lags `LEAK_STEP` samples apart, centred),
/// and the fit's sample stride.
const LEAK_BAND_HZ: ((f64, f64), (f64, f64)) = ((2.9e6, 3.1e6), (4.1e6, 4.3e6));
const LEAK_TAPS: usize = 13;
const LEAK_STEP: usize = 2;
const LEAK_FIT_STRIDE: usize = 7;
/// A chunk is corrected only where the leak explains at least this share of
/// the RF's chroma band (a disc or a stretch with no leak is left untouched).
/// FFT length of the band filter (overlap-save blocks: enough of them to
/// share out over the cores).
const LEAK_FFT_LEN: usize = 1 << 15;
/// Fit samples per parallel span of the normal equations.
const LEAK_FIT_SPAN: usize = 1 << 12;
const LEAK_MIN_SHARE: f64 = 0.05;

/// A player's EFM decoder crystal (384 x 44.1 kHz) and the divisor its lines
/// in the RF are multiples of (its 192nd, 88.2 kHz).
const EFM_XTAL_HZ: f64 = 16.9344e6;
const XTAL_DIV: f64 = 192.0;
/// The crystal is looked for within this fraction of `EFM_XTAL_HZ`.
const XTAL_SEARCH: f64 = 2e-3;
/// Detection spectrum: FFT length, stretches of the capture (spread over it;
/// a steady line is in all), and the span around a bin its floor is the
/// median of (Hz).
const XTAL_FFT: usize = 1 << 21;
const XTAL_SEGMENTS: usize = 4;
const XTAL_FLOOR_HZ: f64 = 20e3;
/// Power over the floor for the crystal itself to count as found, and for a
/// multiple to be subtracted (10 dB both).
const XTAL_FIND: f64 = 10.0;
const XTAL_KEEP: f64 = 10.0;
/// Lines above this are left: nothing the decoder reads reaches that high
/// (its video band ends at 13.8 MHz); nor below `XTAL_LOW_HZ`.
const XTAL_TOP_HZ: f64 = 14.5e6;
const XTAL_LOW_HZ: f64 = 0.3e6;
/// Where the disc's own signal makes steady lines, left out of the other
/// lines (Hz, NTSC then PAL): the analog audio carriers (a pure tone in
/// silence), the chroma leak's band, and the video carrier's swing (its
/// blanking and sync levels repeat line after line).
const LINE_EXCLUDE_HZ: [(f64, f64); 4] = [(2.1e6, 2.5e6), (2.6e6, 3.0e6), (3.2e6, 4.0e6), (7.3e6, 9.6e6)];
const PAL_LINE_EXCLUDE_HZ: [(f64, f64); 4] = [(0.48e6, 0.88e6), (0.86e6, 1.27e6), (4.0e6, 4.9e6), (6.4e6, 8.3e6)];
/// Samples a line's amplitude and phase are measured over, and the
/// super-blocks the crystal's drift is followed on (aligned on the capture,
/// so every read of a sample cleans it alike); super-blocks kept measured.
const XTAL_BLOCK: usize = 1 << 16;
const XTAL_SUPER: usize = 1 << 22;
const XTAL_CACHE: usize = 16;
/// Samples between the exact seeds of a line's phasor in `clean`.
const XTAL_SEED: usize = 1024;

/// Band used to estimate each input's noise floor (no signal there), kept
/// below `NOISE_TOP` of the Nyquist frequency (the capture's anti-alias
/// filter); where less than `NOISE_MIN_BW` of it remains (a capture at 30
/// MHz), it reaches down to `NOISE_LOW_HZ`, where the RF has already fallen
/// to the floor.
const NOISE_BAND_HZ: (f64, f64) = (16.0e6, 19.5e6);
const NOISE_TOP: f64 = 0.95;
const NOISE_MIN_BW: f64 = 1.0e6;
const NOISE_LOW_HZ: f64 = 13.5e6;

/// The noise-floor band at a sample rate (see [`NOISE_BAND_HZ`]).
fn noise_band(fs: f64) -> Result<(f64, f64)> {
    let top = NOISE_TOP * fs / 2.0;
    let (lo, hi) = (NOISE_BAND_HZ.0, NOISE_BAND_HZ.1.min(top));
    if hi - lo >= NOISE_MIN_BW {
        return Ok((lo, hi));
    }
    if top - NOISE_LOW_HZ < NOISE_MIN_BW / 4.0 {
        bail!("input rate {fs} Hz too low: no noise-floor band left above the RF");
    }
    Ok((NOISE_LOW_HZ, top))
}
/// A paired line whose length differs from the primary's by more than this
/// ratio is treated as a skip in that input (not covered).
const MAX_LINE_RATIO_DEV: f64 = 0.02;
/// Amplitude weight floor, as a fraction of the input's video-band noise RMS:
/// where every input has lost its carrier the sum still has weights.
const AMP_FLOOR: f64 = 0.05;
/// Weight of the zero-phase prior of the phase lock, against the window's
/// `samples * snr` (an ordinary lock is ~10^3 or more): pass 0 aligns the
/// carriers within a few hundredths of a radian, so where the measured
/// rotation is noise (a dropout in either input) the lock holds still.
pub(crate) const LOCK_PRIOR: f64 = 10.0;
/// Alignment guard: the coherence with the primary over the lock window, as
/// a fraction of what the two inputs' carrier fractions allow, below which an
/// input is left out (fading in over `KAPPA_SPAN`). It applies once that
/// expected coherence squared exceeds `Q_LO` (fully from `Q_LO + Q_SPAN`):
/// unless both carriers are clean over the whole window (a dropout in either
/// lowers the coherence as a misalignment would) the coherence says nothing
/// about alignment.
pub(crate) const KAPPA_LO: f64 = 0.5;
pub(crate) const KAPPA_SPAN: f64 = 0.3;
pub(crate) const Q_LO: f64 = 0.7;
pub(crate) const Q_SPAN: f64 = 0.15;
/// The lock follows its measured rotation above this expected coherence
/// squared, and holds still (`LOCK_PRIOR`) below.
pub(crate) const Q_ROT: f64 = 0.5;
/// The guard applies only where the primary keeps at the sample (carrier
/// window) at least this fraction of its carrier over the lock window, fully
/// from `H_LO + H_SPAN`.
pub(crate) const H_LO: f64 = 0.5;
pub(crate) const H_SPAN: f64 = 0.3;
/// Window of the carrier measure behind the weights, centred on the sample:
/// short, so an input weighs little as soon as the sample itself is in a
/// dropout (over the lock window, a part of healthy carrier would dominate).
pub(crate) const CARRIER_WIN: usize = 64;
/// Ceiling of a carrier fraction (the lock's `snr` stays finite).
pub(crate) const Q_MAX: f64 = 0.999;
/// What the carrier measure reads on noise alone, over the lock window and
/// over the carrier window (about its 99th and 90th percentiles on the video
/// band's noise; a carrier reads ~0.99): taken out, so a dropout reads no
/// carrier.
pub(crate) const QN_LOCK: f64 = 0.4;
pub(crate) const QN_CARRIER: f64 = 0.5;

/// Band width the carrier lag is set from: the video band's noise decorrelates
/// after about `fs / CARRIER_BW` samples.
const CARRIER_BW: f64 = 15.0e6;

/// The carrier lag at a sample rate (see [`carrier`]).
pub(crate) fn carrier_lag(fs: f64) -> usize {
    ((fs / CARRIER_BW).round() as usize).max(1)
}

/// Carrier power and carrier fraction of an input's video band over the lock
/// window, from `rl = sum(z[t] conj(z[t - lag]))`, the energy `e` of those
/// `z[t]` and their count `n`. Over the lag the FM carrier keeps its
/// coherence (its phase advance barely changes within the window) while the
/// band's noise has lost it, so `|rl|` measures the carrier alone, blind to
/// the noise level: a dropout's noise, however loud, does not pass for
/// carrier. What noise alone reads over the window (`qn` of the energy) is
/// taken out of both. The OpenCL kernel's arithmetic, in its order.
#[inline]
pub(crate) fn carrier(rl: Complex64, e: f64, n: f64, qn: f64) -> (f64, f64) {
    let a = (rl.re * rl.re + rl.im * rl.im).sqrt();
    let s = (a - qn * e).max(0.0) / ((1.0 - qn) * n);
    (s, if e > 0.0 { ((a / e - qn).max(0.0) / (1.0 - qn)).min(Q_MAX) } else { 0.0 })
}

/// An input's weight at one sample from its carrier power `s`: the carrier
/// amplitude (floored at `afl`) times `1/sigma^2`, the maximal ratio
/// combining weight. The OpenCL kernel's arithmetic, in its order.
#[inline]
pub(crate) fn amp_weight(s: f64, afl: f64, w: f64) -> f64 {
    s.sqrt().max(afl) * w
}

/// Phase lock of an input onto the primary over the lock window, from the
/// correlation `r`, both energies over the `nj` samples they share and both
/// carrier fractions: the rotation, pulled toward zero as the carriers
/// weaken, and the alignment guard's factor on the weight. The OpenCL
/// kernel's arithmetic, in its order.
#[inline]
pub(crate) fn lock(r: Complex64, ea: f64, eb: f64, nj: f64, q0: f64, qi: f64, h0: f64) -> (Complex64, f64) {
    // Squared coherence the two carrier fractions allow.
    let q = q0 * qi;
    let a = (r.re * r.re + r.im * r.im).sqrt();
    let mut g = 1.0;
    if q > Q_LO {
        let kappa = a / (ea * eb).sqrt() / q.sqrt();
        let gk = ((kappa - KAPPA_LO) / KAPPA_SPAN).clamp(0.0, 1.0);
        // Only with the primary's carrier at the sample itself (`h0`): at
        // the edge of a dropout of the primary the window still reads its
        // carrier, but the coherence falls, and the input left out would be
        // the healthy one.
        let hp = ((h0 - H_LO) / H_SPAN).clamp(0.0, 1.0);
        g = 1.0 - (1.0 - gk) * ((q - Q_LO) / Q_SPAN).min(1.0) * hp;
    }
    if a <= 0.0 {
        return (Complex64::new(1.0, 0.0), g);
    }
    // The measured rotation counts only where both carriers are clearly
    // healthy (`Q_ROT`): a window part noise gives a rotation part noise.
    let qr = ((q - Q_ROT) / (1.0 - Q_ROT)).max(0.0);
    let wm = nj * qr / (1.0 - qr);
    let vx = r.re / a * wm + LOCK_PRIOR;
    let vy = r.im / a * wm;
    let b = (vx * vx + vy * vy).sqrt();
    (Complex64::new(vx / b, vy / b), g)
}

/// Per-field record kept from pass 0.
#[derive(Clone)]
pub struct FieldRec {
    pub first: bool,
    /// Frame code words (VBI) of the frame this field belongs to.
    pub code: Option<Vec<i64>>,
    /// Absolute position of each line (input samples).
    pub lines: Vec<f64>,
}

/// Text dump of pass-0 records (`LD_DUMP_PASS0`), exact to the bit.
fn dump_records(fields: &[FieldRec], path: &std::path::Path) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    for f in fields {
        let _ = write!(s, "{} {:?}", f.first, f.code);
        for l in &f.lines {
            let _ = write!(s, " {:016x}", l.to_bits());
        }
        s.push('\n');
    }
    std::fs::write(path, s)
}

/// Frame code and parity: what pairs a field across inputs.
type Key = (Vec<i64>, bool);

/// A secondary input's field is paired only once this many later fields of
/// that input are known, so a frame code it repeats within them (a skip back
/// in that capture) is seen as ambiguous and left out, as a whole-file pass
/// would.
const PAIR_LOOKAHEAD: usize = 16;

/// The timeline, built while pass 0 runs: one pass-0 decode per input feeds
/// its field records in, and the anchors are laid down as far as every input
/// has been located. The virtual source waits on it, so pass 0 runs alongside
/// the decode of the stacked stream and only as far as that decode goes.
pub struct Live {
    state: Mutex<LiveState>,
    cv: Condvar,
    stop: AtomicBool,
    paths: Vec<String>,
    linelen: f64,
    /// Primary capture length (samples): the stream's tail ends there.
    prim_len: f64,
}

struct LiveState {
    fields: Vec<Vec<FieldRec>>,
    done: Vec<bool>,
    error: Option<String>,
    /// Primary field index of each key (first occurrence).
    prim_map: HashMap<Key, usize>,
    /// Paired fields of each secondary (None: code repeated, ambiguous).
    sec_map: Vec<HashMap<Key, Option<usize>>>,
    /// Secondary fields examined for pairing so far.
    confirmed: Vec<usize>,
    /// 1 + the furthest primary field each secondary has been paired with.
    progress: Vec<usize>,
    paired: Vec<usize>,
    next_primary: usize,
    last: f64,
    tl: Timeline,
    finalized: bool,
}

impl Live {
    pub fn new(paths: Vec<String>, prim_len: u64, linelen: f64) -> Arc<Self> {
        let n = paths.len();
        Arc::new(Self {
            state: Mutex::new(LiveState {
                fields: vec![Vec::new(); n],
                done: vec![false; n],
                error: None,
                prim_map: HashMap::new(),
                sec_map: vec![HashMap::new(); n],
                confirmed: vec![0; n],
                progress: vec![0; n],
                paired: vec![0; n],
                next_primary: 0,
                last: f64::NEG_INFINITY,
                tl: Timeline { t: Vec::new(), pos: vec![Vec::new(); n] },
                finalized: false,
            }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            paths,
            linelen,
            prim_len: prim_len as f64,
        })
    }

    pub(crate) fn push(&self, input: usize, rec: FieldRec) {
        let mut guard = self.state.lock().unwrap();
        let st = &mut *guard;
        if input == 0 {
            if let Some(c) = &rec.code {
                let key = (c.clone(), rec.first);
                let idx = *st.prim_map.entry(key.clone()).or_insert(st.fields[0].len());
                for s in 1..self.paths.len() {
                    if st.sec_map[s].contains_key(&key) {
                        st.progress[s] = st.progress[s].max(idx + 1);
                    }
                }
            }
        }
        st.fields[input].push(rec);
        self.advance(st);
        self.cv.notify_all();
    }

    /// The stacked decode is over: pass 0 can stop.
    pub(crate) fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Pass 0 of `input` ended (Err: it failed, unless the stack stopped it).
    pub fn finish(&self, input: usize, result: Result<()>) {
        let mut st = self.state.lock().unwrap();
        st.done[input] = true;
        match result {
            Err(e) if !self.stop.load(Ordering::Relaxed) => {
                st.error.get_or_insert_with(|| format!("pass 0 of {}: {e:#}", self.paths[input]));
            }
            _ => tracing::info!("read-time TBC pass 0: {} fields in {}", st.fields[input].len(), self.paths[input]),
        }
        if let Some(dir) = std::env::var_os("LD_DUMP_PASS0") {
            let path = std::path::Path::new(&dir).join(format!("pass0_{input}.txt"));
            if let Err(e) = dump_records(&st.fields[input], &path) {
                tracing::warn!("LD_DUMP_PASS0: {e}");
            }
        }
        self.advance(&mut st);
        self.cv.notify_all();
    }

    /// Lay down every anchor all inputs are located for.
    fn advance(&self, st: &mut LiveState) {
        let n = self.paths.len();
        let linelen = self.linelen;
        for s in 1..n {
            while st.confirmed[s] < st.fields[s].len()
                && (st.done[s] || st.fields[s].len() - st.confirmed[s] > PAIR_LOOKAHEAD)
            {
                let k = st.confirmed[s];
                st.confirmed[s] += 1;
                let f = &st.fields[s][k];
                let Some(c) = &f.code else { continue };
                let key = (c.clone(), f.first);
                let end = (k + 1 + PAIR_LOOKAHEAD).min(st.fields[s].len());
                let repeated = st.fields[s][k + 1..end].iter().any(|g| g.first == f.first && g.code.as_ref() == Some(c));
                st.sec_map[s]
                    .entry(key.clone())
                    .and_modify(|e| *e = None)
                    .or_insert(if repeated { None } else { Some(k) });
                if let Some(&q) = st.prim_map.get(&key) {
                    st.progress[s] = st.progress[s].max(q + 1);
                }
            }
        }

        while let Some(f) = st.fields[0].get(st.next_primary) {
            let pf = st.next_primary;
            let key = f.code.as_ref().map(|c| (c.clone(), f.first));
            // Each secondary has this frame paired, or is past it (it has
            // been paired with a later primary field), or is finished.
            let ready = (1..n).all(|s| {
                st.done[s]
                    || key.as_ref().is_none_or(|k| st.sec_map[s].contains_key(k))
                    || st.progress[s] > pf + 1
            });
            if !ready {
                break;
            }
            let matched: Vec<Option<usize>> = (0..n)
                .map(|s| if s == 0 { None } else { key.as_ref().and_then(|k| st.sec_map[s].get(k).copied().flatten()) })
                .collect();
            for s in 1..n {
                st.paired[s] += usize::from(matched[s].is_some());
            }
            let f = &st.fields[0][pf];
            for (l, &a) in f.lines.iter().enumerate() {
                // Fields overlap by a few lines; keep each line once, in order.
                if a <= st.last + 0.25 * linelen {
                    continue;
                }
                let tk = match st.tl.t.last() {
                    // The timeline starts at the primary's own first line;
                    // before it, the primary alone at its own rate (the
                    // decoder needs that lead-in to find the first field).
                    None => {
                        if a >= 1.0 {
                            st.tl.t.push(0.0);
                            st.tl.pos[0].push(0.0);
                            for p in st.tl.pos.iter_mut().skip(1) {
                                p.push(f64::NAN);
                            }
                        }
                        a
                    }
                    Some(&tp) => {
                        let lines = ((2.0 * (a - st.last) / linelen).round() / 2.0).max(0.5);
                        tp + lines * linelen
                    }
                };
                st.tl.t.push(tk);
                st.tl.pos[0].push(a);
                for s in 1..n {
                    let p = matched[s].and_then(|g| st.fields[s][g].lines.get(l).copied()).unwrap_or(f64::NAN);
                    st.tl.pos[s].push(p);
                }
                st.last = a;
            }
            st.next_primary += 1;
        }

        if !st.finalized && st.done[0] && st.next_primary == st.fields[0].len() {
            st.finalized = true;
            if st.tl.t.len() < 2 {
                st.error.get_or_insert_with(|| "pass 0 found no usable lines in the primary input".to_string());
                return;
            }
            // After the last line, the primary alone again (the tail).
            let (t_last, a_last) = (*st.tl.t.last().unwrap(), *st.tl.pos[0].last().unwrap());
            let tail = self.prim_len - a_last;
            if tail >= 1.0 {
                st.tl.t.push(t_last + tail);
                st.tl.pos[0].push(a_last + tail);
                for p in st.tl.pos.iter_mut().skip(1) {
                    p.push(f64::NAN);
                }
            }
            for s in 1..n {
                tracing::info!(
                    "stack: {} paired {} of the primary's {} fields",
                    self.paths[s],
                    st.paired[s],
                    st.fields[0].len()
                );
            }
        }
    }

    /// Block until the timeline reaches ideal time `end`, or is complete;
    /// then the stream length if it is known.
    fn wait_cover(&self, end: f64) -> Result<Option<u64>> {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(e) = &st.error {
                bail!("{e}");
            }
            if st.finalized {
                return Ok(Some(*st.tl.t.last().unwrap() as u64));
            }
            if st.tl.t.last().is_some_and(|&t| t >= end) {
                return Ok(None);
            }
            st = self.cv.wait(st).unwrap();
        }
    }

    /// The anchors around `[s0, s0 + len)`, copied out.
    fn slice(&self, s0: f64, len: usize) -> Result<Timeline> {
        self.wait_cover(s0 + len as f64)?;
        let st = self.state.lock().unwrap();
        let tl = &st.tl;
        let a = tl.segment(s0);
        let b = (tl.segment(s0 + len as f64 - 1.0) + 1).min(tl.t.len() - 1);
        Ok(Timeline { t: tl.t[a..=b].to_vec(), pos: tl.pos.iter().map(|p| p[a..=b].to_vec()).collect() })
    }

    /// Absolute position of each input's first located line (waits for it).
    fn first_lines(&self) -> Result<Vec<Option<f64>>> {
        let mut st = self.state.lock().unwrap();
        loop {
            if let Some(e) = &st.error {
                bail!("{e}");
            }
            if (0..self.paths.len()).all(|i| st.done[i] || !st.fields[i].is_empty()) {
                return Ok(st.fields.iter().map(|f| f.first().and_then(|f| f.lines.first().copied())).collect());
            }
            st = self.cv.wait(st).unwrap();
        }
    }

    /// Stop the pass-0 decodes (the stacked decode is over).
    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.cv.notify_all();
    }
}

/// The VBI words that identify a frame: CAV picture numbers and CLV time codes
/// (top nibble 0xF or 0x8), sorted. Two captures of the same frame give the
/// same set; a misread gives a different one, so the field is just left
/// unpaired.
pub(crate) fn frame_code(words: &[i64]) -> Vec<i64> {
    let mut v: Vec<i64> = words
        .iter()
        .copied()
        .filter(|&w| w > 0 && matches!((w >> 20) & 0xF, 0x8 | 0xF))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// One input: a random-access reader.
pub struct StackInput {
    pub path: String,
    reader: RawFile,
    /// The player's EFM decoder crystal lines, subtracted as it is read.
    xtal: Option<CrystalLines>,
}

impl StackInput {
    pub fn new(path: &str, format: SampleFormat) -> Result<Self> {
        Ok(Self { path: path.to_string(), reader: RawFile::open(path, format)?, xtal: None })
    }

    /// Length in samples.
    pub fn len(&self) -> u64 {
        self.reader.len
    }

    /// Samples `[start, start + out.len())`; zeros outside the file.
    pub(crate) fn read_at(&self, start: i64, out: &mut [f32]) -> Result<()> {
        self.reader.read_at(start, out)
    }
}

/// Fixed-width raw capture read by sample index (positioned reads, so the
/// parallel tasks share one handle).
struct RawFile {
    file: File,
    bytes: usize,
    widen: fn(&[u8], &mut [f32]),
    /// 16-bit code of each sample and the offset back to its value (`value =
    /// code + offset`, exact), for the formats whose values fit in 65536
    /// steps: the GPU path moves these instead of f32.
    code: Option<(fn(&[u8], &mut [u16]), f32)>,
    len: u64,
}

fn code_s16(b: &[u8], out: &mut [u16]) {
    for (o, w) in out.iter_mut().zip(b.chunks_exact(2)) {
        *o = u16::from_le_bytes([w[0], w[1]]) ^ 0x8000;
    }
}

fn code_u16(b: &[u8], out: &mut [u16]) {
    for (o, w) in out.iter_mut().zip(b.chunks_exact(2)) {
        *o = u16::from_le_bytes([w[0], w[1]]);
    }
}

fn code_u8(b: &[u8], out: &mut [u16]) {
    for (o, &v) in out.iter_mut().zip(b) {
        *o = u16::from(v);
    }
}

fn code_s8(b: &[u8], out: &mut [u16]) {
    for (o, &v) in out.iter_mut().zip(b) {
        *o = (i32::from(v as i8) * 256 + 32768) as u16;
    }
}

impl RawFile {
    fn open(path: &str, format: SampleFormat) -> Result<Self> {
        if path == "-" {
            bail!("--tbc read reads its input at random positions, which stdin cannot do");
        }
        let Some((bytes, widen)) = crate::reader::raw_layout(format) else {
            bail!("--tbc read needs a fixed-width raw input (s16, r16, r8, s8, rf); {path} is {format:?}");
        };
        let file = File::open(path).with_context(|| format!("opening {path}"))?;
        let len = file.metadata()?.len() / bytes as u64;
        let code = match format {
            SampleFormat::S16Le => Some((code_s16 as fn(&[u8], &mut [u16]), -32768.0)),
            SampleFormat::U16Le => Some((code_u16 as fn(&[u8], &mut [u16]), 0.0)),
            SampleFormat::U8Le => Some((code_u8 as fn(&[u8], &mut [u16]), 0.0)),
            SampleFormat::S8Le => Some((code_s8 as fn(&[u8], &mut [u16]), -32768.0)),
            _ => None,
        };
        Ok(Self { file, bytes, widen, code, len })
    }

    /// The file's bytes for samples `[lo, hi)` (clipped to the file).
    fn bytes_at(&self, lo: u64, hi: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; (hi - lo) as usize * self.bytes];
        let mut filled = 0usize;
        while filled < buf.len() {
            let got = read_at(&self.file, &mut buf[filled..], lo * self.bytes as u64 + filled as u64)?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        buf.truncate(filled / self.bytes * self.bytes);
        Ok(buf)
    }

    /// 16-bit codes of samples `[start, start + out.len())` (see `code`);
    /// the code of 0 outside the file.
    fn read_codes_at(&self, start: i64, out: &mut [u16]) -> Result<f32> {
        let (code, off) = self.code.expect("a format with 16-bit codes");
        out.fill((-off) as u16);
        let lo = start.max(0) as u64;
        let hi = ((start + out.len() as i64).max(0) as u64).min(self.len);
        if hi > lo {
            let buf = self.bytes_at(lo, hi)?;
            let at = (lo as i64 - start) as usize;
            code(&buf, &mut out[at..at + buf.len() / self.bytes]);
        }
        Ok(off)
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

pub(crate) fn bessel_i0(x: f64) -> f64 {
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
fn video_band_response(tau: f64, fs: f64, split: (f64, f64), top: (f64, f64)) -> Complex64 {
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
    let sum = edge(split.0, split.1, -1.0) + int_exp(c, split.1, top.0) + edge(top.0, top.1, 1.0);
    sum * (2.0 / fs)
}

/// The video-band filter at the sinc's fractional phases: `SINC_PHASES + 1`
/// rows of `HI_TAPS` taps, tap `d` weighting input sample
/// `k0 - (HI_TAPS/2 - 1) + d` for a position `k0 + p/SINC_PHASES`, Kaiser
/// windowed.
fn video_band_lut(fs: f64, split: (f64, f64)) -> Vec<Complex64> {
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
            lut.push(video_band_response(tau, fs, split, top) * (bessel_i0(HI_BETA * r.sqrt()) / i0b));
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
    live: Arc<Live>,
    pub(crate) weight: Vec<f64>,
    /// Per input: the amplitude weight floor (see [`amp_weight`]).
    pub(crate) afloor: Vec<f64>,
    /// Lag of the carrier measure (see [`carrier`]).
    pub(crate) lag: usize,
    /// Video-band filter table (see [`video_band_lut`]).
    pub(crate) hi_lut: Vec<Complex64>,
    /// The CPU path's single-precision copies: sinc, video band re and im.
    fir: [Vec<f32>; 3],
    /// Exact arithmetic (`--preset slow`): the FIRs and the phase lock in
    /// double precision, in one fixed order on the CPU and the GPU, so both
    /// give the same output.
    pub(crate) exact: bool,
    /// `--stack-weight adaptive`: every input weighted at every sample by its
    /// carrier amplitude, with the alignment guard (`compute_task`). Off
    /// (`fixed`): the fixed `1/sigma^2` weights and the plain phase lock
    /// (`combine_fixed`).
    pub(crate) carrier: bool,
    /// Audio crosstalk cancellation: the sample rate and the audio carriers.
    pub(crate) xtalk: Option<(f64, [f64; 2], (f64, f64))>,
    /// Chroma leak cancellation: the sample rate and the band.
    leak: Option<(f64, ((f64, f64), (f64, f64)))>,
}

impl Inner {
    /// Whether the combine also returns the combined video band (the audio
    /// crosstalk and chroma leak cancellations need it).
    pub(crate) fn wants_band(&self) -> bool {
        self.xtalk.is_some() || self.leak.is_some()
    }
}

/// `sum(h[i] * x[i])` over 8 single-precision lanes (as the OpenCL kernel,
/// in single precision): the lanes vectorize, a serial f64 chain does not.
#[inline]
fn dot8(h: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (hc, xc) in h.chunks_exact(8).zip(x.chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += hc[l] * xc[l];
        }
    }
    acc.iter().sum()
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
    /// The span's samples: as f32, or (`codes` plans) as 16-bit codes with
    /// `value = code + off`.
    pub(crate) raw: Vec<f32>,
    pub(crate) codes: Vec<u16>,
    pub(crate) off: f32,
    /// Input sample of `raw[0]` / `codes[0]`.
    pub(crate) start: i64,
    /// Per segment: absolute position at `seg_t0`, NaN where the input does
    /// not cover it, and input samples per virtual sample. Absolute, so a
    /// position rounds the same whatever span or chunk it is computed in.
    pub(crate) a0: Vec<f64>,
    pub(crate) slope: Vec<f64>,
}

impl InputSpan {
    /// Absolute input position of plan sample `j`.
    #[inline]
    fn position(&self, plan: &Plan, j: usize) -> f64 {
        let k = plan.seg[j] as usize;
        self.a0[k] + (plan.s0 + j as f64 - plan.seg_t0[k]) * self.slope[k]
    }
}

impl Inner {
    /// Whether every input has 16-bit codes (see `RawFile::code`).
    pub(crate) fn has_codes(&self) -> bool {
        // The cleaned samples are no longer codes.
        self.inputs.iter().all(|s| s.reader.code.is_some() && s.xtal.is_none())
    }

    fn plan(&self, s0: f64, len: usize, codes: bool) -> Result<Plan> {
        let tl = &self.live.slice(s0, len)?;
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
                let (mut raw, mut codes16, mut off) = (Vec::new(), Vec::new(), 0.0);
                if codes {
                    codes16 = vec![0u16; (end - start) as usize];
                    off = self.inputs[i].reader.read_codes_at(start, &mut codes16)?;
                } else {
                    raw = vec![0f32; (end - start) as usize];
                    self.inputs[i].reader.read_at(start, &mut raw)?;
                    if let Some(x) = &self.inputs[i].xtal {
                        x.clean(&self.inputs[i].reader, start, &mut raw)?;
                    }
                }
                Ok(Some(InputSpan { raw, codes: codes16, off, start, a0, slope }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Plan { s0, len, seg, seg_t0, inputs })
    }
}

/// `--rf-out`: the stream written as s16 beside the decode, each chunk at its
/// place as the producer computes it (a seek leaves the skipped span zero).
pub struct RfOut {
    file: File,
    /// s16 value = stream value * scale + shift, the inverse of how the
    /// primary's format was widened (the decoder ignores the DC offset).
    scale: f32,
    shift: f32,
}

impl RfOut {
    pub fn create(path: &str, primary: SampleFormat) -> Result<Self> {
        let (scale, shift) = match primary {
            SampleFormat::U16Le => (1.0, -32768.0),
            SampleFormat::U8Le => (256.0, -32768.0),
            _ => (1.0, 0.0),
        };
        let file = File::create(path).with_context(|| format!("creating {path}"))?;
        tracing::info!("writing the RF the decoder reads to {path} (s16)");
        Ok(Self { file, scale, shift })
    }

    fn write(&self, s0: u64, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| ((v * self.scale + self.shift).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())
            .collect();
        let mut done = 0;
        while done < bytes.len() {
            done += write_at(&self.file, &bytes[done..], s0 * 2 + done as u64).context("writing --rf-out")?;
        }
        Ok(())
    }
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(file, buf, offset)
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(file, buf, offset)
}

/// Virtual samples per read-ahead chunk.
const CHUNK: usize = 1 << 21;
/// Chunks computed ahead of the decoder's read position.
const AHEAD: u64 = 3;
/// Chunks kept behind it, for the decoder's short rewinds.
const BEHIND: u64 = 1;

/// What computes the stream: the inputs and timeline, the pool, the GPU.
struct Engine {
    inner: Arc<Inner>,
    /// Its own pool on every core: the global rayon pool is the decoder's
    /// small tail pool (a quarter of the demod budget).
    pool: rayon::ThreadPool,
    gpu: Option<crate::opencl::StackGpu>,
    rf_out: Option<RfOut>,
}

impl Engine {
    /// The GPU path's plan for `n` samples at `s0` (with the lock margins).
    fn gpu_plan(&self, s0: u64, n: usize) -> Result<Plan> {
        let m = LOCK_MARGIN;
        self.inner.plan(s0 as f64 - m as f64, n + 2 * m, self.inner.has_codes())
    }

    /// Samples `[s0, s0 + n)` of the stream; `plan` is the GPU plan for
    /// exactly that span when it was prepared ahead.
    fn compute(&self, s0: u64, n: usize, plan: Option<Plan>) -> Result<Vec<f32>> {
        let (mut out, v) = self.combine(s0, n, plan)?;
        self.cancel(&mut out, &v);
        Ok(out)
    }

    /// The combined stream over `[s0, s0 + n)` before the cancellations, and
    /// its combined video band when they need it.
    fn combine(&self, s0: u64, n: usize, plan: Option<Plan>) -> Result<(Vec<f32>, Vec<Complex64>)> {
        let mut out = vec![0f32; n];
        let inner = &self.inner;
        // The combined video band, for the crosstalk cancellation.
        let mut v = vec![Complex64::new(0.0, 0.0); if inner.wants_band() { n } else { 0 }];
        self.pool.install(|| match &self.gpu {
            Some(gpu) => {
                let t0 = std::time::Instant::now();
                let plan = match plan {
                    Some(p) => p,
                    None => self.gpu_plan(s0, n)?,
                };
                let t1 = std::time::Instant::now();
                gpu.run(&plan, LOCK_MARGIN, &mut out, (!v.is_empty()).then_some(&mut v[..]))?;
                if std::env::var_os("LD_TIMING").is_some() {
                    ld_decode::teeprintln!("STACKGPU plan_us={} gpu_us={}", (t1 - t0).as_micros(), t1.elapsed().as_micros());
                }
                Ok(())
            }
            None if v.is_empty() => out
                .par_chunks_mut(TASK)
                .enumerate()
                .try_for_each(|(c, chunk)| compute_task(inner, s0 + (c * TASK) as u64, chunk, None)),
            None => out
                .par_chunks_mut(TASK)
                .zip(v.par_chunks_mut(TASK))
                .enumerate()
                .try_for_each(|(c, (chunk, vc))| compute_task(inner, s0 + (c * TASK) as u64, chunk, Some(vc))),
        })?;
        Ok((out, v))
    }

    /// The audio crosstalk and chroma leak cancellations, on `out` and its
    /// combined video band `v`.
    fn cancel(&self, out: &mut [f32], v: &[Complex64]) {
        let inner = &self.inner;
        if let Some((fs, fa, keep)) = inner.xtalk {
            let t0 = std::time::Instant::now();
            self.pool.install(|| cancel_crosstalk(out, v, fs, fa, keep));
            if std::env::var_os("LD_TIMING").is_some() {
                ld_decode::teeprintln!("STACKXT us={}", t0.elapsed().as_micros());
            }
        }
        if let Some((fs, band)) = inner.leak {
            let t0 = std::time::Instant::now();
            self.pool.install(|| cancel_chroma_leak(out, v, fs, band));
            if std::env::var_os("LD_TIMING").is_some() {
                ld_decode::teeprintln!("STACKLEAK us={}", t0.elapsed().as_micros());
            }
        }
    }
}

/// The read-ahead buffer: chunks of the stream computed by a producer thread
/// ahead of the decoder, so the stacking (and pass 0's progress it waits on)
/// overlaps the decode instead of stalling it on every read.
struct Ahead {
    state: Mutex<AheadState>,
    cv: Condvar,
}

#[derive(Default)]
struct AheadState {
    /// Chunk the decoder reads next.
    want: u64,
    chunks: std::collections::BTreeMap<u64, Arc<Vec<f32>>>,
    /// Stream length, once pass 0 has located the whole primary.
    len: Option<u64>,
    error: Option<String>,
    stop: bool,
}

impl Ahead {
    /// The producer: compute the missing chunks of `[want, want + AHEAD)`,
    /// in order. On the GPU the next chunk's plan (its file reads, and the
    /// wait for pass 0) is prepared while the current one runs, and the
    /// previous one's cancellations run on the CPU meanwhile (`held`), that
    /// chunk handed over once they are done.
    fn produce(&self, engine: &Engine) {
        let mut prepared: Option<(u64, Plan)> = None;
        let mut held: Option<Held> = None;
        loop {
            let c = {
                let mut st = self.state.lock().unwrap();
                loop {
                    if st.stop || st.error.is_some() {
                        return;
                    }
                    let end = st.len.map_or(u64::MAX, |l| l.div_ceil(CHUNK as u64));
                    let h = held.as_ref().map(|h| h.c);
                    if let Some(c) = (st.want..(st.want + AHEAD).min(end)).find(|c| !st.chunks.contains_key(c) && h != Some(*c)) {
                        break Some(c);
                    }
                    if held.is_some() {
                        break None;
                    }
                    st = self.cv.wait(st).unwrap();
                }
            };
            let done = match c {
                Some(c) => self.compute_chunk(engine, c, &mut prepared, &mut held),
                // Nothing more to compute for now: hand the held chunk over.
                None => held.take().map_or(Ok(()), |h| self.finish(engine, h)),
            };
            if let Err(e) = done {
                self.state.lock().unwrap().error = Some(format!("{e:#}"));
                self.cv.notify_all();
                return;
            }
        }
    }

    /// A combined chunk's cancellations, then the chunk handed over.
    fn finish(&self, engine: &Engine, mut h: Held) -> Result<()> {
        engine.cancel(&mut h.out, &h.v);
        self.deliver(engine, h.c, h.out, h.len)
    }

    /// Chunk `c` (`data`) handed over to the decoder; `len` is the stream's
    /// length once pass 0 knows it.
    fn deliver(&self, engine: &Engine, c: u64, data: Vec<f32>, len: Option<u64>) -> Result<()> {
        if let Some(out) = &engine.rf_out {
            out.write(c * CHUNK as u64, &data)?;
        }
        let mut st = self.state.lock().unwrap();
        if let Some(l) = len {
            st.len = Some(l);
        }
        st.chunks.insert(c, Arc::new(data));
        let keep_from = st.want.saturating_sub(BEHIND);
        st.chunks.retain(|&k, _| k >= keep_from);
        self.cv.notify_all();
        Ok(())
    }

    fn compute_chunk(&self, engine: &Engine, c: u64, prepared: &mut Option<(u64, Plan)>, held: &mut Option<Held>) -> Result<()> {
        let s0 = c * CHUNK as u64;
        // Wait for pass 0 to cover the chunk; that also says where the stream
        // ends once it is known.
        let len = engine.inner.live.wait_cover((s0 + (CHUNK + LOCK_MARGIN) as u64) as f64)?;
        let n = len.map_or(CHUNK, |l| (l.saturating_sub(s0) as usize).min(CHUNK));
        let plan = prepared.take().filter(|(pc, _)| *pc == c).map(|(_, p)| p);
        if engine.gpu.is_none() {
            let data = engine.compute(s0, n, None)?;
            return self.deliver(engine, c, data, len);
        }
        // Prepare the next chunk's plan, and finish the previous chunk,
        // alongside this chunk's GPU run.
        let next = c + 1;
        let next_ok = len.is_none_or(|l| next * (CHUNK as u64) < l);
        let prev = held.take();
        // On the stack's own pool: from this plain thread a bare `join`
        // would run on the global pool, which the decoders fill and then
        // block on this very buffer.
        let (combined, (next_plan, finished)) = engine.pool.install(|| {
            rayon::join(
                || engine.combine(s0, n, plan),
                || {
                    rayon::join(
                        || -> Result<Option<(u64, Plan)>> {
                            if !next_ok {
                                return Ok(None);
                            }
                            // Only a whole chunk: the last one, shorter, is
                            // planned when it is computed. (A known length
                            // alone is no reason to stop: pass 0 locates the
                            // whole primary well before the decode ends.)
                            let end = engine.inner.live.wait_cover(((next + 1) * CHUNK as u64 + LOCK_MARGIN as u64) as f64)?;
                            if end.is_some_and(|l| (next + 1) * CHUNK as u64 > l) {
                                return Ok(None);
                            }
                            Ok(Some((next, engine.gpu_plan(next * CHUNK as u64, CHUNK)?)))
                        },
                        || prev.map_or(Ok(()), |h| self.finish(engine, h)),
                    )
                },
            )
        });
        finished?;
        *prepared = next_plan?;
        let (out, v) = combined?;
        if engine.inner.wants_band() {
            *held = Some(Held { c, out, v, len });
            Ok(())
        } else {
            self.deliver(engine, c, out, len)
        }
    }

    /// Chunk `c`, waiting for the producer; None past the end.
    fn chunk(&self, c: u64) -> Result<Option<Arc<Vec<f32>>>> {
        let mut st = self.state.lock().unwrap();
        if st.want != c {
            st.want = c;
            // Drop what a seek has left behind.
            let keep_from = c.saturating_sub(BEHIND);
            st.chunks.retain(|&k, _| k >= keep_from);
            self.cv.notify_all();
        }
        loop {
            if let Some(e) = &st.error {
                bail!("{e}");
            }
            if let Some(d) = st.chunks.get(&c) {
                return Ok(Some(Arc::clone(d)));
            }
            if st.len.is_some_and(|l| c * CHUNK as u64 >= l) {
                return Ok(None);
            }
            st = self.cv.wait(st).unwrap();
        }
    }
}

/// A combined chunk waiting for its cancellations (`Ahead::finish`).
struct Held {
    c: u64,
    out: Vec<f32>,
    v: Vec<Complex64>,
    len: Option<u64>,
}

/// The read-time corrected (and, with several inputs, combined) RF stream.
pub struct VirtualSource {
    live: Arc<Live>,
    ahead: Arc<Ahead>,
    producer: Option<std::thread::JoinHandle<()>>,
    pos: u64,
}

impl VirtualSource {
    pub fn new(inputs: Vec<StackInput>, live: Arc<Live>, fs: f64, pal: bool, opencl: bool, exact: bool, carrier: bool, xtalk: bool, xtal: bool, leak: bool, rf_out: Option<RfOut>) -> Result<Self> {
        // 1/sigma^2 from each input's noise floor, measured where pass 0
        // first located it (the stream starts before the rest is known).
        let mut weight = Vec::new();
        let firsts = live.first_lines()?;
        let mut inputs = inputs;
        if xtal {
            for (s, first) in inputs.iter_mut().zip(&firsts) {
                s.xtal = CrystalLines::detect(&s.reader, first.unwrap_or(0.0) as i64, fs, &s.path, if pal { &PAL_LINE_EXCLUDE_HZ } else { &LINE_EXCLUDE_HZ }, if pal { 15625.0 } else { 15734.264 })?;
            }
        }
        for (s, first) in inputs.iter().zip(firsts) {
            let at = first.unwrap_or(0.0) as i64;
            let sig2 = noise_floor(&s.reader, at, fs)?;
            tracing::info!("stack: {} noise floor {:.4e} (weight {:.4e})", s.path, sig2, 1.0 / sig2);
            weight.push(1.0 / sig2);
        }
        tracing::info!("stack: {} input(s), read-time TBC as pass 0 locates the lines", inputs.len());
        let hi_lut = video_band_lut(fs, if pal { PAL_SPLIT_HZ } else { (SPLIT_LO_HZ, SPLIT_HI_HZ) });
        // The noise floor is a per-sample variance (white); through the video
        // band it is scaled by the filter's energy.
        let gain: f64 = hi_lut[..HI_TAPS].iter().map(|h| h.norm_sqr()).sum();
        let afloor = weight.iter().map(|w| AMP_FLOOR * (gain / w).sqrt()).collect();
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let fir = [
            sinc_lut().iter().map(|&v| v as f32).collect(),
            hi_lut.iter().map(|c| c.re as f32).collect(),
            hi_lut.iter().map(|c| c.im as f32).collect(),
        ];
        // NTSC only: measured there; PAL's subcarrier sits elsewhere.
        let leak = (leak && !pal).then_some((fs, LEAK_BAND_HZ));
        let xtalk = xtalk.then_some(if pal {
            (fs, PAL_AUDIO_CARRIERS_HZ, PAL_SPLIT_HZ)
        } else {
            (fs, AUDIO_CARRIERS_HZ, XTALK_KEEP_HZ)
        });
        let inner = Inner { inputs, live: Arc::clone(&live), weight, afloor, lag: carrier_lag(fs), hi_lut, fir, exact, carrier, xtalk, leak };
        let gpu = if opencl { Some(crate::opencl::StackGpu::new(&inner)?) } else { None };
        let engine = Engine { inner: Arc::new(inner), pool, gpu, rf_out };
        let ahead = Arc::new(Ahead { state: Mutex::new(AheadState::default()), cv: Condvar::new() });
        let producer = {
            let ahead = Arc::clone(&ahead);
            std::thread::Builder::new().name("stack-ahead".into()).spawn(move || ahead.produce(&engine))?
        };
        Ok(Self { live, ahead, producer: Some(producer), pos: 0 })
    }
}

fn noise_floor(file: &RawFile, start: i64, fs: f64) -> Result<f64> {
    let mut raw = vec![0f32; NOISE_FFT];
    file.read_at(start, &mut raw)?;
    let fwd = FftPlanner::new().plan_fft_forward(NOISE_FFT);
    let mut x: Vec<Complex64> = raw.iter().map(|&v| Complex64::new(f64::from(v), 0.0)).collect();
    fwd.process(&mut x);
    let (lo, hi) = noise_band(fs)?;
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

/// Audio crosstalk cancellation: take out of the stream `out` the part of its
/// demodulated video that follows the two analog audio carriers. On the disc
/// they leak into the video carrier's phase; the leak is the same on every
/// copy (one soundtrack), so no stacking removes it. Each carrier is taken out
/// of the RF itself (mixed down, three boxcars, mixed back up: `r`), and the
/// video band's instantaneous frequency (from `v`, the combined analytic video
/// band) is fitted on both by least squares (the picture does not follow the
/// carriers): one complex gain each. That frequency error is a phase error
/// `theta = Re(C r / (j f))`, and the video band is turned back by it:
/// `out + Re(v (exp(-j theta) - 1))`, kept above `keep` so the low band
/// (audio, EFM) is left as it is. Only from this chunk's samples, so the same
/// wherever it is computed.
fn cancel_crosstalk(out: &mut [f32], v: &[Complex64], fs: f64, fa: [f64; 2], keep: (f64, f64)) {
    let n = out.len();
    let taper = (XTALK_TAPER_S * fs).round() as usize;
    let boxl = ((XTALK_BOX_S * fs).round() as usize).max(1);
    if n < 4 * taper {
        return;
    }
    let tau = 2.0 * std::f64::consts::PI;
    let zero = Complex64::new(0.0, 0.0);
    // Each carrier mixed down to 0 Hz and through a boxcar at the full rate,
    // then, as it is now narrow, kept every XTALK_DEC samples through two
    // more boxcars there, read back by linear interpolation.
    let dec = XTALK_DEC;
    let nd = n.div_ceil(dec);
    let box_d = (boxl as f64 / dec as f64).round().max(1.0) as usize;
    let boxcar = |z: &[Complex64], len: usize| -> Vec<Complex64> {
        let m = z.len();
        let mut c = Vec::with_capacity(m + 1);
        let mut run = zero;
        c.push(run);
        for &x in z {
            run += x;
            c.push(run);
        }
        (0..m)
            .map(|i| {
                let lo = i.saturating_sub(len / 2);
                (c[(lo + len).min(m)] - c[lo]) / len as f64
            })
            .collect()
    };
    // Per carrier: its phase step, and its narrow band at 0 Hz.
    let ws: Vec<f64> = fa.iter().map(|&f| -tau * f / fs).collect();
    let mixed: Vec<Vec<Complex64>> = ws
        .par_iter()
        .map(|&w| {
            // The full-rate boxcar at the kept samples only, from running sums
            // over each span of XTALK_BOX_SPAN kept samples, the carrier's
            // phasor exact at its start and turned by one step from there.
            let turn = Complex64::from_polar(1.0, w);
            let mut first = vec![zero; nd];
            first.par_chunks_mut(XTALK_BOX_SPAN).enumerate().for_each(|(c, o)| {
                let k0 = c * XTALK_BOX_SPAN;
                let lo_at = |k: usize| (k * dec).saturating_sub(boxl / 2);
                let (lo0, hi1) = (lo_at(k0), (lo_at(k0 + o.len() - 1) + boxl).min(n));
                let mut sums = Vec::with_capacity(hi1 - lo0 + 1);
                let mut run = zero;
                let mut p = Complex64::from_polar(1.0, w * lo0 as f64);
                sums.push(run);
                for t in lo0..hi1 {
                    run += p * f64::from(out[t]);
                    p *= turn;
                    sums.push(run);
                }
                for (j, y) in o.iter_mut().enumerate() {
                    let lo = lo_at(k0 + j);
                    *y = (sums[(lo + boxl).min(n) - lo0] - sums[lo - lo0]) / boxl as f64;
                }
            });
            boxcar(&boxcar(&first, box_d), box_d)
        })
        .collect();
    // The carrier at `t` (analytic, at its frequency again), from its
    // phasor there `ph` (mixing back up: `exp(-j w t)`).
    let carrier_with = |c: usize, t: usize, ph: Complex64| -> Complex64 {
        let k = t / dec;
        let a = (t % dec) as f64 / dec as f64;
        let l = &mixed[c];
        let b = if k + 1 < nd { l[k] * (1.0 - a) + l[k + 1] * a } else { l[k] };
        b * ph * 2.0
    };
    let carrier_at = |c: usize, t: usize| carrier_with(c, t, Complex64::from_polar(1.0, -ws[c] * t as f64));
    let cols = |t: usize| {
        let (r0, r1) = (carrier_at(0, t), carrier_at(1, t));
        [r0.re, r0.im, r1.re, r1.im]
    };
    // Least squares on every XTALK_FIT_STRIDE-th sample away from the edges,
    // means removed.
    let idx: Vec<usize> = (taper..n - taper).step_by(XTALK_FIT_STRIDE).collect();
    let cnt = idx.len() as f64;
    let rows: Vec<([f64; 4], f64)> = idx.par_iter().map(|&t| (cols(t), (v[t] * v[t - 1].conj()).arg() * fs / tau)).collect();
    let (mut mx, mut my) = ([0.0f64; 4], 0.0f64);
    for (c, y) in &rows {
        for k in 0..4 {
            mx[k] += c[k];
        }
        my += y;
    }
    for m in &mut mx {
        *m /= cnt;
    }
    my /= cnt;
    let (mut ata, mut aty) = ([[0.0f64; 4]; 4], [0.0f64; 4]);
    for (c, y) in &rows {
        let y = y - my;
        for a in 0..4 {
            let ca = c[a] - mx[a];
            aty[a] += ca * y;
            for b in 0..4 {
                ata[a][b] += ca * (c[b] - mx[b]);
            }
        }
    }
    let Some(k) = solve4(ata, aty) else { return };
    let gain = [Complex64::new(k[0], -k[1]), Complex64::new(k[2], -k[3])];
    // The correction at `t` (the carriers' phasors there `ph`), computed
    // where the high-pass reads it.
    let corr_with = |t: usize, ph: &[Complex64; 2]| -> f64 {
        let mut th = 0.0;
        for c in 0..2 {
            th += (gain[c] * carrier_with(c, t, ph[c]) / Complex64::new(0.0, fa[c])).re;
        }
        let e = t.min(n - 1 - t);
        if e < taper {
            th *= 0.5 - 0.5 * (std::f64::consts::PI * e as f64 / taper as f64).cos();
        }
        // The turn is a few hundredths of a radian: its series, to the
        // last bit that matters.
        let (sn, cm1) = if th.abs() < 0.05 {
            let t2 = th * th;
            (th * (1.0 - t2 / 6.0), t2 * (t2 / 24.0 - 0.5))
        } else {
            let (sn, cs) = th.sin_cos();
            (sn, cs - 1.0)
        };
        v[t].re * cm1 + v[t].im * sn
    };
    // Only its part in the video band (`keep`, a raised-cosine edge): below,
    // turning the video band also moves what lands on the audio carriers and
    // the EFM, which the video never reads. Overlap-save, blocks in parallel.
    // In single precision, ample for a correction this small; two blocks
    // to each complex FFT, the first as its real part and the second as its
    // imaginary part (the edge's gain is real and even).
    let size = XTALK_FFT_LEN;
    let ov = ((XTALK_OV_S * fs).round() as usize).clamp(1, size / 4);
    let step = size - 2 * ov;
    let (fwd, inv) = {
        let mut planner = LEAK_FFT.get_or_init(|| Mutex::new(FftPlanner::new())).lock().unwrap();
        (planner.plan_fft_forward(size), planner.plan_fft_inverse(size))
    };
    let gain_at: Vec<f32> = (0..size)
        .map(|b| {
            let f = b.min(size - b) as f64 * fs / size as f64;
            let g = if f <= keep.0 {
                0.0
            } else if f >= keep.1 {
                1.0
            } else {
                0.5 - 0.5 * (std::f64::consts::PI * (f - keep.0) / (keep.1 - keep.0)).cos()
            };
            (g / size as f64) as f32
        })
        .collect();
    // A block's correction from `start`, the carriers' phasors exact there
    // and turned by one step from there.
    let unturn = [Complex64::from_polar(1.0, -ws[0]), Complex64::from_polar(1.0, -ws[1])];
    let block = |start: isize, put: &mut dyn FnMut(usize, f32)| {
        let mut ph = [Complex64::from_polar(1.0, -ws[0] * start as f64), Complex64::from_polar(1.0, -ws[1] * start as f64)];
        for i in 0..size {
            let t = start + i as isize;
            put(i, if t >= 0 && (t as usize) < n { corr_with(t as usize, &ph) as f32 } else { 0.0 });
            ph[0] *= unturn[0];
            ph[1] *= unturn[1];
        }
    };
    out.par_chunks_mut(2 * step).enumerate().for_each(|(pair, o)| {
        let s0 = (2 * pair * step) as isize - ov as isize;
        let mut buf = vec![Complex32::new(0.0, 0.0); size];
        block(s0, &mut |i, x| buf[i].re = x);
        block(s0 + step as isize, &mut |i, x| buf[i].im = x);
        fwd.process(&mut buf);
        for (x, g) in buf.iter_mut().zip(&gain_at) {
            *x *= *g;
        }
        inv.process(&mut buf);
        for (i, y) in o.iter_mut().enumerate() {
            let d = if i < step { buf[ov + i].re } else { buf[ov + i - step].im };
            *y += d;
        }
    });
}

/// FFT plans of `cancel_crosstalk` and `cancel_chroma_leak`.
static LEAK_FFT: OnceLock<Mutex<FftPlanner<f32>>> = OnceLock::new();

/// The narrow lines the player's and the capture's clocks put into a
/// capture, subtracted from its samples as they are read (`clean`), before
/// anything else. Two kinds: the player's EFM decoder crystal (found near
/// `EFM_XTAL_HZ`), whose lines are the multiples of its 192nd (`XTAL_DIV`,
/// 88.2 kHz) that stand out of the spectrum, its drift followed as it goes;
/// and any other line that stands out at one frequency all through the
/// capture (the player's 4fsc video clock, the capture board's clocks, and
/// at a low sample rate what of them folds down), outside the bands where
/// the disc's own signal makes steady lines (`exclude`).
pub(crate) struct CrystalLines {
    /// The EFM crystal's 192nd (Hz), the multiples found, and the strongest
    /// one, the drift is followed on (none found: `ks` empty).
    f0: f64,
    ks: Vec<u32>,
    reference: u32,
    /// The other lines (Hz).
    others: Vec<f64>,
    fs: f64,
    /// Measured super-blocks (absolute index), the most recent last.
    cache: Mutex<Vec<(i64, Arc<CrystalBlock>)>>,
}

/// The lines over one super-block of the capture (`XTAL_SUPER` samples
/// from `k * XTAL_SUPER`): the EFM crystal's frequency there, and every
/// line's complex amplitude on each `XTAL_BLOCK` within it (the EFM lines
/// first, then the others).
struct CrystalBlock {
    f0: f64,
    amps: Vec<Vec<Complex64>>,
}

impl CrystalLines {
    /// Find the lines, on `XTAL_SEGMENTS` stretches spread over the capture
    /// from `start` (a line has to be in every one, at the same frequency).
    /// A line with another one a line period's frequency (`line_hz`) away is
    /// the picture's (a still picture's RF is a comb of them) and is left.
    fn detect(file: &RawFile, start: i64, fs: f64, path: &str, exclude: &[(f64, f64)], line_hz: f64) -> Result<Option<Self>> {
        let n = XTAL_FFT;
        let fft = FftPlanner::new().plan_fft_forward(n);
        let win: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
        let total = file.len as i64;
        let room = (total - start - n as i64).max(0);
        let mut segs: Vec<Vec<f64>> = Vec::new();
        let mut raw = vec![0f32; n];
        for seg in 0..XTAL_SEGMENTS {
            let at = start + room * seg as i64 / (XTAL_SEGMENTS as i64 - 1).max(1);
            file.read_at(at, &mut raw)?;
            let mean = raw.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
            let mut x: Vec<Complex64> = raw.iter().zip(&win).map(|(&v, w)| Complex64::new((f64::from(v) - mean) * w, 0.0)).collect();
            fft.process(&mut x);
            segs.push(x[..n / 2].iter().map(|v| v.norm_sqr()).collect());
        }
        let power: Vec<f64> = (0..n / 2).map(|b| segs.iter().map(|s| s[b]).sum()).collect();
        let bin = fs / n as f64;
        // The spectrum's floor: the median within XTAL_FLOOR_HZ, taken every
        // FLOOR_STEP bins.
        const FLOOR_STEP: usize = 128;
        let h = (XTAL_FLOOR_HZ / bin) as usize;
        let floor_of = |p: &[f64]| -> Vec<f64> {
            (0..p.len().div_ceil(FLOOR_STEP))
                .into_par_iter()
                .map(|i| {
                    let c = i * FLOOR_STEP;
                    let mut w: Vec<f64> = p[c.saturating_sub(h)..(c + h).min(p.len())].to_vec();
                    w.sort_by(f64::total_cmp);
                    w[w.len() / 2]
                })
                .collect()
        };
        let floor_sum = floor_of(&power);
        let floor = |b: usize| floor_sum[(b / FLOOR_STEP).min(floor_sum.len() - 1)];
        // The highest bin within `w` bins of `b`, with its parabolic offset.
        let peak = |p: &[f64], b: usize, w: usize| -> (usize, f64) {
            let k = (b.saturating_sub(w)..=(b + w).min(p.len() - 2)).max_by(|&i, &j| p[i].total_cmp(&p[j])).unwrap_or(b);
            let (l, c, r) = (p[k.max(1) - 1].ln(), p[k].ln(), p[k + 1].ln());
            let d = l - 2.0 * c + r;
            (k, if d < 0.0 { 0.5 * (l - r) / d } else { 0.0 })
        };
        let top = XTAL_TOP_HZ.min(0.49 * fs);
        // The EFM crystal and its multiples.
        let (mut f0, mut ks, mut best) = (EFM_XTAL_HZ / XTAL_DIV, Vec::new(), (0.0f64, 0u32));
        let (lo, hi) = ((EFM_XTAL_HZ * (1.0 - XTAL_SEARCH) / bin) as usize, (EFM_XTAL_HZ * (1.0 + XTAL_SEARCH) / bin) as usize);
        if hi + 2 < power.len() {
            let (k, off) = peak(&power, (lo + hi) / 2, (hi - lo) / 2);
            if power[k] >= floor(k) * XTAL_FIND {
                f0 = (k as f64 + off) * bin / XTAL_DIV;
                for m in 1..=(top / f0) as u32 {
                    let (k, _) = peak(&power, (f64::from(m) * f0 / bin).round() as usize, 3);
                    let ratio = power[k] / floor(k);
                    if ratio >= XTAL_KEEP {
                        ks.push(m);
                        if ratio > best.0 {
                            best = (ratio, m);
                        }
                    }
                }
            }
        }
        // Every other line that stands out in each stretch, at one frequency.
        let seg_floors: Vec<Vec<f64>> = segs.iter().map(|s| floor_of(s)).collect();
        let tall = |s: usize, b: usize| {
            let p = &segs[s];
            b >= 3 && b + 3 < p.len() && p[b] >= seg_floors[s][b / FLOOR_STEP] * XTAL_KEEP && (b - 3..=b + 3).all(|i| p[i] <= p[b])
        };
        let near = |s: usize, b: usize| (b.saturating_sub(1)..=b + 1).any(|i| tall(s, i));
        let efm_line = |f: f64| ks.iter().any(|&m| (f64::from(m) * f0 - f).abs() < 3.0 * bin);
        let comb = |f: f64| {
            [f - line_hz, f + line_hz].iter().any(|&g| {
                let (k, _) = peak(&power, (g / bin).round() as usize, 2);
                power[k] >= floor(k) * XTAL_KEEP
            })
        };
        let mut others = Vec::new();
        for b in ((XTAL_LOW_HZ / bin) as usize)..((top / bin) as usize).min(power.len() - 2) {
            if !tall(0, b) || !(1..segs.len()).all(|s| near(s, b)) {
                continue;
            }
            let (k, off) = peak(&power, b, 1);
            let f = (k as f64 + off) * bin;
            if exclude.iter().any(|&(a, z)| f >= a && f <= z) || efm_line(f) || comb(f) || others.iter().any(|&g: &f64| (g - f).abs() < 3.0 * bin) {
                continue;
            }
            others.push(f);
        }
        if !ks.is_empty() {
            tracing::info!(
                "stack: {path}: EFM decoder crystal at {:.5} MHz, {} of its lines subtracted ({} MHz)",
                f0 * XTAL_DIV / 1e6,
                ks.len(),
                ks.iter().map(|&m| format!("{:.4}", f64::from(m) * f0 / 1e6)).collect::<Vec<_>>().join(", ")
            );
        }
        if !others.is_empty() {
            tracing::info!(
                "stack: {path}: {} other steady lines subtracted ({} MHz)",
                others.len(),
                others.iter().map(|f| format!("{:.4}", f / 1e6)).collect::<Vec<_>>().join(", ")
            );
        }
        Ok((!ks.is_empty() || !others.is_empty()).then_some(Self { f0, ks, reference: best.1, others, fs, cache: Mutex::new(Vec::new()) }))
    }

    /// Super-block `k`, measured or from the cache.
    fn block(&self, file: &RawFile, k: i64) -> Result<Arc<CrystalBlock>> {
        if let Some((_, b)) = self.cache.lock().unwrap().iter().find(|(i, _)| *i == k) {
            return Ok(Arc::clone(b));
        }
        let mut raw = vec![0f32; XTAL_SUPER];
        file.read_at(k * XTAL_SUPER as i64, &mut raw)?;
        let b = Arc::new(self.measure(&raw));
        let mut cache = self.cache.lock().unwrap();
        if !cache.iter().any(|(i, _)| *i == k) {
            cache.push((k, Arc::clone(&b)));
            if cache.len() > XTAL_CACHE {
                cache.remove(0);
            }
        }
        Ok(b)
    }

    /// The lines over one super-block: the EFM crystal's drift there from
    /// its reference line's phase advance from block to block, then every
    /// line's amplitude on every block. Phases are counted from the
    /// super-block's first sample.
    fn measure(&self, raw: &[f32]) -> CrystalBlock {
        let (fs, block) = (self.fs, XTAL_BLOCK);
        let nb = raw.len() / block;
        let tau = 2.0 * std::f64::consts::PI;
        // The block's DFT at f, phase from the super-block's first sample
        // (Goertzel's recurrence, then its phase turned back).
        let at = |f: f64, a: usize| -> Complex64 {
            let w = tau * f / fs;
            let k = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for &x in &raw[a..a + block] {
                let s0 = f64::from(x) + k * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let y = Complex64::new(s1, 0.0) - Complex64::from_polar(s2, -w);
            y * Complex64::from_polar(2.0 / block as f64, -w * (a + block - 1) as f64)
        };
        let mut f0 = self.f0;
        if !self.ks.is_empty() {
            let fr = f64::from(self.reference) * self.f0;
            let phases: Vec<f64> = (0..nb).into_par_iter().map(|b| at(fr, b * block).arg()).collect();
            let mut adv = 0.0;
            for w in phases.windows(2) {
                let mut d = w[1] - w[0];
                d -= tau * (d / tau).round();
                adv += d;
            }
            let df = adv / (nb - 1) as f64 / (tau * block as f64 / fs);
            f0 = self.f0 * (1.0 + df / fr);
        }
        let freqs: Vec<f64> = self.ks.iter().map(|&m| f64::from(m) * f0).chain(self.others.iter().copied()).collect();
        let amps = freqs.par_iter().map(|&f| (0..nb).map(|b| at(f, b * block)).collect()).collect();
        CrystalBlock { f0, amps }
    }

    /// Subtract the lines from `raw`, the capture's samples from `start`.
    fn clean(&self, file: &RawFile, start: i64, raw: &mut [f32]) -> Result<()> {
        let sb = XTAL_SUPER as i64;
        let end = (start + raw.len() as i64).min(file.len as i64);
        let mut t = start.max(0);
        let tau = 2.0 * std::f64::consts::PI;
        let nb = XTAL_SUPER / XTAL_BLOCK;
        while t < end {
            let k = t.div_euclid(sb);
            let (b0, b1) = (k * sb, ((k + 1) * sb).min(end));
            let blk = self.block(file, k)?;
            let w: Vec<f64> = self
                .ks
                .iter()
                .map(|&m| tau * f64::from(m) * blk.f0 / self.fs)
                .chain(self.others.iter().map(|&f| tau * f / self.fs))
                .collect();
            let span = &mut raw[(t - start) as usize..(b1 - start) as usize];
            let first = (t - b0) as usize;
            let turn: Vec<Complex64> = w.iter().map(|&wl| Complex64::from_polar(1.0, wl)).collect();
            // Pieces aligned on every XTAL_SEED-th sample of the super-block
            // (the first one up to the next): each line's phasor is exact at
            // that sample and turned by one step from there (a read that
            // starts between turns up to it), so a sample gets the same value
            // whatever read it is in. A piece lies between two block centres,
            // so every line's amplitude is one straight segment over it.
            let head = ((XTAL_SEED - first % XTAL_SEED) % XTAL_SEED).min(span.len());
            let (h, rest) = span.split_at_mut(head);
            let pieces: Vec<(usize, &mut [f32])> =
                std::iter::once((first, h)).chain(rest.chunks_mut(XTAL_SEED).enumerate().map(|(c, o)| (first + head + c * XTAL_SEED, o))).filter(|(_, o)| !o.is_empty()).collect();
            pieces.into_par_iter().for_each(|(a, o)| {
                let at = a - a % XTAL_SEED;
                let u0 = (at as f64 + 0.5) / XTAL_BLOCK as f64 - 0.5;
                let j = (u0.floor().max(0.0) as usize).min(nb - 1);
                let s: Vec<f64> = (a..a + o.len()).map(|pos| ((pos as f64 + 0.5) / XTAL_BLOCK as f64 - 0.5 - j as f64).clamp(0.0, 1.0)).collect();
                let mut v: Vec<f64> = o.iter().map(|&y| f64::from(y)).collect();
                for ((amp, &wl), &tn) in blk.amps.iter().zip(&w).zip(&turn) {
                    let (g0, dg) = if j + 1 < nb { (amp[j], amp[j + 1] - amp[j]) } else { (amp[j], Complex64::new(0.0, 0.0)) };
                    let mut p = Complex64::from_polar(1.0, wl * at as f64);
                    for _ in at..a {
                        p *= tn;
                    }
                    for (y, &si) in v.iter_mut().zip(&s) {
                        *y -= ((g0 + dg * si) * p).re;
                        p *= tn;
                    }
                }
                for (o, y) in o.iter_mut().zip(&v) {
                    *o = *y as f32;
                }
            });
            t = b1;
        }
        Ok(())
    }
}

/// The chroma leak cancellation's two signals through a band of
/// raised-cosine edges `rise` and `fall` (Hz): the instantaneous frequency
/// of the video band `v` (Hz) and the RF `x`, by overlap-save FFT blocks in
/// parallel (zero outside them), each block computing its own stretch of
/// both. One complex FFT carries the pair, the frequency as its real part
/// and the RF as its imaginary part, the band's gain being real and even;
/// in single precision, ample for a correction this small.
fn leak_bands(v: &[Complex64], x: &[f32], fs: f64, rise: (f64, f64), fall: (f64, f64)) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let size = LEAK_FFT_LEN;
    let ov = ((XTALK_OV_S * fs).round() as usize).clamp(1, size / 4);
    let step = size - 2 * ov;
    let (fwd, inv) = {
        let mut planner = LEAK_FFT.get_or_init(|| Mutex::new(FftPlanner::new())).lock().unwrap();
        (planner.plan_fft_forward(size), planner.plan_fft_inverse(size))
    };
    let rc = |f: f64, (a, b): (f64, f64)| {
        if f <= a {
            0.0
        } else if f >= b {
            1.0
        } else {
            0.5 - 0.5 * (std::f64::consts::PI * (f - a) / (b - a)).cos()
        }
    };
    let gain: Vec<f32> = (0..size)
        .map(|b| {
            let f = b.min(size - b) as f64 * fs / size as f64;
            (rc(f, rise) * (1.0 - rc(f, fall)) / size as f64) as f32
        })
        .collect();
    let hz = fs / (2.0 * std::f64::consts::PI);
    let (mut oc, mut or) = (vec![0.0f32; n], vec![0.0f32; n]);
    oc.par_chunks_mut(step).zip(or.par_chunks_mut(step)).enumerate().for_each(|(blk, (a, b))| {
        let s0 = blk * step;
        let mut buf: Vec<Complex32> = (0..size)
            .map(|i| {
                let t = (s0 + i) as isize - ov as isize;
                if t >= 1 && (t as usize) < n {
                    let t = t as usize;
                    Complex32::new(((v[t] * v[t - 1].conj()).arg() * hz) as f32, x[t])
                } else if t == 0 {
                    Complex32::new(0.0, x[0])
                } else {
                    Complex32::new(0.0, 0.0)
                }
            })
            .collect();
        fwd.process(&mut buf);
        for (v, g) in buf.iter_mut().zip(&gain) {
            *v *= *g;
        }
        inv.process(&mut buf);
        for (i, (p, q)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            (*p, *q) = (buf[ov + i].re, buf[ov + i].im);
        }
    });
    (oc, or)
}

/// Solve the n x n system `a x = b` (Gaussian elimination, partial
/// pivoting); `None` when it is singular.
fn solve_n(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 || !a[p][c].is_finite() {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0f64; n];
    for c in (0..n).rev() {
        let mut s = b[c];
        for k in c + 1..n {
            s -= a[c][k] * x[k];
        }
        x[c] = s / a[c][c];
    }
    Some(x)
}

/// Chroma leak cancellation: take out of the stream `out` the composite
/// chroma that leaks into the RF baseband around the colour subcarrier. It
/// beats with the video carrier (carrier minus 3.58 MHz) and makes the fine
/// wavering vertical stripes of saturated areas, the NTSC "colour wobble".
/// The chroma is the demodulated video band `v`'s instantaneous frequency
/// through `band`; the RF's same band is fitted on it, shifted by
/// `LEAK_TAPS` lags `LEAK_STEP` apart (least squares, every
/// `LEAK_FIT_STRIDE`-th sample: the picture's own RF there does not follow
/// it), and the fit subtracted, fading in and out at the chunk's edges. Only
/// from this chunk's samples, so the same wherever it is computed.
fn cancel_chroma_leak(out: &mut [f32], v: &[Complex64], fs: f64, band: ((f64, f64), (f64, f64))) {
    let n = out.len();
    let taper = (XTALK_TAPER_S * fs).round() as usize;
    let half = LEAK_TAPS / 2 * LEAK_STEP;
    if n < 4 * taper + 2 * half {
        return;
    }
    let (c, r) = leak_bands(v, out, fs, band.0, band.1);
    let lags: Vec<isize> = (0..LEAK_TAPS).map(|k| (k * LEAK_STEP) as isize - half as isize).collect();
    let col = |t: usize, k: usize| f64::from(c[(t as isize + lags[k]) as usize]);
    let m = lags.len();
    // The normal equations, summed over fixed spans in parallel and the
    // spans added in order: the same sums whatever the thread count.
    let fit: Vec<usize> = (taper + half..n - taper - half).step_by(LEAK_FIT_STRIDE).collect();
    let parts: Vec<(Vec<Vec<f64>>, Vec<f64>, f64)> = fit
        .par_chunks(LEAK_FIT_SPAN)
        .map(|ts| {
            let (mut ata, mut aty, mut yy) = (vec![vec![0.0f64; m]; m], vec![0.0f64; m], 0.0f64);
            let mut row = vec![0.0f64; m];
            for &t in ts {
                for (k, v) in row.iter_mut().enumerate() {
                    *v = col(t, k);
                }
                let rt = f64::from(r[t]);
                yy += rt * rt;
                for a in 0..m {
                    aty[a] += row[a] * rt;
                    for b in a..m {
                        ata[a][b] += row[a] * row[b];
                    }
                }
            }
            (ata, aty, yy)
        })
        .collect();
    let (mut ata, mut aty, mut yy) = (vec![vec![0.0f64; m]; m], vec![0.0f64; m], 0.0f64);
    for (pa, py, pyy) in &parts {
        yy += pyy;
        for a in 0..m {
            aty[a] += py[a];
            for b in a..m {
                ata[a][b] += pa[a][b];
            }
        }
    }
    for a in 0..m {
        for b in 0..a {
            ata[a][b] = ata[b][a];
        }
    }
    let Some(h) = solve_n(ata, aty.clone()) else { return };
    // Share of the RF's chroma band the leak explains: below LEAK_MIN_SHARE
    // there is no leak to speak of, and the chunk is left as it is.
    let share = h.iter().zip(&aty).map(|(a, b)| a * b).sum::<f64>() / yy.max(1e-300);
    tracing::debug!("stack: chroma leak share {share:.4}");
    if !(share >= LEAK_MIN_SHARE) {
        return;
    }
    const SPAN: usize = 1 << 14;
    out.par_chunks_mut(SPAN).enumerate().for_each(|(blk, o)| {
        for (i, o) in o.iter_mut().enumerate() {
            let t = blk * SPAN + i;
            if t < half || t + half >= n {
                continue;
            }
            let w = &c[t - half..=t + half];
            let mut leak = 0.0;
            for (k, hk) in h.iter().enumerate() {
                leak += hk * f64::from(w[k * LEAK_STEP]);
            }
            let e = t.min(n - 1 - t);
            if e < taper {
                leak *= 0.5 - 0.5 * (std::f64::consts::PI * e as f64 / taper as f64).cos();
            }
            *o = (f64::from(*o) - leak) as f32;
        }
    });
}

/// Solve the 4x4 system `a x = b` (Gaussian elimination, partial pivoting);
/// `None` when it is singular.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for c in 0..4 {
        let p = (c..4).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 || !a[p][c].is_finite() {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..4 {
            let f = a[r][c] / a[c][c];
            for k in c..4 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0f64; 4];
    for c in (0..4).rev() {
        let mut s = b[c];
        for k in c + 1..4 {
            s -= a[c][k] * x[k];
        }
        x[c] = s / a[c][c];
    }
    Some(x)
}

fn compute_task(inner: &Inner, s0: u64, out: &mut [f32], v: Option<&mut [Complex64]>) -> Result<()> {
    let n = out.len();
    let m = LOCK_MARGIN;
    let len = n + 2 * m;
    let plan = inner.plan(s0 as f64 - m as f64, len, false)?;
    let [sinc, hi_re, hi_im] = &inner.fir;
    let (sinc64, hi64) = (sinc_lut(), &inner.hi_lut);

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
                let k0 = k0 - span.start;
                let xs = &span.raw[(k0 - (SINC_TAPS / 2 - 1) as i64) as usize..][..SINC_TAPS];
                let xh = &span.raw[(k0 - (HI_TAPS / 2 - 1) as i64) as usize..][..HI_TAPS];
                let row = ph * HI_TAPS..(ph + 1) * HI_TAPS;
                if inner.exact {
                    // Plain sequential f64 sums: the OpenCL kernel's order.
                    let mut f = 0.0f64;
                    for (h, &x) in sinc64[ph * SINC_TAPS..][..SINC_TAPS].iter().zip(xs) {
                        f += h * f64::from(x);
                    }
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (h, &x) in hi64[row].iter().zip(xh) {
                        acc.re += h.re * f64::from(x);
                        acc.im += h.im * f64::from(x);
                    }
                    full[j] = f;
                    hi[j] = acc;
                } else {
                    full[j] = f64::from(dot8(&sinc[ph * SINC_TAPS..][..SINC_TAPS], xs));
                    hi[j] = Complex64::new(f64::from(dot8(&hi_re[row.clone()], xh)), f64::from(dot8(&hi_im[row], xh)));
                }
                cov[j] = true;
            }
            (full, hi, cov)
        })
        .collect();
    if !inner.carrier {
        combine_fixed(inner, &reads, m, out, v);
        return Ok(());
    }

    // Every input weighted at every sample by its carrier amplitude over a
    // CARRIER_WIN window (`carrier`, `amp_weight`), and its video band
    // phase-locked onto the primary's over a LOCK_WIN window (`lock`, with the
    // alignment guard). The rest of each input (full minus video band) is
    // summed as read.
    let (half, ch) = (LOCK_WIN / 2, CARRIER_WIN / 2);
    let win = |j: usize| j.saturating_sub(half)..(j + half).min(len);
    let short = |j: usize, t: usize| t + ch >= j && t < j + ch;
    let p2 = |z: Complex64| z.re * z.re + z.im * z.im;
    // Running sums of a per-sample quantity (the fast path's windows).
    let prefix = |f: &dyn Fn(usize) -> f64| {
        let mut s = Vec::with_capacity(len + 1);
        let mut run = 0.0;
        s.push(run);
        for t in 0..len {
            run += f(t);
            s.push(run);
        }
        s
    };
    let cprefix = |f: &dyn Fn(usize) -> Complex64| {
        let mut s = Vec::with_capacity(len + 1);
        let mut run = Complex64::new(0.0, 0.0);
        s.push(run);
        for t in 0..len {
            run += f(t);
            s.push(run);
        }
        s
    };
    let wsum_of = |s: &[f64], j: usize, h: usize| s[(j + h).min(len)] - s[j.saturating_sub(h)];
    let cwsum_of = |s: &[Complex64], j: usize, h: usize| s[(j + h).min(len)] - s[j.saturating_sub(h)];
    let (afl, wt, lag) = (&inner.afloor, &inner.weight, inner.lag);
    let (full0, hi0, cov0) = &reads[0];
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    let mut vacc = vec![Complex64::new(0.0, 0.0); if v.is_some() { n } else { 0 }];
    // The primary's carrier fraction over the lock window at each sample, for
    // the others' locks.
    let mut q0 = vec![0.0f64; n];
    // And how much of its carrier it keeps at the sample itself (carrier
    // window against lock window), for the alignment guard.
    let mut h0 = vec![0.0f64; n];
    // An input's carrier terms at `t`: the lag product, `|z[t]|^2`, 1 (zero
    // where `t` or `t - lag` is not covered).
    let lagged = |hi: &[Complex64], cov: &[bool], t: usize| {
        if t >= lag && cov[t] && cov[t - lag] {
            let (x, y) = (hi[t], hi[t - lag]);
            (Complex64::new(x.re * y.re + x.im * y.im, x.im * y.re - x.re * y.im), p2(x), 1.0)
        } else {
            (Complex64::new(0.0, 0.0), 0.0, 0.0)
        }
    };
    // An input's carrier sums at `j` over the lock window and the carrier
    // window. Exact: directly, in ascending order (the OpenCL kernel's),
    // instead of running-sum differences whose rounding depends on where the
    // task starts.
    type Carrier = ((Complex64, f64, f64), (Complex64, f64, f64));
    let carrier_sums = |hi: &[Complex64], cov: &[bool], pre: &Option<(Vec<Complex64>, Vec<f64>, Vec<f64>)>, j: usize| -> Carrier {
        match pre {
            Some((sr, se, sc)) => (
                (cwsum_of(sr, j, half), wsum_of(se, j, half), wsum_of(sc, j, half)),
                (cwsum_of(sr, j, ch), wsum_of(se, j, ch), wsum_of(sc, j, ch)),
            ),
            None => {
                let (mut l, mut s) = ((Complex64::new(0.0, 0.0), 0.0, 0.0), (Complex64::new(0.0, 0.0), 0.0, 0.0));
                for t in win(j) {
                    let (v, p, k) = lagged(hi, cov, t);
                    l.0.re += v.re;
                    l.0.im += v.im;
                    l.1 += p;
                    l.2 += k;
                    if short(j, t) {
                        s.0.re += v.re;
                        s.0.im += v.im;
                        s.1 += p;
                        s.2 += k;
                    }
                }
                (l, s)
            }
        }
    };
    let carrier_pre = |hi: &[Complex64], cov: &[bool]| {
        (!inner.exact).then(|| {
            (cprefix(&|t| lagged(hi, cov, t).0), prefix(&|t| lagged(hi, cov, t).1), prefix(&|t| lagged(hi, cov, t).2))
        })
    };
    let pre0 = carrier_pre(hi0, cov0);
    for j in m..m + n {
        let ((rl, e, c), (rs, es, cs)) = carrier_sums(hi0, cov0, &pre0, j);
        let s = if cs > 0.0 { carrier(rs, es, cs, QN_CARRIER).0 } else { 0.0 };
        if c > 0.0 {
            let (sl, q) = carrier(rl, e, c, QN_LOCK);
            q0[j - m] = q;
            h0[j - m] = if sl > 0.0 { s / sl } else { 0.0 };
        }
        if cov0[j] {
            let w = amp_weight(s, afl[0], wt[0]);
            acc[j - m] += full0[j] * w;
            wsum[j - m] += w;
            if !vacc.is_empty() {
                vacc[j - m] += hi0[j] * w;
            }
        }
    }
    for (i, (full, hi, cov)) in reads.iter().enumerate().skip(1) {
        let joint = |t: usize| cov0[t] && cov[t];
        let cross = |t: usize| {
            let (x, y) = (hi0[t], hi[t]);
            Complex64::new(x.re * y.re + x.im * y.im, x.im * y.re - x.re * y.im)
        };
        // Running sums: the correlation, both energies and the count over the
        // shared samples.
        let sums = (!inner.exact).then(|| {
            (
                cprefix(&|t| if joint(t) { cross(t) } else { Complex64::new(0.0, 0.0) }),
                prefix(&|t| if joint(t) { p2(hi0[t]) } else { 0.0 }),
                prefix(&|t| if joint(t) { p2(hi[t]) } else { 0.0 }),
                prefix(&|t| if joint(t) { 1.0 } else { 0.0 }),
            )
        });
        let pre = carrier_pre(hi, cov);
        for j in m..m + n {
            if !cov[j] {
                continue;
            }
            let (r, ea, eb, nj) = match &sums {
                Some((cs, sa, sb, sj)) => (cwsum_of(cs, j, half), wsum_of(sa, j, half), wsum_of(sb, j, half), wsum_of(sj, j, half)),
                None => {
                    let mut r = Complex64::new(0.0, 0.0);
                    let (mut ea, mut eb, mut nj) = (0.0, 0.0, 0.0);
                    for t in win(j) {
                        if joint(t) {
                            let v = cross(t);
                            r.re += v.re;
                            r.im += v.im;
                            ea += p2(hi0[t]);
                            eb += p2(hi[t]);
                            nj += 1.0;
                        }
                    }
                    (r, ea, eb, nj)
                }
            };
            let ((rl, eo, no), (rs, es, cs)) = carrier_sums(hi, cov, &pre, j);
            let qi = if no > 0.0 { carrier(rl, eo, no, QN_LOCK).1 } else { 0.0 };
            let s = if cs > 0.0 { carrier(rs, es, cs, QN_CARRIER).0 } else { 0.0 };
            let (rot, g) = if nj > 0.0 { lock(r, ea, eb, nj, q0[j - m], qi, h0[j - m]) } else { (Complex64::new(1.0, 0.0), 1.0) };
            let w = amp_weight(s, afl[i], wt[i]) * g;
            let d = Complex64::new(rot.re - 1.0, rot.im);
            acc[j - m] += (full[j] + (hi[j].re * d.re - hi[j].im * d.im)) * w;
            wsum[j - m] += w;
            if !vacc.is_empty() {
                vacc[j - m] += rotated(hi[j], rot) * w;
            }
        }
    }
    for (o, (a, w)) in out.iter_mut().zip(acc.iter().zip(&wsum)) {
        *o = if *w > 0.0 { (a / w) as f32 } else { 0.0 };
    }
    if let Some(v) = v {
        combined_band(v, &vacc, &wsum);
    }
    Ok(())
}

/// `--stack-weight fixed`: the video band phase-locked onto the primary's
/// over a LOCK_WIN window (running sums, normalized) and every input summed
/// with its fixed `1/sigma^2` weight. The rest of each input (full minus
/// video band) is summed as read.
fn combine_fixed(inner: &Inner, reads: &[(Vec<f64>, Vec<Complex64>, Vec<bool>)], m: usize, out: &mut [f32], v: Option<&mut [Complex64]>) {
    let n = out.len();
    let len = n + 2 * m;
    let half = LOCK_WIN / 2;
    let (full0, hi0, cov0) = &reads[0];
    let mut acc = vec![0.0f64; n];
    let mut wsum = vec![0.0f64; n];
    let mut vacc = vec![Complex64::new(0.0, 0.0); if v.is_some() { n } else { 0 }];
    for j in m..m + n {
        if cov0[j] {
            acc[j - m] += full0[j] * inner.weight[0];
            wsum[j - m] += inner.weight[0];
            if !vacc.is_empty() {
                vacc[j - m] += Complex64::new(hi0[j].re * inner.weight[0], hi0[j].im * inner.weight[0]);
            }
        }
    }
    for (i, (full, hi, cov)) in reads.iter().enumerate().skip(1) {
        if inner.exact {
            // The window sums directly, in ascending order (the OpenCL
            // kernel's), instead of running-sum differences whose rounding
            // depends on where the task starts.
            for j in m..m + n {
                if !cov[j] {
                    continue;
                }
                let mut r = Complex64::new(0.0, 0.0);
                for t in j.saturating_sub(half)..(j + half).min(len) {
                    if cov0[t] && cov[t] {
                        let (x, y) = (hi0[t], hi[t]);
                        r.re += x.re * y.re + x.im * y.im;
                        r.im += x.im * y.re - x.re * y.im;
                    }
                }
                let a = (r.re * r.re + r.im * r.im).sqrt();
                let rot = if a > 0.0 { Complex64::new(r.re / a, r.im / a) } else { Complex64::new(1.0, 0.0) };
                let d = Complex64::new(rot.re - 1.0, rot.im);
                acc[j - m] += (full[j] + (hi[j].re * d.re - hi[j].im * d.im)) * inner.weight[i];
                wsum[j - m] += inner.weight[i];
                if !vacc.is_empty() {
                    vacc[j - m] += rotated(hi[j], rot) * inner.weight[i];
                }
            }
            continue;
        }
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
            if !vacc.is_empty() {
                vacc[j - m] += rotated(hi[j], rot) * inner.weight[i];
            }
        }
    }
    for (o, (a, w)) in out.iter_mut().zip(acc.iter().zip(&wsum)) {
        *o = if *w > 0.0 { (a / w) as f32 } else { 0.0 };
    }
    if let Some(v) = v {
        combined_band(v, &vacc, &wsum);
    }
}

/// `z` turned by `rot`, as the OpenCL kernels compute it.
#[inline]
fn rotated(z: Complex64, rot: Complex64) -> Complex64 {
    Complex64::new(z.re * rot.re - z.im * rot.im, z.re * rot.im + z.im * rot.re)
}

/// The combined video band from its weighted sums (audio crosstalk cancellation).
fn combined_band(v: &mut [Complex64], vacc: &[Complex64], wsum: &[f64]) {
    for (o, (a, &w)) in v.iter_mut().zip(vacc.iter().zip(wsum)) {
        *o = if w > 0.0 { Complex64::new(a.re / w, a.im / w) } else { Complex64::new(0.0, 0.0) };
    }
}

impl Drop for VirtualSource {
    fn drop(&mut self) {
        self.live.stop();
        self.ahead.state.lock().unwrap().stop = true;
        self.ahead.cv.notify_all();
        // The producer may be inside a chunk waiting on pass 0, which the
        // stop above releases; it is not joined, the process is ending.
        drop(self.producer.take());
    }
}

impl SampleSource for VirtualSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let tw = std::time::Instant::now();
        let mut n = 0;
        while n < out.len() {
            let at = self.pos + n as u64;
            let c = at / CHUNK as u64;
            let Some(chunk) = self.ahead.chunk(c)? else { break };
            let from = (at - c * CHUNK as u64) as usize;
            if from >= chunk.len() {
                break;
            }
            let take = (chunk.len() - from).min(out.len() - n);
            out[n..n + take].copy_from_slice(&chunk[from..from + take]);
            n += take;
        }
        if std::env::var_os("LD_TIMING").is_some() {
            ld_decode::teeprintln!("STACKWAIT us={} n={}", tw.elapsed().as_micros(), n);
        }
        self.pos += n as u64;
        Ok(n)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        self.pos = sample;
        Ok(())
    }

    fn is_seekable(&self) -> bool {
        true
    }
}
