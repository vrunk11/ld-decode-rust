//! `pack`, `unpack` and `info`.
//!
//! Layout of a `.rfx` file (Matroska, DocType `rfx`):
//! - track `k + 1` (`D_RFX/PRED`): track `k`, one block per segment of
//!   `segment` samples, each block the range-coded residuals of
//!   [`crate::seg`];
//! - track `N + 1` (`D_RFX/META`): one block of JSON ([`Meta`]): names,
//!   sizes, SHA-256, predictor settings;
//! - track `N + 2` (`D_RFX/MAP`): one block per (track, earlier track)
//!   pair with the integer lag table ([`LagTable`]).
//!
//! Track `k` is predicted from its own past and from the mean of the
//! tracks of its `refs` (all earlier), read at the lag tables' positions in
//! the files themselves, so a segment needs nothing but its own block and
//! the earlier tracks: segments are coded and decoded in parallel.
use crate::align::{build_table, parse_pass0, run_pass0, LagTable, HYST_COARSE, HYST_FINE};
use crate::ebml;
use crate::gpu::GpuSeg;
use crate::fmt::Fmt;
use crate::io::{read_exact_at, read_words, write_all_at};
use crate::pred::Preset;
use crate::seg::{self, refwin_len};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const CODEC_PRED: &str = "D_RFX/PRED";
pub const CODEC_META: &str = "D_RFX/META";
pub const CODEC_MAP: &str = "D_RFX/MAP";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SrcMeta {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    /// Samples of the file (what `fmt` makes of its bytes); the bytes past the
    /// last whole group are in `tail`.
    pub words: u64,
    pub tail: Vec<u8>,
    /// Format of the original file (`s16`, `r16`, `s8`, `r8`, `lds`).
    #[serde(default = "default_fmt")]
    pub fmt: String,
    /// SHA-256 of the samples (as little-endian i32 on the 16-bit scale), to
    /// check a file extracted in another format.
    #[serde(default)]
    pub sample_sha256: String,
    /// Low bits that are zero in every word (the Duplicator's 10 bits in 16).
    pub shift: u32,
    pub mean: i32,
    pub own: usize,
    /// Own past samples when the track has no reference (0: same as `own`).
    #[serde(default)]
    pub solo: usize,
    pub taps: usize,
    /// References with a channel of their own (the best ones by weight), and
    /// the taps of each.
    #[serde(default)]
    pub ntop: usize,
    #[serde(default)]
    pub ttaps: usize,
    pub stride: usize,
    pub every: usize,
    pub lam_bits: u32,
    /// Earlier tracks this one is predicted from (those it shares frames with).
    pub refs: Vec<usize>,
    /// Weight of each reference in the mean they make (they add up to 1); empty
    /// means equal weights.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Where this capture stood on the command line (the archive may hold it
    /// in another place).
    #[serde(default)]
    pub position: usize,
    /// Noise against signal measured by [`crate::order`] (smaller is cleaner).
    #[serde(default)]
    pub noise: Option<f64>,
    /// Files that sit beside the capture (its `.yaml`), kept as they were.
    #[serde(default)]
    pub sidecars: Vec<Sidecar>,
    /// The compressed file the capture came from (`.ldf`, `.flac`), when it was
    /// decoded to raw samples: only the samples are kept.
    #[serde(default)]
    pub origin: Option<String>,
    /// Sample rate of the capture in Hz (from its `.yaml`, else the one given
    /// when it was packed); 0 when unknown.
    #[serde(default)]
    pub rate_hz: f64,
}

/// A small file stored whole in the metadata and given back by `unpack`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sidecar {
    pub name: String,
    pub sha256: String,
    pub data: Vec<u8>,
}

/// Largest sidecar kept.
const SIDECAR_MAX: u64 = 1 << 20;

/// The `.yaml` the Domesday Duplicator writes beside a capture
/// (`x.ddd.s16` -> `x.ddd.yaml`), if there is one.
pub(crate) fn find_sidecars(capture: &Path) -> Vec<Sidecar> {
    let p = capture.with_extension("yaml");
    match (std::fs::metadata(&p), p.file_name()) {
        (Ok(m), Some(name)) if m.is_file() && m.len() <= SIDECAR_MAX => match std::fs::read(&p) {
            Ok(data) => vec![Sidecar { name: name.to_string_lossy().into_owned(), sha256: hex(&Sha256::digest(&data)), data }],
            Err(_) => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn default_fmt() -> String {
    "s16".into()
}

impl SrcMeta {
    pub(crate) fn fmt(&self) -> Fmt {
        Fmt::parse(&self.fmt).unwrap_or(Fmt::S16)
    }

    pub(crate) fn preset(&self) -> Preset {
        let solo = if self.solo == 0 { self.own } else { self.solo };
        Preset { own: self.own, solo, plain: self.own, taps: self.taps, ntop: self.ntop, ttaps: self.ttaps, stride: self.stride, every: self.every, lam_bits: self.lam_bits }
    }

    /// The `ntop` references with their own channels, heaviest first (ties by
    /// place in `refs`).
    pub(crate) fn tops(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.refs.len()).collect();
        if self.weights.len() == self.refs.len() {
            idx.sort_by(|&a, &b| self.weights[b].partial_cmp(&self.weights[a]).unwrap_or(std::cmp::Ordering::Equal));
        }
        idx.into_iter().take(self.ntop).map(|i| self.refs[i]).collect()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Meta {
    pub format: String,
    pub version: u32,
    /// Samples per segment.
    pub segment: usize,
    pub sources: Vec<SrcMeta>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Gpu,
}

/// In which order the captures are coded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrderMode {
    /// The order on the command line.
    Given,
    /// Each group of captures of one programme from the cleanest to the
    /// noisiest ([`crate::order::plan`]).
    Auto,
}

pub struct PackOpts {
    pub order: OrderMode,
    pub device: Device,
    pub level: u32,
    pub segment: usize,
    pub verify: bool,
    /// Directory of pass 0 dumps (`pass0_<i>.txt`), if already made.
    pub pass0_dir: Option<PathBuf>,
    pub ld_decode: PathBuf,
    pub rate_mhz: f64,
    pub format: String,
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[derive(Clone)]
pub(crate) struct Stats {
    pub(crate) size: u64,
    pub(crate) sha: String,
    pub(crate) shift: u32,
    pub(crate) mean: i32,
    pub(crate) sample_sha: String,
}

/// Hash of samples as little-endian i32.
pub(crate) fn hash_samples(f: &File, fmt: Fmt, count: u64) -> Result<String> {
    let mut h = Sha256::new();
    let mut buf = vec![0i32; 1 << 22];
    let mut at = 0u64;
    while at < count {
        let n = ((count - at) as usize).min(buf.len());
        fmt.load(f, count, at as i64, &mut buf[..n])?;
        for v in &buf[..n] {
            h.update(v.to_le_bytes());
        }
        at += n as u64;
    }
    Ok(hex(&h.finalize()))
}

pub(crate) fn file_stats(path: &Path, fmt: Fmt) -> Result<Stats> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = f.metadata()?.len();
    let words = fmt.samples(size);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 24];
    let mut off = 0u64;
    while off < size {
        let n = ((size - off) as usize).min(buf.len());
        read_exact_at(&f, &mut buf[..n], off)?;
        hasher.update(&buf[..n]);
        off += n as u64;
    }
    let (mut or, mut sum) = (0u32, 0i64);
    let mut vals = vec![0i32; 1 << 22];
    let mut at = 0u64;
    while at < words {
        let n = ((words - at) as usize).min(vals.len());
        fmt.load(&f, words, at as i64, &mut vals[..n])?;
        for &v in &vals[..n] {
            or |= v as u32;
            sum += i64::from(v);
        }
        at += n as u64;
    }
    let shift = if or == 0 { 0 } else { or.trailing_zeros() };
    let mean = if words == 0 { 0 } else { ((sum >> shift) as f64 / words as f64).round() as i32 };
    let sample_sha = hash_samples(&f, fmt, words)?;
    Ok(Stats { size, sha: hex(&hasher.finalize()), shift, mean, sample_sha })
}

/// Track `j`'s files, as the predictor of a later track reads them.
/// Samples of a track by index, for reading references that are not files
/// (the tracks of an archive decoded as they are needed).
pub(crate) trait Store: Sync {
    /// Samples `[lo, lo + out.len())` of track `j`, on the scale of the original
    /// file (inside the track).
    fn load(&self, j: usize, lo: u64, out: &mut [i32]) -> Result<()>;
}

pub(crate) struct Refs<'a> {
    pub(crate) files: &'a [File],
    pub(crate) meta: &'a Meta,
    pub(crate) tables: &'a HashMap<(usize, usize), LagTable>,
    /// Where the reference tracks are read from when they are not in `files`.
    pub(crate) store: Option<&'a dyn Store>,
}

/// Runs `(from, to, lag)` of constant lag covering positions `[p0, p1)`.
fn runs(t: &LagTable, p0: i64, p1: i64) -> Vec<(i64, i64, i64)> {
    let mut out = Vec::new();
    let mut p = p0;
    if p < 0 {
        let e = p1.min(0);
        out.push((p, e, t.lag[0]));
        p = e;
    }
    while p < p1 {
        let i = t.pos.partition_point(|&q| q as i64 <= p).saturating_sub(1);
        let end = t.pos.get(i + 1).map_or(p1, |&q| (q as i64).min(p1));
        out.push((p, end, t.lag[i]));
        p = end;
    }
    out
}

impl Refs<'_> {
    /// Add `w` times the samples `[lo, hi)` of track `j` (shifted, minus its mean) to `dst`.
    fn add_range(&self, j: usize, lo: i64, hi: i64, w: f32, dst: &mut [f32]) -> Result<()> {
        let rj = &self.meta.sources[j];
        match self.store {
            None => crate::io::add_words_scaled(&self.files[j], rj.fmt(), rj.words, lo, hi, rj.shift, rj.mean, w, dst)?,
            Some(st) => {
                let mut buf = vec![0i32; dst.len()];
                st.load(j, lo as u64, &mut buf)?;
                for (o, &v) in dst.iter_mut().zip(&buf) {
                    let x = (v >> rj.shift) - rj.mean;
                    *o += w * x as f32;
                }
            }
        }
        Ok(())
    }

    /// The samples `[lo, hi)` of each reference track that the window of
    /// `[s0, s0 + n)` of track `k` reads.
    pub(crate) fn spans(&self, k: usize, s0: i64, n: usize) -> Vec<(usize, i64, i64)> {
        let sm = &self.meta.sources[k];
        let p0 = s0 - sm.taps as i64;
        let mut out = Vec::new();
        for &j in &sm.refs {
            let Some(t) = self.tables.get(&(k, j)) else { continue };
            for (a, b, lag) in runs(t, p0, p0 + (n + 2 * sm.taps) as i64) {
                let (lo, hi) = ((a + lag).max(0), (b + lag).min(self.meta.sources[j].words as i64));
                if hi > lo {
                    out.push((j, lo, hi));
                }
            }
        }
        out
    }

    /// The reference window over positions `[s0 - taps, s0 + n + taps)`, the
    /// channels one after the other.
    pub(crate) fn window(&self, k: usize, s0: i64, n: usize) -> Result<Vec<f32>> {
        let pre = self.meta.sources[k].preset();
        let mut out = vec![0f32; refwin_len(n, &pre)];
        self.window_into(k, s0, n, n + 2 * pre.taps, &mut out)?;
        Ok(out)
    }

    /// [`Refs::window`] into `out`, whose channels start `l` apart (`l` at
    /// least `n + 2 * taps`). Channel 0 is the weighted sum of the references,
    /// added in the order of `refs`, each term a product rounded and then
    /// added (no fused multiply-add), so every machine gets the same bits;
    /// channel `c` is the reference `tops()[c - 1]` alone.
    pub(crate) fn window_into(&self, k: usize, s0: i64, n: usize, l: usize, out: &mut [f32]) -> Result<()> {
        let sm = &self.meta.sources[k];
        let len = n + 2 * sm.taps;
        let p0 = s0 - sm.taps as i64;
        let tops = sm.tops();
        let out = &mut out[..(1 + tops.len()) * l];
        out.fill(0.0);
        let equal = 1.0f32 / sm.refs.len().max(1) as f32;
        let mut add = |j: usize, w: f32, dst: &mut [f32]| -> Result<()> {
            let rj = &self.meta.sources[j];
            let t = &self.tables[&(k, j)];
            for (a, b, lag) in runs(t, p0, p0 + len as i64) {
                let lo = (a + lag).max(0);
                let hi = (b + lag).min(rj.words as i64);
                if hi <= lo {
                    continue;
                }
                let base = (lo - lag - p0) as usize;
                self.add_range(j, lo, hi, w, &mut dst[base..base + (hi - lo) as usize])?;
            }
            Ok(())
        };
        for (idx, &j) in sm.refs.iter().enumerate() {
            let w = if sm.weights.len() == sm.refs.len() { sm.weights[idx] } else { equal };
            add(j, w, &mut out[..len])?;
        }
        for (c, &j) in tops.iter().enumerate() {
            add(j, 1.0, &mut out[(c + 1) * l..(c + 1) * l + len])?;
        }
        Ok(())
    }
}

/// Windows (of `CHOOSE_WINDOW` samples, spread through the target) on which
/// the references of a capture are compared.
const CHOOSE_WINDOWS: usize = 4;
const CHOOSE_WINDOW: usize = 131072;
/// A reference's weight falls by half for each `1 / CHOOSE_STEEP` bit its
/// cost alone is above the best one's.
const CHOOSE_STEEP: f64 = 8.0;
/// References weighing less than this (of the best's 1) are dropped, with
/// their lag tables.
const CHOOSE_FLOOR: f32 = 0.2;

/// The references of capture `k` and their weights. The mean of every
/// earlier capture treats a good reference (another capture of the same disc,
/// which shares the disc's own noise) like a poor one (another disc); each is
/// weighed by how well it predicts the capture alone.
fn choose_refs(refs: &Refs, k: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    let sm = &refs.meta.sources[k];
    if sm.refs.len() < 2 {
        return Ok((sm.refs.clone(), vec![1.0; sm.refs.len()]));
    }
    let pre = sm.preset();
    let win = CHOOSE_WINDOW.min(sm.words as usize);
    let starts: Vec<usize> = (0..CHOOSE_WINDOWS).map(|w| ((sm.words as usize - win) * (2 * w + 1)) / (2 * CHOOSE_WINDOWS)).collect();
    let work: Vec<(usize, usize)> = (0..sm.refs.len()).flat_map(|r| starts.iter().map(move |&s| (r, s))).collect();
    let costs: Vec<f64> = work
        .par_iter()
        .map(|&(r, s0)| -> Result<f64> {
            let y: Vec<i32> = read_words(&refs.files[k], sm.fmt(), sm.words, s0 as i64, win, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let w = refs.window_with(k, s0 as i64, win, &[(sm.refs[r], 1.0)])?;
            let preds = seg::predict_all(&y, &pre, true, &w);
            let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            Ok(seg::encode_residuals(&res).len() as f64 * 8.0 / win as f64)
        })
        .collect::<Result<_>>()?;
    let cost = |r: usize| costs[r * starts.len()..(r + 1) * starts.len()].iter().sum::<f64>() / starts.len() as f64;
    let best = (0..sm.refs.len()).map(cost).fold(f64::MAX, f64::min);
    let raw: Vec<f32> = (0..sm.refs.len()).map(|r| 2f64.powf(-CHOOSE_STEEP * (cost(r) - best)) as f32).collect();
    let keep: Vec<usize> = (0..raw.len()).filter(|&r| raw[r] >= CHOOSE_FLOOR).collect();
    let sum: f32 = keep.iter().map(|&r| raw[r]).sum();
    Ok((keep.iter().map(|&r| sm.refs[r]).collect(), keep.iter().map(|&r| raw[r] / sum).collect()))
}

pub struct SourceReport {
    pub name: String,
    pub size: u64,
    pub packed: u64,
    pub refs: usize,
}

/// What `pack` works out before it writes anything: the captures in coding
/// order, the metadata, the lag tables and the open files.
pub(crate) struct Prepared {
    pub(crate) inputs: Vec<PathBuf>,
    pub(crate) names: Vec<String>,
    pub(crate) meta: Meta,
    pub(crate) tables: HashMap<(usize, usize), LagTable>,
    pub(crate) files: Vec<File>,
    /// Raw files made from compressed captures, removed with this.
    pub(crate) work: Option<PathBuf>,
}

impl Drop for Prepared {
    fn drop(&mut self) {
        if let Some(w) = &self.work {
            let _ = std::fs::remove_dir_all(w);
        }
    }
}

/// How an input stands when an archive is edited: carried over from another
/// archive (`carry`: its metadata there), and whether its coding can be
/// copied as it is (`frozen`: every reference it was coded with is still
/// there, before it).
#[derive(Clone, Default)]
pub(crate) struct Spec {
    pub(crate) carry: Option<SrcMeta>,
    pub(crate) frozen: bool,
}

pub(crate) fn prepare(given: &[PathBuf], out: &Path, o: &PackOpts) -> Result<Prepared> {
    prepare_with(given, &vec![Spec::default(); given.len()], &HashMap::new(), out, o)
}

/// [`prepare`] for a list of captures some of which come from other archives:
/// the frozen ones keep their metadata and lag tables (`known`, by their
/// places in `given`), the others are worked out as usual against everything
/// before them.
pub(crate) fn prepare_with(given: &[PathBuf], specs: &[Spec], known: &HashMap<(usize, usize), LagTable>, out: &Path, o: &PackOpts) -> Result<Prepared> {
    if given.is_empty() {
        bail!("no input");
    }
    let edit = specs.iter().any(|s| s.carry.is_some());
    // Compressed captures (FLAC, Ogg-FLAC) are decoded to raw 16-bit files first:
    // the references are read at arbitrary positions.
    let (converted, work) = decode_compressed(given, out, o)?;
    let inputs: &[PathBuf] = &converted;
    let origs: Vec<PathBuf> = given.to_vec();
    let preset = Preset::level(o.level).with_context(|| format!("level {} is not 1 to 4", o.level))?;
    let mut names = Vec::new();
    for (p, sp) in inputs.iter().zip(specs) {
        let n = match &sp.carry {
            Some(c) => c.name.clone(),
            None => p.file_name().context("input has no file name")?.to_string_lossy().into_owned(),
        };
        if names.contains(&n) {
            bail!("two captures are named {n}: the archive keeps file names only");
        }
        names.push(n);
    }
    let fmts: Vec<Fmt> = inputs
        .iter()
        .zip(specs)
        .map(|(p, sp)| match &sp.carry {
            Some(c) => Ok(c.fmt()),
            None => input_fmt(p, &o.format),
        })
        .collect::<Result<_>>()?;
    if specs.iter().any(|s| s.carry.is_none()) {
        eprintln!("hashing and scanning {} files", specs.iter().filter(|s| s.carry.is_none()).count());
    }
    let stats: Vec<Stats> = inputs
        .par_iter()
        .zip(&fmts)
        .zip(specs)
        .map(|((p, f), sp)| match &sp.carry {
            Some(c) => Ok(Stats { size: c.size, sha: c.sha256.clone(), shift: c.shift, mean: c.mean, sample_sha: c.sample_sha256.clone() }),
            None => file_stats(p, *f),
        })
        .collect::<Result<_>>()?;

    // Where the lines of every capture are, from ld-decode's pass 0 (in the
    // order given), then the order to code them in.
    let need_tables = specs.iter().enumerate().any(|(k, s)| !s.frozen && k >= 1);
    let mut fields: Vec<Vec<crate::align::Field>> = Vec::new();
    if inputs.len() > 1 && need_tables {
        let dir = match &o.pass0_dir {
            Some(d) => d.clone(),
            None => {
                let d = out.with_extension("pass0");
                eprintln!("locating the lines of every capture (ld-decode pass 0)");
                if fmts.iter().any(|f| *f != fmts[0]) {
                    bail!("the captures are not all in one format: the pass 0 of ld-decode takes one");
                }
                run_pass0(&o.ld_decode, inputs, specs[0].carry.as_ref().map(|c| c.rate_hz).filter(|r| *r > 0.0).or_else(|| sidecar_rate_hz(&find_sidecars(&given[0]))).map_or(o.rate_mhz, |hz| hz / 1e6), fmts[0].name(), &d)?;
                d
            }
        };
        fields = (0..inputs.len()).map(|i| parse_pass0(&dir.join(format!("pass0_{i}.txt")))).collect::<Result<_>>()?;
    }
    let mut noise: Vec<Option<f64>> = specs.iter().map(|s| s.carry.as_ref().and_then(|c| c.noise)).collect();
    let mut ord: Vec<usize> = (0..inputs.len()).collect();
    if inputs.len() > 1 && o.order == OrderMode::Auto && !edit {
        let (order, scores) = crate::order::plan(inputs, &fmts, &fields);
        ord = order;
        noise = scores;
        if ord.iter().enumerate().any(|(k, &i)| k != i) {
            eprintln!("coding order, cleanest first (noise against signal, smaller is cleaner):");
            for (k, &i) in ord.iter().enumerate() {
                match noise[i] {
                    Some(c) => eprintln!("  {k:>2}: {} (was {i}, {c:.3})", names[i]),
                    None => eprintln!("  {k:>2}: {} (was {i})", names[i]),
                }
            }
        }
    }
    // From here on, everything is in coding order; `ord[k]` is the place on
    // the command line.
    let inputs: Vec<PathBuf> = ord.iter().map(|&i| inputs[i].clone()).collect();
    let origs: Vec<PathBuf> = ord.iter().map(|&i| origs[i].clone()).collect();
    let names: Vec<String> = ord.iter().map(|&i| names[i].clone()).collect();
    let stats: Vec<Stats> = ord.iter().map(|&i| stats[i].clone()).collect();
    let fmts: Vec<Fmt> = ord.iter().map(|&i| fmts[i]).collect();
    let noise: Vec<Option<f64>> = ord.iter().map(|&i| noise[i]).collect();
    let specs: Vec<Spec> = ord.iter().map(|&i| specs[i].clone()).collect();
    let mut tables: HashMap<(usize, usize), LagTable> = HashMap::new();
    let fields: Vec<Vec<crate::align::Field>> = {
        let mut slots: Vec<Option<Vec<crate::align::Field>>> = fields.into_iter().map(Some).collect();
        if slots.is_empty() {
            Vec::new()
        } else {
            ord.iter().map(|&i| slots[i].take().expect("each once")).collect()
        }
    };
    for k in 1..inputs.len() {
        for j in 0..k {
            if specs[k].frozen {
                if let Some(t) = known.get(&(k, j)) {
                    tables.insert((k, j), t.clone());
                }
                continue;
            }
            match build_table(&fields[k], &fields[j], fmts[k].samples(stats[k].size), HYST_FINE) {
                Some(t) => {
                    tables.insert((k, j), t);
                }
                None => eprintln!("{} and {} share no frame: not used as a reference", names[k], names[j]),
            }
        }
    }

    let sources: Vec<SrcMeta> = inputs
        .iter()
        .enumerate()
        .map(|(k, p)| {
            if let (true, Some(c)) = (specs[k].frozen, &specs[k].carry) {
                return Ok(SrcMeta { position: k, ..c.clone() });
            }
            let st = &stats[k];
            let covered = fmts[k].covered(st.size);
            let mut tail = vec![0u8; (st.size - covered) as usize];
            if !tail.is_empty() {
                read_exact_at(&File::open(p)?, &mut tail, covered)?;
            }
            let c = specs[k].carry.as_ref();
            let sidecars = match c {
                Some(c) => c.sidecars.clone(),
                None => find_sidecars(&origs[k]),
            };
            let rate_hz = c.map(|c| c.rate_hz).filter(|r| *r > 0.0).or_else(|| sidecar_rate_hz(&sidecars)).unwrap_or(o.rate_mhz * 1e6);
            Ok(SrcMeta {
                name: names[k].clone(),
                size: st.size,
                sha256: st.sha.clone(),
                words: fmts[k].samples(st.size),
                tail,
                fmt: fmts[k].name().to_string(),
                sample_sha256: st.sample_sha.clone(),
                shift: st.shift,
                mean: st.mean,
                own: preset.own,
                solo: preset.solo,
                taps: preset.taps,
                ntop: 0,
                ttaps: preset.ttaps,
                stride: preset.stride,
                every: preset.every,
                lam_bits: preset.lam_bits,
                refs: (0..k).filter(|j| tables.contains_key(&(k, *j))).collect(),
                weights: Vec::new(),
                position: if edit { k } else { ord[k] },
                noise: noise[k],
                sidecars,
                rate_hz,
                origin: match c {
                    Some(c) => c.origin.clone(),
                    None => (origs[k] != *p).then(|| origs[k].file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
                },
            })
        })
        .collect::<Result<_>>()?;
    let meta = Meta { format: "rfx".into(), version: 1, segment: o.segment, sources };
    let files: Vec<File> = inputs.iter().map(|p| File::open(p).map_err(Into::into)).collect::<Result<_>>()?;
    // Weigh the references of each capture by how well each predicts it.
    let mut meta = meta;
    if inputs.len() > 1 {
        let choice: Vec<Option<(Vec<usize>, Vec<f32>)>> = {
            let refs = Refs { files: &files, meta: &meta, tables: &tables, store: None };
            (0..inputs.len()).map(|k| if specs[k].frozen { Ok(None) } else { choose_refs(&refs, k).map(Some) }).collect::<Result<_>>()?
        };
        for (k, ch) in choice.into_iter().enumerate() {
            let Some((r, w)) = ch else { continue };
            for j in meta.sources[k].refs.clone() {
                if !r.contains(&j) {
                    tables.remove(&(k, j));
                }
            }
            if r.len() > 1 && std::env::var_os("RFX_TIMING").is_some() {
                let list: Vec<String> = r.iter().zip(&w).map(|(j, w)| format!("{}:{:.2}", j, w)).collect();
                eprintln!("  {k:>2} {}: {} references [{}]", names[k], r.len(), list.join(" "));
            }
            meta.sources[k].refs = r;
            meta.sources[k].weights = w;
        }
        // With three references or more, try giving the best ones channels of
        // their own (and a shorter own history) against the plain mean, and keep
        // what costs less on a few windows.
        for k in 0..inputs.len() {
            if specs[k].frozen {
                continue;
            }
            let nr = meta.sources[k].refs.len();
            meta.sources[k].own = preset.plain;
            meta.sources[k].ntop = 0;
            if nr < 3 {
                continue;
            }
            let plain = trial_cost(&Refs { files: &files, meta: &meta, tables: &tables, store: None }, k)?;
            // The channels take up the misalignment: try them with coarse tables.
            let fine: Vec<(usize, LagTable)> = meta.sources[k].refs.iter().filter_map(|&j| tables.get(&(k, j)).map(|t| (j, t.clone()))).collect();
            for &(j, _) in &fine {
                if let Some(t) = build_table(&fields[k], &fields[j], meta.sources[k].words, HYST_COARSE) {
                    tables.insert((k, j), t);
                }
            }
            meta.sources[k].own = preset.own;
            meta.sources[k].ntop = preset.ntop.min(nr);
            let split = trial_cost(&Refs { files: &files, meta: &meta, tables: &tables, store: None }, k)?;
            if std::env::var_os("RFX_TIMING").is_some() {
                eprintln!("  {k:>2} {}: plain mean {plain:.4}, with channels {split:.4} bits/sample", names[k]);
            }
            if split >= plain {
                meta.sources[k].own = preset.plain;
                meta.sources[k].ntop = 0;
                for (j, t) in fine {
                    tables.insert((k, j), t);
                }
            }
        }
    }
    Ok(Prepared { inputs, names, meta, tables, files, work })
}

/// The head of an archive: tracks, metadata and lag tables.
pub(crate) fn write_head(w: &mut impl Write, meta: &Meta, tables: &HashMap<(usize, usize), LagTable>) -> Result<()> {
    let n_src = meta.sources.len() as u64;
    w.write_all(&ebml::file_head())?;
    w.write_all(&ebml::info())?;
    let mut entries: Vec<Vec<u8>> = meta.sources.iter().enumerate().map(|(k, s)| ebml::track_entry(k as u64 + 1, CODEC_PRED, &s.name)).collect();
    entries.push(ebml::track_entry(n_src + 1, CODEC_META, "meta"));
    entries.push(ebml::track_entry(n_src + 2, CODEC_MAP, "map"));
    w.write_all(&ebml::tracks(&entries))?;
    w.write_all(&ebml::cluster(0, n_src + 1, &serde_json::to_vec(meta)?))?;
    let mut pairs: Vec<_> = tables.keys().copied().collect();
    pairs.sort_unstable();
    for (i, (k, j)) in pairs.iter().enumerate() {
        let mut b = Vec::new();
        crate::align::put_varint(&mut b, *k as u64);
        crate::align::put_varint(&mut b, *j as u64);
        b.extend(tables[&(*k, *j)].encode());
        w.write_all(&ebml::cluster(i as u64, n_src + 2, &b))?;
    }
    Ok(())
}

/// Copy the coded blocks of track `src` of `arch` as track `k` of `w`.
pub(crate) fn copy_blocks(w: &mut impl Write, arch: &Opened, src: usize, k: usize) -> Result<u64> {
    let mut order: Vec<(&u64, &(u64, u64))> = arch.blocks[src].iter().collect();
    order.sort();
    let mut total = 0;
    for (ts, (off, len)) in order {
        let mut bytes = vec![0u8; *len as usize];
        read_exact_at(&arch.file, &mut bytes, *off)?;
        total += len;
        w.write_all(&ebml::cluster(*ts, k as u64 + 1, &bytes))?;
    }
    Ok(total)
}

pub fn pack(inputs: &[PathBuf], out: &Path, o: &PackOpts) -> Result<Vec<SourceReport>> {
    let p = prepare(inputs, out, o)?;
    pack_prepared(&p, out, o, &vec![None; p.inputs.len()])
}

/// Write the archive for `p`; the tracks with a place in `copy` take their
/// coded blocks from there instead of being coded again.
pub(crate) fn pack_prepared(p: &Prepared, out: &Path, o: &PackOpts, copy: &[Option<(&Opened, usize)>]) -> Result<Vec<SourceReport>> {
    let Prepared { inputs, names: _, meta, tables, files, .. } = p;
    let refs = Refs { files, meta, tables, store: None };
    let mut w = std::io::BufWriter::new(File::create(out).with_context(|| format!("creating {}", out.display()))?);
    write_head(&mut w, meta, tables)?;

    let mut report = Vec::new();
    let s = o.segment;
    for k in 0..inputs.len() {
        let sm = &meta.sources[k];
        if let Some((arch, src)) = copy[k] {
            let packed = copy_blocks(&mut w, arch, src, k)?;
            eprintln!("{}: {} -> {} bytes copied", sm.name, sm.size, packed);
            report.push(SourceReport { name: sm.name.clone(), size: sm.size, packed, refs: sm.refs.len() });
            continue;
        }
        let pre = sm.preset();
        let use_ref = !sm.refs.is_empty();
        let nseg = (sm.words as usize).div_ceil(s);
        let t0 = std::time::Instant::now();
        let mut packed = 0u64;
        if o.device == Device::Gpu {
            packed = crate::gpupipe::pack_source(k, sm, &pre, s, o.verify, &files, &refs, &mut w)?;
        }
        let chunk = rayon::current_num_threads() * 4;
        let nseg_cpu = if o.device == Device::Gpu { 0 } else { nseg };
        for first in (0..nseg_cpu).step_by(chunk) {
            let segs: Vec<usize> = (first..(first + chunk).min(nseg)).collect();
            let blocks: Vec<Result<Vec<u8>>> = segs
                .par_iter()
                .map(|&g| {
                    let s0 = g * s;
                    let n = s.min(sm.words as usize - s0);
                    let y: Vec<i32> = read_words(&files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
                    let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
                    let bytes = seg::encode(&y, &pre, use_ref, &win);
                    if o.verify {
                        let back = seg::decode(&bytes, n, &pre, use_ref, &win);
                        if back.as_deref() != Some(&y[..]) {
                            bail!("self-check failed: segment {g} of {} does not decode to itself", sm.name);
                        }
                    }
                    Ok(bytes)
                })
                .collect();
            for (g, b) in segs.iter().zip(blocks) {
                let b = b?;
                packed += b.len() as u64;
                w.write_all(&ebml::cluster(*g as u64, k as u64 + 1, &b))?;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "{}: {} -> {} bytes ({:.3} bits/sample, {:.1}%), {} reference(s), {:.1} Msmp/s",
            sm.name,
            sm.size,
            packed,
            packed as f64 * 8.0 / sm.words.max(1) as f64,
            100.0 * packed as f64 / sm.size.max(1) as f64,
            sm.refs.len(),
            sm.words as f64 / secs / 1e6
        );
        report.push(SourceReport { name: sm.name.clone(), size: sm.size, packed, refs: sm.refs.len() });
    }
    w.flush()?;
    Ok(report)
}

pub(crate) struct Opened {
    pub(crate) meta: Meta,
    pub(crate) tables: HashMap<(usize, usize), LagTable>,
    pub(crate) blocks: Vec<HashMap<u64, (u64, u64)>>,
    pub(crate) file: File,
}

pub(crate) fn open(path: &Path) -> Result<Opened> {
    if crate::flacmode::is_flac_archive(path) {
        bail!("{} holds FLAC tracks (packed with --flac): it is a Matroska audio file, and rfx reads it only to unpack, info, or extract the metadata", path.display());
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let idx = ebml::scan(&file)?;
    let track_of = |codec: &str| idx.codecs.iter().find(|(_, c)| c == codec).map(|(n, _)| *n);
    let meta_track = track_of(CODEC_META).context("no metadata track")?;
    let map_track = track_of(CODEC_MAP).context("no map track")?;
    let mb = idx.blocks.iter().find(|b| b.track == meta_track).context("no metadata block")?;
    let mut buf = vec![0u8; mb.len as usize];
    read_exact_at(&file, &mut buf, mb.off)?;
    let meta: Meta = serde_json::from_slice(&buf).context("bad metadata")?;
    if meta.format != "rfx" || meta.version != 1 {
        bail!("unsupported rfx version {}", meta.version);
    }
    let mut tables = HashMap::new();
    for b in idx.blocks.iter().filter(|b| b.track == map_track) {
        let mut buf = vec![0u8; b.len as usize];
        read_exact_at(&file, &mut buf, b.off)?;
        let mut at = 0;
        let k = crate::align::get_varint(&buf, &mut at)? as usize;
        let j = crate::align::get_varint(&buf, &mut at)? as usize;
        tables.insert((k, j), LagTable::decode(&buf[at..])?);
    }
    let mut blocks = vec![HashMap::new(); meta.sources.len()];
    for b in &idx.blocks {
        if b.track >= 1 && (b.track as usize) <= meta.sources.len() {
            blocks[b.track as usize - 1].insert(b.ts, (b.off, b.len));
        }
    }
    Ok(Opened { meta, tables, blocks, file })
}

pub fn info(path: &Path) -> Result<()> {
    let o = open(path)?;
    println!("{}: {} tracks, segments of {} samples", path.display(), o.meta.sources.len(), o.meta.segment);
    for (k, sm) in o.meta.sources.iter().enumerate() {
        let packed: u64 = o.blocks[k].values().map(|b| b.1).sum();
        println!(
            "  {k}: {}  {} bytes -> {} ({:.3} bits/sample)  shift {} mean {}  refs {:?}  predictor own {} taps {} stride {} every {}",
            sm.name,
            sm.size,
            packed,
            packed as f64 * 8.0 / sm.words.max(1) as f64,
            sm.shift,
            sm.mean,
            sm.refs,
            sm.own,
            sm.taps,
            sm.stride,
            sm.every
        );
        println!("     sha256 {}", sm.sha256);
        let noise = sm.noise.map(|c| format!("  noise {c:.3}")).unwrap_or_default();
        let side: Vec<&str> = sm.sidecars.iter().map(|s| s.name.as_str()).collect();
        println!("     on the command line: {}{}  sidecars {:?}", sm.position, noise, side);
    }
    Ok(())
}

/// Rebuild the original files in `outdir`, each checked against its SHA-256.
/// With `only`, just that track (the earlier ones are decoded too, as its
/// references, and removed).
/// What `unpack` writes when it is not the capture as it came.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutFmt {
    Raw(Fmt),
    /// FLAC (`.flac`) or Ogg-FLAC (`.ldf`), made by ffmpeg like ld-decode's own.
    Flac,
    Ldf,
}

impl OutFmt {
    pub fn parse(s: &str) -> Option<OutFmt> {
        match s {
            "flac" => Some(OutFmt::Flac),
            "ldf" => Some(OutFmt::Ldf),
            other => Fmt::parse(other).map(OutFmt::Raw),
        }
    }
}

pub fn unpack(path: &Path, outdir: &Path, only: Option<usize>, device: Device, as_fmt: Option<OutFmt>) -> Result<()> {
    let o = open(path)?;
    let meta = &o.meta;
    std::fs::create_dir_all(outdir)?;
    let last = only.unwrap_or(meta.sources.len() - 1);
    if last >= meta.sources.len() {
        bail!("the archive has {} tracks", meta.sources.len());
    }
    let mut outs: Vec<File> = Vec::new();
    for sm in &meta.sources[..=last] {
        let name = Path::new(&sm.name).file_name().context("bad name in the archive")?;
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(outdir.join(name))?;
        f.set_len(sm.size)?;
        outs.push(f);
    }
    let refs = Refs { files: &outs, meta, tables: &o.tables, store: None };
    let s = meta.segment;
    for k in 0..=last {
        let sm = &meta.sources[k];
        let pre = sm.preset();
        let use_ref = !sm.refs.is_empty();
        let nseg = (sm.words as usize).div_ceil(s);
        let t0 = std::time::Instant::now();
        if device == Device::Gpu {
            crate::gpupipe::unpack_source(k, sm, &pre, s, &o.blocks[k], &o.file, &refs, &outs[k], sm.fmt())?;
        }
        let chunk = rayon::current_num_threads() * 4;
        let nseg_cpu = if device == Device::Gpu { 0 } else { nseg };
        for first in (0..nseg_cpu).step_by(chunk) {
            let segs: Vec<usize> = (first..(first + chunk).min(nseg)).collect();
            segs.par_iter().try_for_each(|&g| -> Result<()> {
                let s0 = g * s;
                let n = s.min(sm.words as usize - s0);
                let &(off, len) = o.blocks[k].get(&(g as u64)).with_context(|| format!("{}: segment {g} is missing", sm.name))?;
                let mut bytes = vec![0u8; len as usize];
                read_exact_at(&o.file, &mut bytes, off)?;
                let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
                let y = seg::decode(&bytes, n, &pre, use_ref, &win).with_context(|| format!("{}: segment {g} is corrupt", sm.name))?;
                let vals: Vec<i32> = y.iter().map(|&v| (v + sm.mean) << sm.shift).collect();
                sm.fmt().store(&outs[k], s0 as u64, &vals)?;
                Ok(())
            })?;
        }
        if !sm.tail.is_empty() {
            write_all_at(&outs[k], &sm.tail, sm.fmt().covered(sm.size))?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let got = {
            let f = &outs[k];
            let mut h = Sha256::new();
            let mut buf = vec![0u8; 1 << 24];
            let mut off = 0u64;
            while off < sm.size {
                let n = ((sm.size - off) as usize).min(buf.len());
                read_exact_at(f, &mut buf[..n], off)?;
                h.update(&buf[..n]);
                off += n as u64;
            }
            hex(&h.finalize())
        };
        if got != sm.sha256 {
            bail!("{}: SHA-256 differs from the original ({got} against {})", sm.name, sm.sha256);
        }
        eprintln!("{}: {} bytes, SHA-256 matches, {:.1} Msmp/s", sm.name, sm.size, sm.words as f64 / secs / 1e6);
        if only.is_none() || only == Some(k) {
            for sc in &sm.sidecars {
                let got = hex(&Sha256::digest(&sc.data));
                if got != sc.sha256 {
                    bail!("{}: the stored copy of {} is damaged", sm.name, sc.name);
                }
                let name = Path::new(&sc.name).file_name().context("bad sidecar name in the archive")?;
                std::fs::write(outdir.join(name), &sc.data)?;
            }
        }
    }
    if let Some(keep) = only {
        drop(outs);
        for (k, sm) in meta.sources[..=last].iter().enumerate() {
            if k != keep {
                let _ = std::fs::remove_file(outdir.join(&sm.name));
            }
        }
    } else {
        drop(outs);
    }
    // Another format asked for: the files above are the captures as they were
    // given; each is converted sample by sample and checked by the hash of its
    // samples.
    if let Some(target) = as_fmt {
        for (k, sm) in meta.sources[..=last].iter().enumerate() {
            if only.is_some_and(|keep| keep != k) || target == OutFmt::Raw(sm.fmt()) {
                continue;
            }
            match target {
                OutFmt::Raw(t) => convert_file(&outdir.join(&sm.name), sm, t, outdir)?,
                OutFmt::Flac | OutFmt::Ldf => encode_flac(&outdir.join(&sm.name), sm, target == OutFmt::Ldf, outdir)?,
            }
        }
    }
    Ok(())
}

/// Write the samples of `src` (in the format of `sm`) as a `target` file in
/// `outdir`, named with the extension of the new format; remove `src`.
pub(crate) fn convert_file(src: &Path, sm: &SrcMeta, target: Fmt, outdir: &Path) -> Result<()> {
    let name = Path::new(&sm.name).with_extension(target.name());
    let dst_path = outdir.join(name.file_name().context("bad name")?);
    let from = File::open(src)?;
    let dst = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&dst_path)?;
    let mut buf = vec![0i32; 1 << 22];
    let mut at = 0u64;
    while at < sm.words {
        let n = ((sm.words - at) as usize).min(buf.len());
        sm.fmt().load(&from, sm.words, at as i64, &mut buf[..n])?;
        target.store(&dst, at, &buf[..n]).with_context(|| format!("{}: converting to {}", sm.name, target.name()))?;
        at += n as u64;
    }
    drop(from);
    let got = hash_samples(&dst, target, sm.words)?;
    if got != sm.sample_sha256 {
        bail!("{}: the samples differ after conversion to {}", sm.name, target.name());
    }
    std::fs::remove_file(src)?;
    eprintln!("{} -> {}: samples identical", sm.name, dst_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    Ok(())
}

/// Development check: the predictions of the first `nseg` segments of a file,
/// CPU against GPU, and the first sample where they differ.
pub fn gpu_check(path: &Path, level: u32, nseg: usize, segment: usize, with_ref: bool) -> Result<()> {
    let pre = Preset::level(level).context("level 1 to 4")?;
    let st = file_stats(path, Fmt::S16)?;
    let f = File::open(path)?;
    let words = st.size / 2;
    let mut input = vec![0i32; nseg * segment];
    let mut lens = Vec::new();
    let mut cpu = Vec::new();
    let l = segment + 2 * pre.taps;
    let rw = pre.chans() * l;
    let mut refw = if with_ref { vec![0f32; nseg * rw] } else { Vec::new() };
    for g in 0..nseg {
        let n = segment.min((words as usize).saturating_sub(g * segment));
        let y: Vec<i32> = read_words(&f, Fmt::S16, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect();
        input[g * segment..g * segment + n].copy_from_slice(&y);
        lens.push(n as i32);
        let ln = n + 2 * pre.taps;
        let val = |c: usize, i: usize| y[(i + 5 + 3 * c).min(n - 1)] as f32;
        let win: Vec<f32> = if with_ref { (0..pre.chans()).flat_map(|c| (0..ln).map(move |i| (c, i))).map(|(c, i)| val(c, i)).collect() } else { Vec::new() };
        if with_ref {
            for c in 0..pre.chans() {
                for i in 0..ln {
                    refw[g * rw + c * l + i] = val(c, i);
                }
            }
        }
        cpu.push(seg::predict_all(&y, &pre, with_ref, &win));
    }
    let mut gpu = GpuSeg::new(&pre, with_ref, segment)?;
    let _ = gpu.run(0, &input, &refw, &lens)?;
    GpuSeg::reset_times();
    let out = gpu.run(0, &input, &refw, &lens)?;
    let total = lens.iter().sum::<i32>() as f64;
    let t = GpuSeg::times();
    eprintln!("GPU: kernel {:.1} Msmp/s ({:.3} s), upload {:.3} s, download {:.3} s", total / t[1] / 1e6, t[1], t[0], t[2]);
    for g in 0..nseg {
        let n = lens[g] as usize;
        let diff = (0..n).filter(|&i| cpu[g][i] != out[g * segment + i]).collect::<Vec<_>>();
        match diff.first() {
            None => println!("segment {g}: identical ({n} samples)"),
            Some(&i) => println!(
                "segment {g}: {} of {n} samples differ, first at {i} (cpu {} gpu {}) ; solve due at {}",
                diff.len(),
                cpu[g][i],
                out[g * segment + i],
                (i / pre.every) * pre.every + pre.every - 1
            ),
        }
    }
    Ok(())
}

/// Development: single-thread cost of each stage on the first `nseg` segments.
pub fn bench(path: &Path, level: u32, nseg: usize, segment: usize) -> Result<()> {
    let pre = Preset::level(level).context("level 1 to 4")?;
    let st = file_stats(path, Fmt::S16)?;
    let f = File::open(path)?;
    let words = st.size / 2;
    let time = |what: &str, n: usize, t0: std::time::Instant| {
        println!("{what:<28} {:>7.1} Msmp/s", n as f64 / t0.elapsed().as_secs_f64() / 1e6);
    };
    for use_ref in [false, true] {
        let (mut tp, mut te, mut td, mut tr) = (0f64, 0f64, 0f64, 0f64);
        let mut total = 0usize;
        for g in 0..nseg {
            let n = segment.min((words as usize).saturating_sub(g * segment));
            let y: Vec<i32> = read_words(&f, Fmt::S16, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect();
            let win: Vec<f32> = (0..refwin_len(n, &pre)).map(|i| y[(i + 5).min(n - 1)] as f32).collect();
            let t = std::time::Instant::now();
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            tp += t.elapsed().as_secs_f64();
            let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            let t = std::time::Instant::now();
            let bytes = seg::encode_residuals(&res);
            te += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let back = seg::decode_residuals(&bytes, n).context("decode")?;
            td += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let ys = seg::reconstruct(&back, &pre, use_ref, &win);
            tr += t.elapsed().as_secs_f64();
            assert_eq!(ys, y);
            total += n;
        }
        println!("-- {} reference", if use_ref { "with" } else { "without" });
        for (what, t) in [("predict (coding)", tp), ("entropy encode", te), ("entropy decode", td), ("reconstruct (decoding)", tr)] {
            println!("{what:<28} {:>7.1} Msmp/s", total as f64 / t / 1e6);
        }
        let _ = time;
    }
    Ok(())
}

/// Development: where the predictor's time goes (single thread, with reference).
pub fn bench_pred(path: &Path, level: u32, nseg: usize, segment: usize) -> Result<()> {
    let pre = Preset::level(level).context("level 1 to 4")?;
    let st = file_stats(path, Fmt::S16)?;
    let f = File::open(path)?;
    let words = st.size / 2;
    let n = segment;
    let ys: Vec<Vec<i32>> = (0..nseg)
        .map(|g| Ok(read_words(&f, Fmt::S16, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect()))
        .collect::<Result<_>>()?;
    let win: Vec<f32> = (0..refwin_len(n, &pre)).map(|i| ys[0][(i + 5).min(n - 1)] as f32).collect();
    for (name, p) in [
        ("full", pre),
        ("no solve", Preset { every: 1 << 30, ..pre }),
        ("no solve, no update", Preset { every: 1 << 30, stride: 1 << 30, ..pre }),
    ] {
        let t = std::time::Instant::now();
        for y in &ys {
            std::hint::black_box(seg::predict_all(y, &p, true, &win));
        }
        let dt = t.elapsed().as_secs_f64();
        println!("{name:<22} {:>7.1} Msmp/s  ({:.0} ns/sample)", (nseg * n) as f64 / dt / 1e6, dt * 1e9 / (nseg * n) as f64);
    }
    Ok(())
}

impl Refs<'_> {
    /// Like [`Refs::window_into`] over the references `(source, weight)` given
    /// (an experiment: any subset, any weights).
    pub(crate) fn window_with(&self, k: usize, s0: i64, n: usize, which: &[(usize, f32)]) -> Result<Vec<f32>> {
        let sm = &self.meta.sources[k];
        let len = n + 2 * sm.taps;
        let p0 = s0 - sm.taps as i64;
        let mut out = vec![0f32; len];
        for &(j, w) in which {
            let rj = &self.meta.sources[j];
            let t = &self.tables[&(k, j)];
            for (a, b, lag) in runs(t, p0, p0 + len as i64) {
                let lo = (a + lag).max(0);
                let hi = (b + lag).min(rj.words as i64);
                if hi <= lo {
                    continue;
                }
                let base = (lo - lag - p0) as usize;
                self.add_range(j, lo, hi, w, &mut out[base..base + (hi - lo) as usize])?;
            }
        }
        Ok(out)
    }
}

/// Development: what would a different choice of references have cost, for
/// the target `k`, on `nseg` segments spread through the capture? Prints the
/// cost of the mean of all its references (what `pack` does), of each one
/// alone, of the best few together, and the best choice segment by segment.
pub fn refsel(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("refsel.rfx"), o)?;
    let Prepared { names, meta, tables, files, .. } = &p;
    let refs = Refs { files, meta, tables, store: None };
    let sm = &meta.sources[k];
    let pre = Preset { ntop: 0, ..sm.preset() };
    let s = meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    if sm.refs.len() < 2 {
        bail!("{} has {} reference(s): nothing to choose from", sm.name, sm.refs.len());
    }
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let cost = |y: &[i32], win: &[f32]| -> f64 {
        let preds = seg::predict_all(y, &pre, true, win);
        let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
        seg::encode_residuals(&res).len() as f64 * 8.0 / y.len() as f64
    };
    // [segment][config] costs; configs: all, then each reference, then top-m.
    let tops = [2usize, 3, 5, 8];
    let rows: Vec<(Vec<f64>, Vec<f64>)> = segs
        .par_iter()
        .map(|&q| -> Result<(Vec<f64>, Vec<f64>)> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let nr = sm.refs.len();
            let all = refs.window_with(k, s0 as i64, n, &sm.refs.iter().map(|&j| (j, 1.0 / nr as f32)).collect::<Vec<_>>())?;
            let mut costs = vec![cost(&y, &all)];
            let mut single = Vec::new();
            for &j in &sm.refs {
                let w = refs.window_with(k, s0 as i64, n, &[(j, 1.0)])?;
                single.push(cost(&y, &w));
            }
            costs.extend(&single);
            let mut rank: Vec<usize> = (0..nr).collect();
            rank.sort_by(|&a, &b| single[a].partial_cmp(&single[b]).unwrap());
            for &m in &tops {
                let m = m.min(nr);
                let which: Vec<(usize, f32)> = rank[..m].iter().map(|&r| (sm.refs[r], 1.0 / m as f32)).collect();
                costs.push(cost(&y, &refs.window_with(k, s0 as i64, n, &which)?));
            }
for e in [2.0f32, 4.0, 8.0] {                let best = single.iter().cloned().fold(f64::MAX, f64::min);                let raw: Vec<f32> = single.iter().map(|c| 2f32.powf(-e * (*c - best) as f32)).collect();                let sum: f32 = raw.iter().sum();                let which: Vec<(usize, f32)> = sm.refs.iter().zip(&raw).map(|(&j, w)| (j, w / sum)).collect();                costs.push(cost(&y, &refs.window_with(k, s0 as i64, n, &which)?));            }
            Ok((costs, single))
        })
        .collect::<Result<_>>()?;
    let nc = rows[0].0.len();
    let mean = |c: usize| rows.iter().map(|r| r.0[c]).sum::<f64>() / rows.len() as f64;
    println!("target {} ({} references), {} segments of {} samples", sm.name, sm.refs.len(), segs.len(), s);
    println!("  mean of all references (current)      {:.4} bits/sample", mean(0));
    for (i, &j) in sm.refs.iter().enumerate() {
        println!("  reference {:<28} alone {:.4}", names[j], mean(1 + i));
    }
    let nr = sm.refs.len();
    for (i, &m) in tops.iter().enumerate() {
        println!("  best {:<2} of them, chosen per segment   {:.4}", m.min(nr), mean(1 + nr + i));
    }
    for (i, e) in [2.0, 4.0, 8.0].iter().enumerate() {
        println!("  all of them, weights 2^(-{e} x excess cost)   {:.4}", mean(1 + nr + tops.len() + i));
    }
    let best_fixed = (0..nc).map(|c| (c, mean(c))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let oracle = rows.iter().map(|r| r.0.iter().cloned().fold(f64::MAX, f64::min)).sum::<f64>() / rows.len() as f64;
    println!("  best single configuration for all segments  {:.4} (config {})", best_fixed.1, best_fixed.0);
    println!("  best configuration segment by segment       {:.4}  ({:+.2}% against the mean of all)", oracle, 100.0 * (oracle / mean(0) - 1.0));
    Ok(())
}

/// Cost, in bits per sample, of coding the trial windows of capture `k` with
/// the predictor its metadata describes.
fn trial_cost(refs: &Refs, k: usize) -> Result<f64> {
    let sm = &refs.meta.sources[k];
    let pre = sm.preset();
    let win = CHOOSE_WINDOW.min(sm.words as usize);
    let starts: Vec<usize> = (0..CHOOSE_WINDOWS).map(|w| ((sm.words as usize - win) * (2 * w + 1)) / (2 * CHOOSE_WINDOWS)).collect();
    let costs: Vec<f64> = starts
        .par_iter()
        .map(|&s0| -> Result<f64> {
            let y: Vec<i32> = read_words(&refs.files[k], sm.fmt(), sm.words, s0 as i64, win, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let w = refs.window(k, s0 as i64, win)?;
            let preds = seg::predict_all(&y, &pre, true, &w);
            let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            Ok(seg::encode_residuals(&res).len() as f64 * 8.0 / win as f64)
        })
        .collect::<Result<_>>()?;
    Ok(costs.iter().sum::<f64>() / costs.len() as f64)
}

/// The format of an input: `format` (`auto` guesses from the extension).
pub(crate) fn input_fmt(path: &Path, format: &str) -> Result<Fmt> {
    if matches!(format, "ldf" | "flac") {
        return Ok(Fmt::S16);
    }
    if format == "auto" {
        Fmt::from_path(path).with_context(|| format!("cannot tell the format of {} from its extension: use --format", path.display()))
    } else {
        Fmt::parse(format).with_context(|| format!("unknown format {format}"))
    }
}

/// Whether `path` is a compressed capture to decode first.
fn is_compressed(path: &Path, format: &str) -> bool {
    matches!(format, "ldf" | "flac")
        || (format == "auto" && path.extension().is_some_and(|e| matches!(e.to_string_lossy().to_ascii_lowercase().as_str(), "ldf" | "flac")))
}

/// Decode the FLAC and Ogg-FLAC inputs with ffmpeg into raw 16-bit files in a
/// work directory beside `out`; the other inputs go through as they are.
/// Returns the list to pack and the work directory, if one was made.
pub(crate) fn decode_compressed(given: &[PathBuf], out: &Path, o: &PackOpts) -> Result<(Vec<PathBuf>, Option<PathBuf>)> {
    if !given.iter().any(|p| is_compressed(p, &o.format)) {
        return Ok((given.to_vec(), None));
    }
    let work = out.with_extension("work");
    std::fs::create_dir_all(&work)?;
    let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let mut list = Vec::new();
    for p in given {
        if !is_compressed(p, &o.format) {
            list.push(p.clone());
            continue;
        }
        let stem = p.file_stem().context("input has no file name")?;
        let dst = work.join(format!("{}.s16", stem.to_string_lossy()));
        eprintln!("decoding {} to raw 16-bit samples", p.display());
        let status = std::process::Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-i"])
            .arg(p)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&dst)
            .status()
            .with_context(|| format!("starting {ffmpeg} (needed for FLAC and .ldf inputs)"))?;
        if !status.success() {
            let _ = std::fs::remove_dir_all(&work);
            bail!("ffmpeg could not decode {}", p.display());
        }
        list.push(dst);
    }
    Ok((list, Some(work)))
}

/// Write the capture `src` as FLAC (or Ogg-FLAC, `.ldf`) with ffmpeg, the way
/// ld-decode's tools do (mono, 16 bits, the rate field `40k`), and check that
/// decoding it gives back the samples; remove `src`.
pub(crate) fn encode_flac(src: &Path, sm: &SrcMeta, ogg: bool, outdir: &Path) -> Result<()> {
    let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    // ffmpeg reads 16-bit samples: an .lds capture goes through a raw copy.
    let raw = if sm.fmt() == Fmt::S16 {
        src.to_path_buf()
    } else {
        let tmp = outdir.join(format!("{}.rfx-tmp.s16", sm.name));
        let from = File::open(src)?;
        let dst = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&tmp)?;
        let mut buf = vec![0i32; 1 << 22];
        let mut at = 0u64;
        while at < sm.words {
            let n = ((sm.words - at) as usize).min(buf.len());
            sm.fmt().load(&from, sm.words, at as i64, &mut buf[..n])?;
            Fmt::S16.store(&dst, at, &buf[..n]).with_context(|| format!("{}: samples do not fit 16 bits", sm.name))?;
            at += n as u64;
        }
        tmp
    };
    let stem = Path::new(&sm.name).with_extension("");
    let dst_path = outdir.join(format!("{}.{}", stem.to_string_lossy(), if ogg { "ldf" } else { "flac" }));
    let mut cmd = std::process::Command::new(&ffmpeg);
    // The container rate is the RF rate in kHz (75 MHz -> 75000 Hz), as ld-decode reads it.
    let rate = if sm.rate_hz > 0.0 { format!("{}", (sm.rate_hz / 1000.0).round() as u64) } else { "40k".to_string() };
    cmd.args(["-v", "error", "-y", "-f", "s16le", "-ar", &rate, "-ac", "1", "-i"]).arg(&raw).args(["-c:a", "flac"]);
    if ogg {
        cmd.args(["-f", "ogg"]);
    }
    let status = cmd.arg(&dst_path).status().with_context(|| format!("starting {ffmpeg}"))?;
    if !status.success() {
        bail!("ffmpeg could not encode {}", sm.name);
    }
    // Decode it back and compare.
    let back = outdir.join(format!("{}.rfx-check.s16", sm.name));
    let status = std::process::Command::new(&ffmpeg).args(["-v", "error", "-y", "-i"]).arg(&dst_path).args(["-f", "s16le", "-acodec", "pcm_s16le"]).arg(&back).status()?;
    let ok = status.success() && hash_samples(&File::open(&back)?, Fmt::S16, sm.words)? == sm.sample_sha256;
    let _ = std::fs::remove_file(&back);
    if raw != src {
        let _ = std::fs::remove_file(&raw);
    }
    if !ok {
        bail!("{}: the samples differ after encoding with ffmpeg", sm.name);
    }
    std::fs::remove_file(src)?;
    eprintln!("{} -> {}: samples identical", sm.name, dst_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    Ok(())
}

/// The sample rate (Hz) a capture's `.yaml` gives, if one of the sidecars does.
pub(crate) fn sidecar_rate_hz(sidecars: &[Sidecar]) -> Option<f64> {
    sidecars.iter().find_map(|s| {
        std::str::from_utf8(&s.data).ok()?.lines().find_map(|l| l.trim().strip_prefix("\"sample_rate_hz\":")?.trim().trim_end_matches(',').parse().ok())
    })
}

/// The whole metadata of an archive as YAML: the archive's own fields, every
/// track's (its predictor, references and weights, format, hashes, the rate,
/// the bytes past the last whole sample as hex), the `.yaml` sidecars as text
/// and the size each track takes in the archive.
pub fn meta_yaml(path: &Path) -> Result<String> {
    let o = open(path)?;
    let mut tracks = Vec::new();
    for (k, sm) in o.meta.sources.iter().enumerate() {
        let mut v = serde_json::to_value(sm)?;
        let packed: u64 = o.blocks[k].values().map(|b| b.1).sum();
        if let Some(m) = v.as_object_mut() {
            m.insert("track".into(), k.into());
            m.insert("packed_bytes".into(), packed.into());
            m.insert("bits_per_sample".into(), (packed as f64 * 8.0 / sm.words.max(1) as f64).into());
            m.insert("tail".into(), hex(&sm.tail).into());
            let side: Vec<serde_json::Value> = sm
                .sidecars
                .iter()
                .map(|s| {
                    let mut e = serde_json::json!({ "name": s.name, "sha256": s.sha256, "bytes": s.data.len() });
                    if let Ok(t) = std::str::from_utf8(&s.data) {
                        e["text"] = t.into();
                    }
                    e
                })
                .collect();
            m.insert("sidecars".into(), side.into());
            let tables: Vec<serde_json::Value> = sm
                .refs
                .iter()
                .filter_map(|&j| o.tables.get(&(k, j)).map(|t| serde_json::json!({ "reference": j, "steps": t.pos.len() })))
                .collect();
            m.insert("lag_tables".into(), tables.into());
        }
        tracks.push(v);
    }
    let doc = serde_json::json!({
        "format": o.meta.format,
        "version": o.meta.version,
        "segment": o.meta.segment,
        "tracks": tracks,
    });
    let mut out = String::new();
    yaml_value(&doc, 0, &mut out);
    Ok(out)
}

/// `value` as block-style YAML at `indent` (strings in JSON quoting, which
/// YAML reads; multi-line text as a literal block).
pub(crate) fn yaml_value(v: &serde_json::Value, indent: usize, out: &mut String) {
    use serde_json::Value;
    let pad = " ".repeat(indent);
    match v {
        Value::Object(m) => {
            for (key, val) in m {
                match val {
                    Value::Object(o) if !o.is_empty() => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        yaml_value(val, indent + 2, out);
                    }
                    Value::Array(a) if !a.is_empty() && a.iter().any(|e| e.is_object() || e.is_array()) => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        yaml_value(val, indent + 2, out);
                    }
                    Value::String(s) if s.contains('\n') => {
                        out.push_str(&format!("{pad}{key}: |\n"));
                        for line in s.lines() {
                            out.push_str(&format!("{pad}  {line}\n"));
                        }
                    }
                    _ => out.push_str(&format!("{pad}{key}: {}\n", scalar(val))),
                }
            }
        }
        Value::Array(a) => {
            for e in a {
                let mut inner = String::new();
                yaml_value(e, indent + 2, &mut inner);
                if e.is_object() && !inner.is_empty() {
                    // "- " takes the place of the first line's indent.
                    out.push_str(&format!("{pad}- {}", &inner[indent + 2..]));
                } else {
                    out.push_str(&format!("{pad}- {}\n", scalar(e)));
                }
            }
        }
        _ => out.push_str(&format!("{pad}{}\n", scalar(v))),
    }
}

fn scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Array(a) => format!("[{}]", a.iter().map(scalar).collect::<Vec<_>>().join(", ")),
        serde_json::Value::Object(_) => "{}".into(),
        other => other.to_string(),
    }
}

/// Write what the archive keeps besides the samples, without decoding any:
/// each track's `.yaml` sidecars, as they were (checked by hash), and
/// `<archive>.yaml`, the whole metadata.
pub fn unpack_meta(path: &Path, outdir: &Path, only: Option<usize>) -> Result<()> {
    let o = open(path)?;
    std::fs::create_dir_all(outdir)?;
    for (k, sm) in o.meta.sources.iter().enumerate() {
        if only.is_some_and(|t| t != k) {
            continue;
        }
        for sc in &sm.sidecars {
            let name = Path::new(&sc.name).file_name().context("bad sidecar name in the archive")?;
            if hex(&Sha256::digest(&sc.data)) != sc.sha256 {
                bail!("{}: the stored {} does not match its hash", sm.name, sc.name);
            }
            std::fs::write(outdir.join(name), &sc.data)?;
            println!("{}", outdir.join(name).display());
        }
    }
    let stem = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "archive.rfx".into());
    let y = outdir.join(format!("{stem}.yaml"));
    std::fs::write(&y, meta_yaml(path)?)?;
    println!("{}", y.display());
    Ok(())
}
