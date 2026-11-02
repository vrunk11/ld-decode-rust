//! `pack`, `unpack` and `info`.
//!
//! Layout of a `.rfx` file (Matroska, DocType `rfx`):
//! - track `k + 1` (`D_RFX/PRED`): source `k`, one block per segment of
//!   `segment` samples, each block the range-coded residuals of
//!   [`crate::seg`];
//! - track `N + 1` (`D_RFX/META`): one block of JSON ([`Meta`]): names,
//!   sizes, SHA-256, predictor settings;
//! - track `N + 2` (`D_RFX/MAP`): one block per (source, earlier source)
//!   pair with the integer lag table ([`LagTable`]).
//!
//! Source `k` is predicted from its own past and from the mean of the
//! sources of its `refs` (all earlier), read at the lag tables' positions in
//! the files themselves, so a segment needs nothing but its own block and
//! the earlier sources: segments are coded and decoded in parallel.
use crate::align::{build_table, parse_pass0, run_pass0, LagTable, HYST_COARSE, HYST_FINE};
use crate::ebml;
use crate::gpu::GpuSeg;
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
    /// 16-bit words of the file; an odd last byte is in `tail`.
    pub words: u64,
    pub tail: Vec<u8>,
    /// Low bits that are zero in every word (the Duplicator's 10 bits in 16).
    pub shift: u32,
    pub mean: i32,
    pub own: usize,
    /// Own past samples when the source has no reference (0: same as `own`).
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
    /// Earlier sources this one is predicted from (those it shares frames with).
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
fn find_sidecars(capture: &Path) -> Vec<Sidecar> {
    let p = capture.with_extension("yaml");
    match (std::fs::metadata(&p), p.file_name()) {
        (Ok(m), Some(name)) if m.is_file() && m.len() <= SIDECAR_MAX => match std::fs::read(&p) {
            Ok(data) => vec![Sidecar { name: name.to_string_lossy().into_owned(), sha256: hex(&Sha256::digest(&data)), data }],
            Err(_) => Vec::new(),
        },
        _ => Vec::new(),
    }
}

impl SrcMeta {
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

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[derive(Clone)]
struct Stats {
    size: u64,
    sha: String,
    shift: u32,
    mean: i32,
}

fn file_stats(path: &Path) -> Result<Stats> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let size = f.metadata()?.len();
    let words = size / 2;
    let mut hasher = Sha256::new();
    let (mut or, mut sum) = (0u16, 0i64);
    let mut buf = vec![0u8; 1 << 24];
    let mut off = 0u64;
    while off < size {
        let n = ((size - off) as usize).min(buf.len());
        read_exact_at(&f, &mut buf[..n], off)?;
        hasher.update(&buf[..n]);
        for w in buf[..n / 2 * 2].chunks_exact(2) {
            let v = i16::from_le_bytes([w[0], w[1]]);
            or |= v as u16;
            sum += i64::from(v);
        }
        off += n as u64;
    }
    let shift = if or == 0 { 0 } else { or.trailing_zeros() };
    let mean = if words == 0 { 0 } else { ((sum >> shift) as f64 / words as f64).round() as i32 };
    Ok(Stats { size, sha: hex(&hasher.finalize()), shift, mean })
}

/// Source `j`'s files, as the predictor of a later source reads them.
pub(crate) struct Refs<'a> {
    pub(crate) files: &'a [File],
    pub(crate) meta: &'a Meta,
    pub(crate) tables: &'a HashMap<(usize, usize), LagTable>,
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
                crate::io::add_words_scaled(&self.files[j], lo, hi, rj.shift, rj.mean, w, &mut dst[base..base + (hi - lo) as usize])?;
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
            let y: Vec<i32> = read_words(&refs.files[k], sm.words, s0 as i64, win, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
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
}

pub(crate) fn prepare(inputs: &[PathBuf], out: &Path, o: &PackOpts) -> Result<Prepared> {
    if inputs.is_empty() {
        bail!("no input");
    }
    let preset = Preset::level(o.level).with_context(|| format!("level {} is not 1 to 4", o.level))?;
    let mut names = Vec::new();
    for p in inputs {
        let n = p.file_name().context("input has no file name")?.to_string_lossy().into_owned();
        if names.contains(&n) {
            bail!("two inputs are named {n}: the archive keeps file names only");
        }
        names.push(n);
    }
    eprintln!("hashing and scanning {} files", inputs.len());
    let stats: Vec<Stats> = inputs.par_iter().map(|p| file_stats(p)).collect::<Result<_>>()?;

    // Where the lines of every capture are, from ld-decode's pass 0 (in the
    // order given), then the order to code them in.
    let mut fields: Vec<Vec<crate::align::Field>> = Vec::new();
    if inputs.len() > 1 {
        let dir = match &o.pass0_dir {
            Some(d) => d.clone(),
            None => {
                let d = out.with_extension("pass0");
                eprintln!("locating the lines of every capture (ld-decode pass 0)");
                run_pass0(&o.ld_decode, inputs, o.rate_mhz, &o.format, &d)?;
                d
            }
        };
        fields = (0..inputs.len()).map(|i| parse_pass0(&dir.join(format!("pass0_{i}.txt")))).collect::<Result<_>>()?;
    }
    let mut noise: Vec<Option<f64>> = vec![None; inputs.len()];
    let mut ord: Vec<usize> = (0..inputs.len()).collect();
    if inputs.len() > 1 && o.order == OrderMode::Auto {
        let (order, scores) = crate::order::plan(inputs, &fields);
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
    let names: Vec<String> = ord.iter().map(|&i| names[i].clone()).collect();
    let stats: Vec<Stats> = ord.iter().map(|&i| stats[i].clone()).collect();
    let noise: Vec<Option<f64>> = ord.iter().map(|&i| noise[i]).collect();
    let mut tables: HashMap<(usize, usize), LagTable> = HashMap::new();
    let fields: Vec<Vec<crate::align::Field>> = {
        let mut slots: Vec<Option<Vec<crate::align::Field>>> = fields.into_iter().map(Some).collect();
        if slots.is_empty() {
            Vec::new()
        } else {
            ord.iter().map(|&i| slots[i].take().expect("each once")).collect()
        }
    };
    if inputs.len() > 1 {
        for k in 1..inputs.len() {
            for j in 0..k {
                match build_table(&fields[k], &fields[j], stats[k].size / 2, HYST_FINE) {
                    Some(t) => {
                        tables.insert((k, j), t);
                    }
                    None => eprintln!("{} and {} share no frame: not used as a reference", names[k], names[j]),
                }
            }
        }
    }

    let sources: Vec<SrcMeta> = inputs
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let st = &stats[k];
            let tail = if st.size % 2 == 1 {
                let f = File::open(p)?;
                let mut b = [0u8; 1];
                read_exact_at(&f, &mut b, st.size - 1)?;
                b.to_vec()
            } else {
                Vec::new()
            };
            Ok(SrcMeta {
                name: names[k].clone(),
                size: st.size,
                sha256: st.sha.clone(),
                words: st.size / 2,
                tail,
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
                position: ord[k],
                noise: noise[k],
                sidecars: find_sidecars(p),
            })
        })
        .collect::<Result<_>>()?;
    let meta = Meta { format: "rfx".into(), version: 1, segment: o.segment, sources };
    let files: Vec<File> = inputs.iter().map(|p| File::open(p).map_err(Into::into)).collect::<Result<_>>()?;
    // Weigh the references of each capture by how well each predicts it.
    let mut meta = meta;
    if inputs.len() > 1 {
        let choice: Vec<(Vec<usize>, Vec<f32>)> = {
            let refs = Refs { files: &files, meta: &meta, tables: &tables };
            (0..inputs.len()).map(|k| choose_refs(&refs, k)).collect::<Result<_>>()?
        };
        for (k, (r, w)) in choice.into_iter().enumerate() {
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
            let nr = meta.sources[k].refs.len();
            meta.sources[k].own = preset.plain;
            meta.sources[k].ntop = 0;
            if nr < 3 {
                continue;
            }
            let plain = trial_cost(&Refs { files: &files, meta: &meta, tables: &tables }, k)?;
            // The channels take up the misalignment: try them with coarse tables.
            let fine: Vec<(usize, LagTable)> = meta.sources[k].refs.iter().filter_map(|&j| tables.get(&(k, j)).map(|t| (j, t.clone()))).collect();
            for &(j, _) in &fine {
                if let Some(t) = build_table(&fields[k], &fields[j], meta.sources[k].words, HYST_COARSE) {
                    tables.insert((k, j), t);
                }
            }
            meta.sources[k].own = preset.own;
            meta.sources[k].ntop = preset.ntop.min(nr);
            let split = trial_cost(&Refs { files: &files, meta: &meta, tables: &tables }, k)?;
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
    Ok(Prepared { inputs, names, meta, tables, files })
}

pub fn pack(inputs: &[PathBuf], out: &Path, o: &PackOpts) -> Result<Vec<SourceReport>> {
    let p = prepare(inputs, out, o)?;
    let Prepared { inputs, names, meta, tables, files } = &p;
    let refs = Refs { files, meta, tables };

    let n_src = inputs.len() as u64;
    let mut w = std::io::BufWriter::new(File::create(out).with_context(|| format!("creating {}", out.display()))?);
    w.write_all(&ebml::file_head())?;
    w.write_all(&ebml::info())?;
    let mut entries: Vec<Vec<u8>> = names.iter().enumerate().map(|(k, n)| ebml::track_entry(k as u64 + 1, CODEC_PRED, n)).collect();
    entries.push(ebml::track_entry(n_src + 1, CODEC_META, "meta"));
    entries.push(ebml::track_entry(n_src + 2, CODEC_MAP, "map"));
    w.write_all(&ebml::tracks(&entries))?;
    w.write_all(&ebml::cluster(0, n_src + 1, &serde_json::to_vec(&meta)?))?;
    let mut pairs: Vec<_> = tables.keys().copied().collect();
    pairs.sort_unstable();
    for (i, (k, j)) in pairs.iter().enumerate() {
        let mut b = Vec::new();
        crate::align::put_varint(&mut b, *k as u64);
        crate::align::put_varint(&mut b, *j as u64);
        b.extend(tables[&(*k, *j)].encode());
        w.write_all(&ebml::cluster(i as u64, n_src + 2, &b))?;
    }

    let mut report = Vec::new();
    let s = o.segment;
    for k in 0..inputs.len() {
        let sm = &meta.sources[k];
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
                    let y: Vec<i32> = read_words(&files[k], sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
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

struct Opened {
    meta: Meta,
    tables: HashMap<(usize, usize), LagTable>,
    blocks: Vec<HashMap<u64, (u64, u64)>>,
    file: File,
}

fn open(path: &Path) -> Result<Opened> {
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
    println!("{}: {} sources, segments of {} samples", path.display(), o.meta.sources.len(), o.meta.segment);
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
/// With `only`, just that source (the earlier ones are decoded too, as its
/// references, and removed).
pub fn unpack(path: &Path, outdir: &Path, only: Option<usize>, device: Device) -> Result<()> {
    let o = open(path)?;
    let meta = &o.meta;
    std::fs::create_dir_all(outdir)?;
    let last = only.unwrap_or(meta.sources.len() - 1);
    if last >= meta.sources.len() {
        bail!("the archive has {} sources", meta.sources.len());
    }
    let mut outs: Vec<File> = Vec::new();
    for sm in &meta.sources[..=last] {
        let name = Path::new(&sm.name).file_name().context("bad name in the archive")?;
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(outdir.join(name))?;
        f.set_len(sm.size)?;
        outs.push(f);
    }
    let refs = Refs { files: &outs, meta, tables: &o.tables };
    let s = meta.segment;
    for k in 0..=last {
        let sm = &meta.sources[k];
        let pre = sm.preset();
        let use_ref = !sm.refs.is_empty();
        let nseg = (sm.words as usize).div_ceil(s);
        let t0 = std::time::Instant::now();
        if device == Device::Gpu {
            crate::gpupipe::unpack_source(k, sm, &pre, s, &o.blocks[k], &o.file, &refs, &outs[k])?;
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
                let mut raw = Vec::with_capacity(n * 2);
                for v in y {
                    raw.extend_from_slice(&(((v + sm.mean) << sm.shift) as i16).to_le_bytes());
                }
                write_all_at(&outs[k], &raw, s0 as u64 * 2)?;
                Ok(())
            })?;
        }
        if !sm.tail.is_empty() {
            write_all_at(&outs[k], &sm.tail, sm.words * 2)?;
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
    }
    Ok(())
}

/// Development check: the predictions of the first `nseg` segments of a file,
/// CPU against GPU, and the first sample where they differ.
pub fn gpu_check(path: &Path, level: u32, nseg: usize, segment: usize, with_ref: bool) -> Result<()> {
    let pre = Preset::level(level).context("level 1 to 4")?;
    let st = file_stats(path)?;
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
        let y: Vec<i32> = read_words(&f, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect();
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
    let st = file_stats(path)?;
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
            let y: Vec<i32> = read_words(&f, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect();
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
    let st = file_stats(path)?;
    let f = File::open(path)?;
    let words = st.size / 2;
    let n = segment;
    let ys: Vec<Vec<i32>> = (0..nseg)
        .map(|g| Ok(read_words(&f, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect()))
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
                crate::io::add_words_scaled(&self.files[j], lo, hi, rj.shift, rj.mean, w, &mut out[base..base + (hi - lo) as usize])?;
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
    let refs = Refs { files, meta, tables };
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
            let y: Vec<i32> = read_words(&files[k], sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
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
            let y: Vec<i32> = read_words(&refs.files[k], sm.words, s0 as i64, win, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let w = refs.window(k, s0 as i64, win)?;
            let preds = seg::predict_all(&y, &pre, true, &w);
            let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            Ok(seg::encode_residuals(&res).len() as f64 * 8.0 / win as f64)
        })
        .collect::<Result<_>>()?;
    Ok(costs.iter().sum::<f64>() / costs.len() as f64)
}
