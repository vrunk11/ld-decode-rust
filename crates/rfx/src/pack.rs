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
use crate::align::{build_table, parse_pass0, run_pass0, LagTable};
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
    pub taps: usize,
    pub stride: usize,
    pub every: usize,
    pub lam_bits: u32,
    /// Earlier sources this one is predicted from (those it shares frames with).
    pub refs: Vec<usize>,
}

impl SrcMeta {
    fn preset(&self) -> Preset {
        Preset { own: self.own, taps: self.taps, stride: self.stride, every: self.every, lam_bits: self.lam_bits }
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

/// Segments sent to the GPU at once.
const GPU_CHUNK: usize = 512;

pub struct PackOpts {
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
struct Refs<'a> {
    files: &'a [File],
    meta: &'a Meta,
    tables: &'a HashMap<(usize, usize), LagTable>,
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
    /// Mean of the reference sources over positions `[s0 - taps, s0 + n + taps)`.
    fn window(&self, k: usize, s0: i64, n: usize) -> Result<Vec<f32>> {
        let sm = &self.meta.sources[k];
        let len = refwin_len(n, &sm.preset());
        let p0 = s0 - sm.taps as i64;
        let mut acc = vec![0i32; len];
        for &j in &sm.refs {
            let rj = &self.meta.sources[j];
            let t = &self.tables[&(k, j)];
            for (a, b, lag) in runs(t, p0, p0 + len as i64) {
                let lo = (a + lag).max(0);
                let hi = (b + lag).min(rj.words as i64);
                if hi <= lo {
                    continue;
                }
                let v = read_words(&self.files[j], rj.words, lo, (hi - lo) as usize, rj.shift)?;
                let base = (lo - lag - p0) as usize;
                for (o, x) in acc[base..].iter_mut().zip(v) {
                    *o += x - rj.mean;
                }
            }
        }
        let inv = if sm.refs.is_empty() { 0.0 } else { 1.0f32 / sm.refs.len() as f32 };
        Ok(acc.into_iter().map(|a| a as f32 * inv).collect())
    }
}

pub struct SourceReport {
    pub name: String,
    pub size: u64,
    pub packed: u64,
    pub refs: usize,
}

pub fn pack(inputs: &[PathBuf], out: &Path, o: &PackOpts) -> Result<Vec<SourceReport>> {
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

    // Lag tables between the captures, from ld-decode's pass 0.
    let mut tables: HashMap<(usize, usize), LagTable> = HashMap::new();
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
        let fields: Vec<_> = (0..inputs.len()).map(|i| parse_pass0(&dir.join(format!("pass0_{i}.txt")))).collect::<Result<_>>()?;
        for k in 1..inputs.len() {
            for j in 0..k {
                match build_table(&fields[k], &fields[j], stats[k].size / 2) {
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
                taps: preset.taps,
                stride: preset.stride,
                every: preset.every,
                lam_bits: preset.lam_bits,
                refs: (0..k).filter(|j| tables.contains_key(&(k, *j))).collect(),
            })
        })
        .collect::<Result<_>>()?;
    let meta = Meta { format: "rfx".into(), version: 1, segment: o.segment, sources };
    let files: Vec<File> = inputs.iter().map(|p| File::open(p).map_err(Into::into)).collect::<Result<_>>()?;
    let refs = Refs { files: &files, meta: &meta, tables: &tables };

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
        let chunk = if o.device == Device::Gpu { GPU_CHUNK } else { rayon::current_num_threads() * 4 };
        let mut gpu = if o.device == Device::Gpu { Some(GpuSeg::new(&pre, use_ref, s)?) } else { None };
        for first in (0..nseg).step_by(chunk) {
            let segs: Vec<usize> = (first..(first + chunk).min(nseg)).collect();
            if let Some(g) = gpu.as_mut() {
                let rw = g.window_len();
                let prepared: Vec<Result<(Vec<i32>, Vec<f32>)>> = segs
                    .par_iter()
                    .map(|&q| {
                        let s0 = q * s;
                        let n = s.min(sm.words as usize - s0);
                        let y: Vec<i32> = read_words(&files[k], sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
                        let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
                        Ok((y, win))
                    })
                    .collect();
                let prepared: Vec<(Vec<i32>, Vec<f32>)> = prepared.into_iter().collect::<Result<_>>()?;
                let mut input = vec![0i32; segs.len() * s];
                let mut refw = if use_ref { vec![0f32; segs.len() * rw] } else { Vec::new() };
                let mut lens = Vec::new();
                for (i, (y, win)) in prepared.iter().enumerate() {
                    input[i * s..i * s + y.len()].copy_from_slice(y);
                    if use_ref {
                        refw[i * rw..i * rw + win.len()].copy_from_slice(win);
                    }
                    lens.push(y.len() as i32);
                }
                let preds = g.run(0, &input, &refw, &lens)?;
                let coded: Vec<(Vec<u8>, Vec<i32>)> = prepared
                    .par_iter()
                    .enumerate()
                    .map(|(i, (y, _))| {
                        let res: Vec<i32> = y.iter().zip(&preds[i * s..]).map(|(a, b)| a - b).collect();
                        (seg::encode_residuals(&res), res)
                    })
                    .collect();
                if o.verify {
                    let decoded: Vec<Option<Vec<i32>>> = coded.par_iter().zip(&prepared).map(|((bytes, _), (y, _))| seg::decode_residuals(bytes, y.len())).collect();
                    let mut flat = vec![0i32; segs.len() * s];
                    for (i, ((_, res), d)) in coded.iter().zip(&decoded).enumerate() {
                        match d {
                            Some(d) if d == res => flat[i * s..i * s + d.len()].copy_from_slice(d),
                            _ => bail!("self-check failed: segment {} of {} does not decode to its residuals", segs[i], sm.name),
                        }
                    }
                    let back = g.run(1, &flat, &refw, &lens)?;
                    for (i, (y, _)) in prepared.iter().enumerate() {
                        if back[i * s..i * s + y.len()] != y[..] {
                            bail!("self-check failed: segment {} of {} does not decode to itself on the GPU", segs[i], sm.name);
                        }
                    }
                }
                for (q, (bytes, _)) in segs.iter().zip(&coded) {
                    packed += bytes.len() as u64;
                    w.write_all(&ebml::cluster(*q as u64, k as u64 + 1, bytes))?;
                }
                continue;
            }
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
        let chunk = if device == Device::Gpu { GPU_CHUNK } else { rayon::current_num_threads() * 4 };
        let mut gpu = if device == Device::Gpu { Some(GpuSeg::new(&pre, use_ref, s)?) } else { None };
        for first in (0..nseg).step_by(chunk) {
            let segs: Vec<usize> = (first..(first + chunk).min(nseg)).collect();
            if let Some(g) = gpu.as_mut() {
                let rw = g.window_len();
                let prepared: Vec<Result<(Vec<i32>, Vec<f32>)>> = segs
                    .par_iter()
                    .map(|&q| {
                        let s0 = q * s;
                        let n = s.min(sm.words as usize - s0);
                        let &(off, len) = o.blocks[k].get(&(q as u64)).with_context(|| format!("{}: segment {q} is missing", sm.name))?;
                        let mut bytes = vec![0u8; len as usize];
                        read_exact_at(&o.file, &mut bytes, off)?;
                        let res = seg::decode_residuals(&bytes, n).with_context(|| format!("{}: segment {q} is corrupt", sm.name))?;
                        let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
                        Ok((res, win))
                    })
                    .collect();
                let prepared: Vec<(Vec<i32>, Vec<f32>)> = prepared.into_iter().collect::<Result<_>>()?;
                let mut input = vec![0i32; segs.len() * s];
                let mut refw = if use_ref { vec![0f32; segs.len() * rw] } else { Vec::new() };
                let mut lens = Vec::new();
                for (i, (res, win)) in prepared.iter().enumerate() {
                    input[i * s..i * s + res.len()].copy_from_slice(res);
                    if use_ref {
                        refw[i * rw..i * rw + win.len()].copy_from_slice(win);
                    }
                    lens.push(res.len() as i32);
                }
                let ys = g.run(1, &input, &refw, &lens)?;
                segs.par_iter().enumerate().try_for_each(|(i, &q)| -> Result<()> {
                    let n = lens[i] as usize;
                    let mut raw = Vec::with_capacity(n * 2);
                    for &v in &ys[i * s..i * s + n] {
                        raw.extend_from_slice(&(((v + sm.mean) << sm.shift) as i16).to_le_bytes());
                    }
                    write_all_at(&outs[k], &raw, (q * s) as u64 * 2)?;
                    Ok(())
                })?;
                continue;
            }
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
pub fn gpu_check(path: &Path, level: u32, nseg: usize, segment: usize) -> Result<()> {
    let pre = Preset::level(level).context("level 1 to 4")?;
    let st = file_stats(path)?;
    let f = File::open(path)?;
    let words = st.size / 2;
    let mut input = vec![0i32; nseg * segment];
    let mut lens = Vec::new();
    let mut cpu = Vec::new();
    for g in 0..nseg {
        let n = segment.min((words as usize).saturating_sub(g * segment));
        let y: Vec<i32> = read_words(&f, words, (g * segment) as i64, n, st.shift)?.into_iter().map(|v| v - st.mean).collect();
        input[g * segment..g * segment + n].copy_from_slice(&y);
        lens.push(n as i32);
        cpu.push(seg::predict_all(&y, &pre, false, &[]));
    }
    let mut gpu = GpuSeg::new(&pre, false, segment)?;
    let t0 = std::time::Instant::now();
    let out = gpu.run(0, &input, &[], &lens)?;
    eprintln!("GPU: {:.1} Msmp/s", lens.iter().sum::<i32>() as f64 / t0.elapsed().as_secs_f64() / 1e6);
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
