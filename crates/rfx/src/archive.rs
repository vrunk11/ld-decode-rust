//! Opening an archive: the metadata, the lag tables and the blocks of a `.rfx`
//! file, and the reference windows a segment is predicted from.
//!
//! Layout of a `.rfx` file (Matroska, DocType `rfx`; see the specification in
//! the RF-tools repository):
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
//! the earlier tracks: segments are decoded in parallel.
use crate::align::LagTable;
use crate::ebml;
use crate::fmt::Fmt;
use crate::io::read_exact_at;
use crate::pred::Preset;
use crate::seg::refwin_len;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub const CODEC_PRED: &str = "D_RFX/PRED";
pub const CODEC_META: &str = "D_RFX/META";
pub const CODEC_MAP: &str = "D_RFX/MAP";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SrcMeta {
    pub name: String,
    /// A note about the capture (empty when there is none).
    #[serde(default)]
    pub notes: String,
    /// Date of the original file, as the packer wrote it (empty when not known).
    #[serde(default)]
    pub file_time: String,
    /// Bits the capture carries (0: not told).
    #[serde(default)]
    pub bits: u32,
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
        let add = |j: usize, w: f32, dst: &mut [f32]| -> Result<()> {
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

pub(crate) struct Opened {
    pub(crate) meta: Meta,
    pub(crate) tables: HashMap<(usize, usize), LagTable>,
    pub(crate) blocks: Vec<HashMap<u64, (u64, u64)>>,
    pub(crate) file: File,
}

pub(crate) fn open(path: &Path) -> Result<Opened> {
    if crate::flacmode::is_flac_archive(path) {
        bail!("{} holds FLAC or raw tracks (packed with --flac or --raw): it is a Matroska audio file, and rfx reads it only to unpack, info, or extract the metadata", path.display());
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
