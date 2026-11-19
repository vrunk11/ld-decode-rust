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
    /// The face of the disc: `A`, `B`, `AB` or `BA` (when it was said).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face: Option<String>,
    /// The video standard of the disc as a person said it: `pal`, `ntsc`,
    /// `pal-m`, `secam` or `ntsc-vhd` (when it was said).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard: Option<String>,
    /// A reference of the disc (a catalogue number, a title).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The number of the physical copy of the disc the capture was made from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy: Option<u32>,
    /// The number of the capture itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture: Option<u32>,
}

impl SrcMeta {
    /// What the archive says of this capture, without what its predictor
    /// needs: the object `ldx.rfsource` of an LDX file keeps (the members that
    /// have their default value are left out).
    pub fn source_json(&self) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        let mut o = Map::new();
        o.insert("name".into(), json!(self.name));
        o.insert("size".into(), json!(self.size));
        o.insert("sha256".into(), json!(self.sha256));
        o.insert("words".into(), json!(self.words));
        o.insert("fmt".into(), json!(self.fmt));
        if !self.sample_sha256.is_empty() {
            o.insert("sample_sha256".into(), json!(self.sample_sha256));
        }
        if self.bits != 0 {
            o.insert("bits".into(), json!(self.bits));
        }
        if self.rate_hz != 0.0 {
            o.insert("rate_hz".into(), json!(self.rate_hz));
        }
        if !self.file_time.is_empty() {
            o.insert("file_time".into(), json!(self.file_time));
        }
        if !self.notes.is_empty() {
            o.insert("notes".into(), json!(self.notes));
        }
        for (k, v) in [("origin", &self.origin), ("face", &self.face), ("standard", &self.standard), ("reference", &self.reference)] {
            if let Some(v) = v {
                o.insert(k.into(), json!(v));
            }
        }
        for (k, v) in [("copy", self.copy), ("capture", self.capture)] {
            if let Some(v) = v {
                o.insert(k.into(), json!(v));
            }
        }
        o.insert("position".into(), json!(self.position));
        if let Some(n) = self.noise {
            o.insert("noise".into(), json!(n));
        }
        if !self.sidecars.is_empty() {
            o.insert("sidecars".into(), serde_json::to_value(&self.sidecars).unwrap_or(Value::Null));
        }
        Value::Object(o)
    }
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
    #[serde(flatten)]
    pub root: Root,
}

/// What the archive says of the whole file (specification 3): the source, the
/// cover of the disc and its link, and the history of the file. All optional.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Root {
    /// `ld`, `vhs`, `vhd` or `cvbs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<String>,
    /// The picture, PNG or JPEG, as Base64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cover_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_url: Option<String>,
    /// One entry per time a tool wrote or changed the file, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<serde_json::Value>,
}

/// The cover picture of an archive: the bytes and `image/png` or `image/jpeg`.
#[derive(Clone, Debug)]
pub struct Cover {
    pub media_type: &'static str,
    pub data: Vec<u8>,
}

impl Root {
    /// The picture, when the file keeps one that fits (PNG or JPEG, at most
    /// 2 MiB; the type is found from the first bytes, as the specification
    /// says, when `cover_type` is missing).
    pub fn cover(&self) -> Option<Cover> {
        let data = base64_decode(self.cover.as_deref()?)?;
        if data.len() > 2 << 20 {
            return None;
        }
        let media_type = if data.starts_with(&[0x89, b'P', b'N', b'G']) {
            "image/png"
        } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "image/jpeg"
        } else {
            return None;
        };
        Some(Cover { media_type, data })
    }

    /// `link_url` when it is a web address on one line.
    pub fn link(&self) -> Option<&str> {
        let u = self.link_url.as_deref()?.trim();
        ((u.starts_with("http://") || u.starts_with("https://")) && !u.contains(['\n', '\r'])).then_some(u)
    }
}

/// Base64 (standard alphabet, padding and white space allowed).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
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
    let mut meta: Meta = serde_json::from_slice(&buf).context("bad metadata")?;
    if meta.format != "rfx" || meta.version != 1 {
        bail!("unsupported rfx version {}", meta.version);
    }
    // A newer copy of the metadata added at the end (specification 3.2): the
    // last block of timestamp 2 counts instead of the first one, for the same archive.
    if let Some(ub) = idx.blocks.iter().rev().find(|b| b.track == meta_track && b.ts == 2) {
        let mut buf = vec![0u8; ub.len as usize];
        read_exact_at(&file, &mut buf, ub.off)?;
        let newer: Meta = serde_json::from_slice(&buf).context("bad updated metadata")?;
        if newer.format != "rfx" || newer.version != 1 || newer.segment != meta.segment || newer.sources.len() != meta.sources.len() {
            bail!("the updated metadata of {} do not describe the same archive", path.display());
        }
        meta = newer;
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

/// A field of a capture as pass 0 located it (specification 3.1).
#[derive(Clone, Debug, PartialEq)]
pub struct SavedField {
    pub first: bool,
    /// Frame code words, as ld-decode's pass 0 reads them (`None`: no code).
    pub code: Option<Vec<i64>>,
    /// Sample position of each line (rounded to 0.01 sample).
    pub lines: Vec<f64>,
}

/// What pass 0 found in the RF of a track, kept in the archive.
#[derive(Clone, Debug)]
pub struct SavedPass0 {
    pub pal: bool,
    pub rate_hz: f64,
    pub fields: Vec<SavedField>,
}

/// The text of a frame code as pass 0 writes it: `None`, or `Some([w0, w1, ...])`.
fn parse_code(s: &str) -> Option<Vec<i64>> {
    let inner = s.trim().strip_prefix("Some([")?.strip_suffix("])")?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').map(|w| w.trim().parse().ok()).collect::<Option<Vec<i64>>>()
}

/// The pass 0 curves an archive of the predictor variant keeps (the last
/// block of timestamp 1 of the metadata track), one entry per track (`None`:
/// that track has none). An archive that keeps none, or is of another
/// variant, gives all `None`.
pub fn saved_pass0(path: &Path) -> Result<Vec<Option<SavedPass0>>> {
    if crate::flacmode::is_flac_archive(path) {
        return Ok(Vec::new());
    }
    let o = open(path)?;
    let idx = ebml::scan(&o.file)?;
    let n = o.meta.sources.len();
    let mut out: Vec<Option<SavedPass0>> = vec![None; n];
    let meta_track = idx.codecs.iter().find(|(_, c)| c == CODEC_META).map(|(t, _)| *t).context("no metadata track")?;
    let Some(b) = idx.blocks.iter().rev().find(|b| b.track == meta_track && b.ts == 1) else { return Ok(out) };
    let mut buf = vec![0u8; b.len as usize];
    read_exact_at(&o.file, &mut buf, b.off)?;
    let v: serde_json::Value = serde_json::from_slice(&buf).context("bad pass 0 block")?;
    for e in v.get("pass0").and_then(|x| x.as_array()).into_iter().flatten() {
        let Some(t) = e.get("track").and_then(|x| x.as_u64()).map(|t| t as usize).filter(|&t| t < n) else { continue };
        let fields: Vec<SavedField> = e
            .get("fields")
            .and_then(|x| x.as_array())
            .into_iter()
            .flatten()
            .map(|f| SavedField {
                first: f.get("first").and_then(|x| x.as_i64()).unwrap_or(0) != 0,
                code: f.get("code").and_then(|x| x.as_str()).and_then(parse_code),
                lines: f.get("lines").and_then(|x| x.as_array()).map(|l| l.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default(),
            })
            .collect();
        if !fields.is_empty() {
            out[t] = Some(SavedPass0 { pal: e.get("system").and_then(|x| x.as_str()) == Some("pal"), rate_hz: e.get("rate_hz").and_then(|x| x.as_f64()).unwrap_or(0.0), fields });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod pass0_tests {
    #[test]
    fn frame_codes_read_as_pass_0_writes_them() {
        assert_eq!(super::parse_code("None"), None);
        assert_eq!(super::parse_code("Some([15, 4096, -3])"), Some(vec![15, 4096, -3]));
        assert_eq!(super::parse_code("Some([])"), Some(vec![]));
    }

    /// The newer copy of the metadata added at the end (timestamp 2, section
    /// 3.2) counts instead of the first one; the root says what the file says.
    #[test]
    fn a_newer_copy_of_the_metadata_replaces_the_first() {
        use std::io::Write;
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/reader_ldx.rfx");
        let dir = std::env::temp_dir().join(format!("rfx-meta2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.rfx");
        std::fs::copy(&src, &path).unwrap();
        let first = super::open(&path).unwrap();
        assert_eq!(first.meta.root.medium.as_deref(), Some("ld"));
        assert_eq!(first.meta.root.history.len(), 2);
        assert_eq!(first.meta.root.cover().unwrap().media_type, "image/png");
        assert_eq!(first.meta.root.link(), Some("https://example.com/disc"));
        assert_eq!((first.meta.sources[0].face.as_deref(), first.meta.sources[0].copy), (Some("A"), Some(2)));
        let mut v = serde_json::to_value(&first.meta).unwrap();
        v["medium"] = "vhs".into();
        v["sources"][0]["notes"] = "second look".into();
        v["history"].as_array_mut().unwrap().push(serde_json::json!({"tool": "rfx", "command": "metadata"}));
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&crate::ebml::cluster(2, 2, v.to_string().as_bytes())).unwrap();
        drop(f);
        let now = super::open(&path).unwrap();
        assert_eq!(now.meta.root.medium.as_deref(), Some("vhs"));
        assert_eq!(now.meta.root.history.len(), 3);
        assert_eq!(now.meta.sources[0].notes, "second look");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A block of timestamp 1 appended to a file written by the rfx tool (as
    /// the tool appends it): the last one counts, a track without one gives None.
    #[test]
    fn reads_the_pass_0_block_appended_to_an_archive() {
        use std::io::Write;
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/reader.rfx");
        let dir = std::env::temp_dir().join(format!("rfx-pass0-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.rfx");
        std::fs::copy(&src, &path).unwrap();
        assert!(super::saved_pass0(&path).unwrap().iter().all(|p| p.is_none()));
        let old = br#"{"pass0":[{"track":0,"system":"pal","rate_hz":1.0,"fields":[{"first":1,"lines":[1.0]}]}]}"#;
        let new = br#"{"pass0":[{"track":0,"system":"ntsc","rate_hz":40000000.0,"fields":[{"first":1,"code":"Some([5, 7])","lines":[10.5,74.25]},{"first":0,"lines":[300.0]}]}]}"#;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&crate::ebml::cluster(1, 2, old)).unwrap();
        f.write_all(&crate::ebml::cluster(1, 2, new)).unwrap();
        drop(f);
        let got = super::saved_pass0(&path).unwrap();
        let p = got[0].clone().unwrap();
        assert!(!p.pal && p.rate_hz == 40e6);
        assert_eq!(p.fields.len(), 2);
        assert_eq!(p.fields[0], super::SavedField { first: true, code: Some(vec![5, 7]), lines: vec![10.5, 74.25] });
        assert_eq!(p.fields[1], super::SavedField { first: false, code: None, lines: vec![300.0] });
        // The archive still reads as before.
        assert_eq!(crate::reader::Archive::open(&path).unwrap().tracks(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
