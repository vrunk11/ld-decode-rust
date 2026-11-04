//! The FLAC variant of the archive, read by sample index (what ld-decode does):
//! an ordinary Matroska audio file with a FLAC track per capture (`A_FLAC`) and
//! one attachment, `rfx.json`, that keeps what the archive holds besides the
//! samples. Writing such a file is the job of the `rfx` tool of RF-tools.
//!
//! The samples are 16-bit ones: the capture's sample scaled to 16 bits (a capture of 8 bits is multiplied by 256, an unsigned one is offset), and
//! `from_i16` gives it back.
use crate::archive::SrcMeta;
use crate::fmt::Fmt;
use crate::io::read_exact_at;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct FlacMeta {
    pub format: String,
    pub codec: String,
    pub version: u32,
    pub tracks: Vec<SrcMeta>,
}

/// Whether `path` is a FLAC-mode archive (a Matroska file) rather than one of
/// the predictor's (document type `rfx`).
pub fn is_flac_archive(path: &Path) -> bool {
    let mut head = [0u8; 64];
    match File::open(path).and_then(|mut f| f.read(&mut head)) {
        Ok(n) => head[..n].windows(8).any(|w| w == b"matroska"),
        Err(_) => false,
    }
}

/// A stored 16-bit sample as the capture's own.
fn from_i16(fmt: Fmt, s: i16) -> i32 {
    let s = i32::from(s);
    match fmt {
        Fmt::S8 => s / 256,
        Fmt::U8 => s / 256 + 128,
        Fmt::U16 => s + 32768,
        Fmt::S16 | Fmt::Lds => s,
    }
}

/// A FLAC-mode archive opened for reading by sample index (what ld-decode
/// does): the FLAC frames of each track are found in the Matroska file, and
/// the frames asked for are decoded with claxon.
pub struct FlacArchive {
    file: File,
    meta: FlacMeta,
    /// `fLaC` and the STREAMINFO block, put before a frame to decode it.
    header: Vec<u8>,
    /// Samples a frame holds (a fixed block size).
    frame: usize,
    /// Offset and length of each frame, per track.
    frames: Vec<Vec<(u64, u64)>>,
    cache: std::sync::Mutex<std::collections::HashMap<(usize, usize), std::sync::Arc<Vec<i16>>>>,
}

/// Frames kept (a few thousand samples each).
const FRAME_CACHE: usize = 8192;

impl FlacArchive {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let idx = crate::ebml::scan_with(&file, None)?;
        let flac_tracks: Vec<u64> = idx.codecs.iter().filter(|(_, c)| c == "A_FLAC").map(|(n, _)| *n).collect();
        let att = idx.attachments.iter().find(|a| a.0 == "rfx.json").with_context(|| format!("{}: no rfx.json attachment (not an rfx archive)", path.display()))?;
        let mut buf = vec![0u8; att.2 as usize];
        read_exact_at(&file, &mut buf, att.1)?;
        let meta: FlacMeta = serde_json::from_slice(&buf).context("bad metadata")?;
        if meta.format != "rfx" || meta.codec != "flac" || meta.version != 1 {
            bail!("unsupported FLAC-mode archive ({} {} v{})", meta.format, meta.codec, meta.version);
        }
        if flac_tracks.len() != meta.tracks.len() {
            bail!("{} has {} FLAC tracks but its metadata lists {}", path.display(), flac_tracks.len(), meta.tracks.len());
        }
        let private = idx.private.iter().find(|(n, _)| Some(n) == flac_tracks.first()).map(|(_, p)| p.clone()).context("the FLAC track has no stream header")?;
        let b = private.strip_prefix(b"fLaC").unwrap_or(&private);
        if b.len() < 8 {
            bail!("the FLAC stream header is too short");
        }
        let len = (usize::from(b[1]) << 16) | (usize::from(b[2]) << 8) | usize::from(b[3]);
        if b.len() < 4 + len || len < 34 {
            bail!("the FLAC stream header is damaged");
        }
        let (min, max) = (u16::from_be_bytes([b[4], b[5]]), u16::from_be_bytes([b[6], b[7]]));
        if min != max || min == 0 {
            bail!("the FLAC tracks have a variable block size ({min} to {max}), which is not supported");
        }
        let mut header = b"fLaC".to_vec();
        header.push(b[0] | 0x80);
        header.extend_from_slice(&b[1..4 + len]);
        let frame = usize::from(min);
        let frames: Vec<Vec<(u64, u64)>> = flac_tracks.iter().map(|n| idx.blocks.iter().filter(|b| b.track == *n).map(|b| (b.off, b.len)).collect()).collect();
        for (k, f) in frames.iter().enumerate() {
            if f.len() as u64 != meta.tracks[k].words.div_ceil(frame as u64) {
                bail!("{}: track {k} has {} frames for {} samples", path.display(), f.len(), meta.tracks[k].words);
            }
        }
        Ok(Self { file, meta, header, frame, frames, cache: Default::default() })
    }

    pub fn tracks(&self) -> usize {
        self.meta.tracks.len()
    }

    pub fn name(&self, k: usize) -> &str {
        &self.meta.tracks[k].name
    }

    pub fn len(&self, k: usize) -> u64 {
        self.meta.tracks[k].words
    }

    pub fn fmt(&self, k: usize) -> Fmt {
        self.meta.tracks[k].fmt()
    }

    pub fn rate_hz(&self, k: usize) -> f64 {
        self.meta.tracks[k].rate_hz
    }

    fn decode_frame(&self, k: usize, f: usize) -> Result<std::sync::Arc<Vec<i16>>> {
        if let Some(v) = self.cache.lock().map_err(|_| anyhow::anyhow!("cache poisoned"))?.get(&(k, f)) {
            return Ok(std::sync::Arc::clone(v));
        }
        let (off, len) = self.frames[k][f];
        let mut stream = self.header.clone();
        let at = stream.len();
        stream.resize(at + len as usize, 0);
        read_exact_at(&self.file, &mut stream[at..], off)?;
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(stream)).context("FLAC header")?;
        let block = reader.blocks().read_next_or_eof(Vec::new()).with_context(|| format!("{}: frame {f} of track {k}", self.meta.tracks[k].name))?.context("empty FLAC frame")?;
        let want = self.frame.min((self.meta.tracks[k].words as usize).saturating_sub(f * self.frame));
        if block.duration() as usize != want {
            bail!("{}: frame {f} holds {} samples, not {want}", self.meta.tracks[k].name, block.duration());
        }
        let v = std::sync::Arc::new(block.channel(0).iter().map(|&s| s as i16).collect::<Vec<i16>>());
        let mut c = self.cache.lock().map_err(|_| anyhow::anyhow!("cache poisoned"))?;
        if c.len() >= FRAME_CACHE {
            c.clear();
        }
        c.insert((k, f), std::sync::Arc::clone(&v));
        Ok(v)
    }

    /// Samples `[start, start + out.len())` of track `k` on the scale of the
    /// original file; zero outside the track.
    pub fn read(&self, k: usize, start: i64, out: &mut [i32]) -> Result<()> {
        use rayon::prelude::*;
        if k >= self.tracks() {
            bail!("the archive has {} tracks: no track {k}", self.tracks());
        }
        out.fill(0);
        let words = self.len(k) as i64;
        let (lo, hi) = (start.max(0), (start + out.len() as i64).min(words));
        if hi <= lo {
            return Ok(());
        }
        let n = self.frame as i64;
        let frames: Vec<usize> = ((lo / n) as usize..=((hi - 1) / n) as usize).collect();
        let decoded: Vec<std::sync::Arc<Vec<i16>>> = frames.par_iter().map(|&f| self.decode_frame(k, f)).collect::<Result<_>>()?;
        let fmt = self.fmt(k);
        for (&f, v) in frames.iter().zip(&decoded) {
            let base = f as i64 * n;
            let (a, b) = (base.max(lo), (base + v.len() as i64).min(hi));
            for p in a..b {
                out[(p - start) as usize] = from_i16(fmt, v[(p - base) as usize]);
            }
        }
        Ok(())
    }
}
