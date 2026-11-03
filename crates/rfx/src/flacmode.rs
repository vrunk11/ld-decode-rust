//! FLAC mode (`pack --flac`, off by default): the codec of the tracks is FLAC
//! instead of the predictor, so that the file is an ordinary Matroska audio
//! file (`.mka` with `A_FLAC` tracks, one per capture, named after it) that
//! any audio or video program opens. The tracks do not predict each other here
//! (the archive is several times larger than with the predictor); what the
//! archive keeps besides the samples (names, hashes, formats, the sample rate,
//! the `.yaml` files) is one attachment, `rfx.json`.
//!
//! The samples are the 16-bit ones of [`crate::mflac`]; `unpack` gives back the
//! captures as they were, checked by their SHA-256. ffmpeg does the FLAC and
//! the Matroska work.
use crate::fmt::Fmt;
use crate::io::{read_exact_at, write_all_at};
use crate::mflac::to_i16;
use crate::pack::{convert_file, decode_compressed, encode_flac, file_stats, find_sidecars, hex, input_fmt, sidecar_rate_hz, OutFmt, PackOpts, SrcMeta};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Serialize, Deserialize)]
pub struct FlacMeta {
    pub format: String,
    pub codec: String,
    pub version: u32,
    pub tracks: Vec<SrcMeta>,
}

fn ffmpeg() -> String {
    std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into())
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

/// The inverse of [`to_i16`].
fn from_i16(fmt: Fmt, s: i16) -> i32 {
    let s = i32::from(s);
    match fmt {
        Fmt::S8 => s / 256,
        Fmt::U8 => s / 256 + 128,
        Fmt::U16 => s + 32768,
        Fmt::S16 | Fmt::Lds => s,
    }
}

/// Write the samples of a capture as 16-bit little-endian ones (the file itself
/// when it is one).
fn raw16(src: &Path, fmt: Fmt, work: &Path, i: usize) -> Result<PathBuf> {
    if fmt == Fmt::S16 {
        return Ok(src.to_path_buf());
    }
    let f = File::open(src)?;
    let words = fmt.samples(f.metadata()?.len());
    let dst = work.join(format!("{i}.s16"));
    let mut w = std::io::BufWriter::new(File::create(&dst)?);
    let mut buf = vec![0i32; 1 << 22];
    let mut at = 0u64;
    while at < words {
        let n = ((words - at) as usize).min(buf.len());
        fmt.load(&f, words, at as i64, &mut buf[..n])?;
        for &v in &buf[..n] {
            w.write_all(&to_i16(fmt, v).context("a sample does not fit 16 bits")?.to_le_bytes())?;
        }
        at += n as u64;
    }
    w.flush()?;
    Ok(dst)
}

fn hash_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Pack `inputs` (in that order) as FLAC tracks in `out`.
pub fn pack(inputs: &[PathBuf], out: &Path, o: &PackOpts) -> Result<()> {
    if inputs.is_empty() {
        bail!("no input");
    }
    let (conv, converted) = decode_compressed(inputs, out, o)?;
    let work = out.with_extension("flac-work");
    std::fs::create_dir_all(&work)?;
    let result = (|| -> Result<()> {
        let ff = ffmpeg();
        let mut tracks = Vec::new();
        let mut flacs = Vec::new();
        let mut hashes = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for (i, p) in conv.iter().enumerate() {
            let name = inputs[i].file_name().context("input has no file name")?.to_string_lossy().into_owned();
            if names.contains(&name) {
                bail!("two captures are named {name}: the archive keeps file names only");
            }
            names.push(name.clone());
            let fmt = input_fmt(p, &o.format)?;
            eprintln!("{name}: hashing and encoding");
            let st = file_stats(p, fmt)?;
            let words = fmt.samples(st.size);
            let covered = fmt.covered(st.size);
            let mut tail = vec![0u8; (st.size - covered) as usize];
            if !tail.is_empty() {
                read_exact_at(&File::open(p)?, &mut tail, covered)?;
            }
            let sidecars = find_sidecars(&inputs[i]);
            let rate_hz = sidecar_rate_hz(&sidecars).unwrap_or(o.rate_mhz * 1e6);
            let raw = raw16(p, fmt, &work, i)?;
            hashes.push(hash_file(&raw)?);
            let flac = work.join(format!("{i}.flac"));
            let status = Command::new(&ff)
                .args(["-v", "error", "-y", "-f", "s16le", "-ar", &format!("{}", (rate_hz / 1000.0).round() as u64), "-ac", "1", "-i"])
                .arg(&raw)
                .args(["-c:a", "flac", "-compression_level", "8"])
                .arg(&flac)
                .status()
                .with_context(|| format!("starting {ff}"))?;
            if !status.success() {
                bail!("ffmpeg could not encode {name}");
            }
            if raw != *p {
                let _ = std::fs::remove_file(&raw);
            }
            flacs.push(flac);
            tracks.push(SrcMeta {
                name,
                size: st.size,
                sha256: st.sha,
                words,
                tail,
                fmt: fmt.name().to_string(),
                sample_sha256: st.sample_sha,
                shift: st.shift,
                mean: st.mean,
                own: 0,
                solo: 0,
                taps: 0,
                ntop: 0,
                ttaps: 0,
                stride: 0,
                every: 0,
                lam_bits: 0,
                refs: Vec::new(),
                weights: Vec::new(),
                position: i,
                noise: None,
                sidecars,
                origin: (inputs[i] != *p).then(|| inputs[i].file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
                rate_hz,
            });
        }
        let meta = FlacMeta { format: "rfx".into(), codec: "flac".into(), version: 1, tracks };
        let json = work.join("rfx.json");
        std::fs::write(&json, serde_json::to_vec(&meta)?)?;
        let tmp = out.with_extension("rfx-tmp");
        let mut cmd = Command::new(&ff);
        cmd.args(["-v", "error", "-y"]);
        for f in &flacs {
            cmd.arg("-i").arg(f);
        }
        for i in 0..flacs.len() {
            cmd.args(["-map", &i.to_string()]);
        }
        cmd.args(["-c", "copy"]);
        for (i, t) in meta.tracks.iter().enumerate() {
            cmd.args([format!("-metadata:s:a:{i}"), format!("title={}", t.name)]);
        }
        cmd.arg("-attach").arg(&json).args(["-metadata:s:t:0", "mimetype=application/json", "-metadata:s:t:0", "filename=rfx.json", "-f", "matroska"]).arg(&tmp);
        if !cmd.status().with_context(|| format!("starting {ff}"))?.success() {
            let _ = std::fs::remove_file(&tmp);
            bail!("ffmpeg could not write {}", out.display());
        }
        if o.verify {
            for (i, want) in hashes.iter().enumerate() {
                if &decode_hash(&tmp, i)? != want {
                    let _ = std::fs::remove_file(&tmp);
                    bail!("{}: the samples differ after encoding with ffmpeg", meta.tracks[i].name);
                }
            }
        }
        let total: u64 = std::fs::metadata(&tmp)?.len();
        let raw: u64 = meta.tracks.iter().map(|t| t.size).sum();
        if out.exists() {
            std::fs::remove_file(out)?;
        }
        std::fs::rename(&tmp, out)?;
        eprintln!("total: {raw} -> {total} bytes ({:.1}%), FLAC tracks", 100.0 * total as f64 / raw.max(1) as f64);
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&work);
    if let Some(w) = converted {
        let _ = std::fs::remove_dir_all(w);
    }
    result
}

/// SHA-256 of the 16-bit samples of audio stream `i` of `path`, decoded.
fn decode_hash(path: &Path, i: usize) -> Result<String> {
    let mut dec = Command::new(ffmpeg())
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", &format!("0:a:{i}"), "-f", "s16le", "-acodec", "pcm_s16le", "-"])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut out = dec.stdout.take().context("ffmpeg has no output")?;
    let mut h = Sha256::new();
    let mut chunk = vec![0u8; 1 << 22];
    loop {
        let n = out.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        h.update(&chunk[..n]);
    }
    if !dec.wait()?.success() {
        bail!("ffmpeg could not decode stream {i} of {}", path.display());
    }
    Ok(hex(&h.finalize()))
}

/// What the attachment `rfx.json` holds.
pub fn read_meta(path: &Path) -> Result<FlacMeta> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let idx = crate::ebml::scan_with(&file, None)?;
    let att = idx.attachments.iter().find(|a| a.0 == "rfx.json").with_context(|| format!("{}: no rfx.json attachment (not an rfx archive)", path.display()))?;
    let mut buf = vec![0u8; att.2 as usize];
    read_exact_at(&file, &mut buf, att.1)?;
    let meta: FlacMeta = serde_json::from_slice(&buf).context("bad metadata")?;
    if meta.format != "rfx" || meta.codec != "flac" || meta.version != 1 {
        bail!("unsupported FLAC-mode archive ({} {} v{})", meta.format, meta.codec, meta.version);
    }
    Ok(meta)
}

pub fn info(path: &Path) -> Result<()> {
    let meta = read_meta(path)?;
    println!("{}: {} tracks, FLAC", path.display(), meta.tracks.len());
    for (k, sm) in meta.tracks.iter().enumerate() {
        println!("  {k}: {}  {} bytes  {} samples at {} Hz  format {}", sm.name, sm.size, sm.words, sm.rate_hz, sm.fmt);
        println!("     sha256 {}", sm.sha256);
        let side: Vec<&str> = sm.sidecars.iter().map(|s| s.name.as_str()).collect();
        println!("     on the command line: {}  sidecars {:?}", sm.position, side);
    }
    Ok(())
}

pub fn meta_yaml(path: &Path) -> Result<String> {
    let meta = read_meta(path)?;
    let tracks: Vec<serde_json::Value> = meta
        .tracks
        .iter()
        .enumerate()
        .map(|(k, sm)| {
            let mut v = serde_json::to_value(sm).unwrap_or_default();
            if let Some(m) = v.as_object_mut() {
                m.insert("track".into(), k.into());
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
                for key in ["own", "solo", "taps", "ntop", "ttaps", "stride", "every", "lam_bits", "refs", "weights", "noise"] {
                    m.remove(key);
                }
            }
            v
        })
        .collect();
    let doc = serde_json::json!({ "format": meta.format, "codec": meta.codec, "version": meta.version, "tracks": tracks });
    let mut out = String::new();
    crate::pack::yaml_value(&doc, 0, &mut out);
    Ok(out)
}

/// Write the `.yaml` sidecars and `<archive>.yaml` (see `pack::unpack_meta`).
pub fn unpack_meta(path: &Path, outdir: &Path, only: Option<usize>) -> Result<()> {
    let meta = read_meta(path)?;
    std::fs::create_dir_all(outdir)?;
    for (k, sm) in meta.tracks.iter().enumerate() {
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

/// Give the captures back as they were, each checked by its SHA-256; with
/// `target`, then in that format (checked by the hash of the samples).
pub fn unpack(path: &Path, outdir: &Path, only: Option<usize>, target: Option<OutFmt>) -> Result<()> {
    let meta = read_meta(path)?;
    std::fs::create_dir_all(outdir)?;
    for (k, sm) in meta.tracks.iter().enumerate() {
        if only.is_some_and(|t| t != k) {
            continue;
        }
        let orig = sm.fmt();
        let name = Path::new(&sm.name).file_name().context("bad name in the archive")?;
        let dst = outdir.join(name);
        let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&dst)?;
        f.set_len(sm.size)?;
        let mut dec = Command::new(ffmpeg())
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-map", &format!("0:a:{k}"), "-f", "s16le", "-acodec", "pcm_s16le", "-"])
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", ffmpeg()))?;
        let mut out = dec.stdout.take().context("ffmpeg has no output")?;
        let mut bytes = vec![0u8; 1 << 22];
        let (mut held, mut at) = (0usize, 0u64);
        loop {
            let n = out.read(&mut bytes[held..])?;
            held += n;
            if n == 0 || held == bytes.len() {
                let whole = held / 2;
                let vals: Vec<i32> = (0..whole).map(|i| from_i16(orig, i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))).collect();
                if at + vals.len() as u64 > sm.words {
                    bail!("{}: more samples than were packed", sm.name);
                }
                orig.store(&f, at, &vals)?;
                at += whole as u64;
                bytes.copy_within(2 * whole..held, 0);
                held -= 2 * whole;
                if n == 0 {
                    break;
                }
            }
        }
        if !dec.wait()?.success() {
            bail!("ffmpeg could not decode {}", sm.name);
        }
        if at != sm.words {
            bail!("{}: {} samples decoded, {} packed", sm.name, at, sm.words);
        }
        if !sm.tail.is_empty() {
            write_all_at(&f, &sm.tail, orig.covered(sm.size))?;
        }
        drop(f);
        let got = hash_file(&dst)?;
        if got != sm.sha256 {
            bail!("{}: SHA-256 differs from the original ({got} against {})", sm.name, sm.sha256);
        }
        eprintln!("{}: {} bytes, SHA-256 matches", sm.name, sm.size);
        for sc in &sm.sidecars {
            let name = Path::new(&sc.name).file_name().context("bad sidecar name in the archive")?;
            std::fs::write(outdir.join(name), &sc.data)?;
        }
        match target {
            Some(OutFmt::Raw(t)) if t != orig => convert_file(&dst, sm, t, outdir)?,
            Some(t @ (OutFmt::Flac | OutFmt::Ldf)) => encode_flac(&dst, sm, t == OutFmt::Ldf, outdir)?,
            _ => {}
        }
    }
    Ok(())
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
