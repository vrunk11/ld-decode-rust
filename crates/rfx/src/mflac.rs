//! The tracks of an archive as the channels of FLAC files, to look at in an
//! audio program: one channel per track, 8 channels a file (FLAC's limit), the
//! sample rate the RF rate in kHz (75 MHz -> 75000 Hz, as ld-decode reads it
//! from a `.flac`/`.ldf`). A shorter track is padded with zeros; the tracks
//! are not aligned on each other (each starts where its capture did).
//! The samples come from the streaming reader, then through ffmpeg; the file
//! is decoded again and compared with what was sent.
use crate::fmt::Fmt;
use crate::reader::Archive;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// FLAC's most channels in a file.
const CHANNELS: usize = 8;
/// Frames read at a time.
const FRAMES: usize = 1 << 20;

/// A stored sample as a 16-bit one, or `None` when it does not fit.
pub(crate) fn to_i16(fmt: Fmt, v: i32) -> Option<i16> {
    let x = match fmt {
        Fmt::S8 => v * 256,
        Fmt::U8 => (v - 128) * 256,
        Fmt::U16 => v - 32768,
        Fmt::S16 | Fmt::Lds => v,
    };
    i16::try_from(x).ok()
}

pub fn export(archive: &Path, outdir: &Path, only: Option<usize>) -> Result<()> {
    let a = Archive::open(archive)?;
    let tracks: Vec<usize> = match only {
        Some(t) if t >= a.tracks() => bail!("{} has {} tracks: no track {t}", archive.display(), a.tracks()),
        Some(t) => vec![t],
        None => (0..a.tracks()).collect(),
    };
    std::fs::create_dir_all(outdir)?;
    let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let stem = archive.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "archive".into());
    let groups: Vec<&[usize]> = tracks.chunks(CHANNELS).collect();
    for group in &groups {
        let name = if groups.len() == 1 && only.is_none() { format!("{stem}.flac") } else { format!("{stem}_{}-{}.flac", group[0], group[group.len() - 1]) };
        let dst = outdir.join(&name);
        let rate = a.rate_hz(group[0]);
        let rate_arg = if rate > 0.0 { format!("{}", (rate / 1000.0).round() as u64) } else { "40000".to_string() };
        if group.iter().any(|&k| (a.rate_hz(k) - rate).abs() > 1.0) {
            eprintln!("{name}: the tracks were not captured at one rate; the file says {rate_arg} Hz (track {})", group[0]);
        }
        let mut enc = Command::new(&ffmpeg)
            .args(["-v", "error", "-y", "-f", "s16le", "-ar", &rate_arg, "-ac", &group.len().to_string(), "-i", "-", "-c:a", "flac"])
            .arg(&dst)
            .stdin(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {ffmpeg}"))?;
        let mut stdin = enc.stdin.take().context("ffmpeg has no input")?;
        let frames = group.iter().map(|&k| a.len(k)).max().unwrap_or(0);
        let fmts: Vec<Fmt> = group.iter().map(|&k| a.fmt(k)).collect();
        let mut sent = Sha256::new();
        let mut bufs: Vec<Vec<i32>> = group.iter().map(|_| vec![0i32; FRAMES]).collect();
        let mut bytes = Vec::with_capacity(FRAMES * group.len() * 2);
        let mut at = 0u64;
        while at < frames {
            let n = ((frames - at) as usize).min(FRAMES);
            for (buf, &k) in bufs.iter_mut().zip(group.iter()) {
                a.read(k, at as i64, &mut buf[..n])?;
            }
            bytes.clear();
            for i in 0..n {
                for (c, buf) in bufs.iter().enumerate() {
                    let past_end = at + i as u64 >= a.len(group[c]);
                    let s = if past_end { 0 } else { to_i16(fmts[c], buf[i]).with_context(|| format!("track {}: a sample does not fit 16 bits", group[c]))? };
                    bytes.extend_from_slice(&s.to_le_bytes());
                }
            }
            sent.update(&bytes);
            stdin.write_all(&bytes).context("ffmpeg stopped reading")?;
            at += n as u64;
        }
        drop(stdin);
        if !enc.wait()?.success() {
            bail!("ffmpeg could not encode {name}");
        }
        // Decode it back and compare.
        let mut dec = Command::new(&ffmpeg).args(["-v", "error", "-i"]).arg(&dst).args(["-f", "s16le", "-acodec", "pcm_s16le", "-"]).stdout(Stdio::piped()).spawn()?;
        let mut out = dec.stdout.take().context("ffmpeg has no output")?;
        let mut back = Sha256::new();
        let mut chunk = vec![0u8; 1 << 22];
        loop {
            let n = out.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            back.update(&chunk[..n]);
        }
        if !dec.wait()?.success() || back.finalize() != sent.finalize() {
            bail!("{name}: the samples differ after encoding with ffmpeg");
        }
        let list: Vec<String> = group.iter().map(|k| format!("{k}: {}", a.name(*k))).collect();
        println!("{}  {} Hz, {} channels, {} frames  [{}]", dst.display(), rate_arg, group.len(), frames, list.join(", "));
    }
    Ok(())
}
