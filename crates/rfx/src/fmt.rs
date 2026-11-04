//! Input formats whose bytes and samples map one to one: a capture is read as
//! a series of integer samples by index, and written back by index, so that
//! `unpack` gives the very bytes `pack` was given. The formats are the raw
//! ones of ld-decode (`--format`): signed or unsigned 16 bits, signed or
//! unsigned 8 bits, and the packed 10-bit `.lds` of the Domesday Duplicator
//! (four samples in five bytes). Bytes past the last whole group are kept
//! by the metadata.
use crate::io::{read_exact_at, write_all_at};
use anyhow::{bail, Result};
use std::fs::File;
use std::io;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fmt {
    S16,
    U16,
    S8,
    U8,
    Lds,
}

/// The 10-bit value of `.lds` is stored with the offset ld-decode removes.
const LDS_ZERO: i32 = 512;

impl Fmt {
    /// The `--format` value of ld-decode.
    pub fn name(self) -> &'static str {
        match self {
            Fmt::S16 => "s16",
            Fmt::U16 => "r16",
            Fmt::S8 => "s8",
            Fmt::U8 => "r8",
            Fmt::Lds => "lds",
        }
    }

    pub fn parse(s: &str) -> Option<Fmt> {
        Some(match s {
            "s16" => Fmt::S16,
            "r16" | "u16" => Fmt::U16,
            "s8" => Fmt::S8,
            "r8" | "u8" => Fmt::U8,
            "lds" => Fmt::Lds,
            _ => return None,
        })
    }

    /// Guess from the extension, as ld-decode does.
    pub fn from_path(p: &std::path::Path) -> Option<Fmt> {
        Fmt::parse(&p.extension()?.to_string_lossy().to_ascii_lowercase())
    }

    /// (bytes, samples) of one group.
    fn group(self) -> (u64, u64) {
        match self {
            Fmt::S16 | Fmt::U16 => (2, 1),
            Fmt::S8 | Fmt::U8 => (1, 1),
            Fmt::Lds => (5, 4),
        }
    }

    /// Samples in a file of `size` bytes.
    pub fn samples(self, size: u64) -> u64 {
        let (b, s) = self.group();
        size / b * s
    }

    /// Bytes of a file of `size` bytes that its samples cover (the rest is
    /// the tail).
    pub fn covered(self, size: u64) -> u64 {
        let (b, s) = self.group();
        self.samples(size) / s * b
    }

    /// Samples `[start, start + out.len())` of a file of `count` samples,
    /// zero outside the file.
    pub fn load(self, f: &File, count: u64, start: i64, out: &mut [i32]) -> io::Result<()> {
        out.fill(0);
        let lo = start.max(0);
        let hi = (start + out.len() as i64).min(count as i64);
        if hi <= lo {
            return Ok(());
        }
        let base = (lo - start) as usize;
        let (gb, gs) = self.group();
        let (g0, g1) = (lo as u64 / gs, (hi as u64).div_ceil(gs));
        let mut buf = vec![0u8; ((g1 - g0) * gb) as usize];
        read_exact_at(f, &mut buf, g0 * gb)?;
        let first = lo as u64 - g0 * gs;
        let n = (hi - lo) as usize;
        let dst = &mut out[base..base + n];
        match self {
            Fmt::S16 => {
                for (o, w) in dst.iter_mut().zip(buf.chunks_exact(2)) {
                    *o = i32::from(i16::from_le_bytes([w[0], w[1]]));
                }
            }
            Fmt::U16 => {
                for (o, w) in dst.iter_mut().zip(buf.chunks_exact(2)) {
                    *o = i32::from(u16::from_le_bytes([w[0], w[1]]));
                }
            }
            Fmt::S8 => {
                for (o, &w) in dst.iter_mut().zip(&buf) {
                    *o = i32::from(w as i8);
                }
            }
            Fmt::U8 => {
                for (o, &w) in dst.iter_mut().zip(&buf) {
                    *o = i32::from(w);
                }
            }
            Fmt::Lds => {
                let mut i = 0usize;
                let mut skip = first as usize;
                for g in buf.chunks_exact(5) {
                    let t = unpack_lds(g);
                    for &v in &t[skip..] {
                        if i == n {
                            break;
                        }
                        dst[i] = v;
                        i += 1;
                    }
                    skip = 0;
                }
            }
        }
        Ok(())
    }

    /// Write the samples `vals` from sample index `start` (a whole number of
    /// groups, or ending at the last sample of the file).
    pub fn store(self, f: &File, start: u64, vals: &[i32]) -> Result<()> {
        let (gb, gs) = self.group();
        if start % gs != 0 {
            bail!("a segment must start on a group of samples");
        }
        let mut raw = Vec::with_capacity(vals.len() * gb as usize / gs as usize + 5);
        match self {
            Fmt::S16 => vals.iter().for_each(|&v| raw.extend_from_slice(&(v as i16).to_le_bytes())),
            Fmt::U16 => vals.iter().for_each(|&v| raw.extend_from_slice(&(v as u16).to_le_bytes())),
            Fmt::S8 => vals.iter().for_each(|&v| raw.push(v as i8 as u8)),
            Fmt::U8 => vals.iter().for_each(|&v| raw.push(v as u8)),
            Fmt::Lds => {
                if vals.len() % 4 != 0 {
                    bail!("an .lds segment must hold whole groups of four samples");
                }
                if let Some(v) = vals.iter().find(|&&v| !lds_ok(v)) {
                    bail!("the sample {v} does not fit the 10 bits of an .lds file");
                }
                for g in vals.chunks_exact(4) {
                    raw.extend_from_slice(&pack_lds([g[0], g[1], g[2], g[3]]));
                }
            }
        }
        write_all_at(f, &raw, start / gs * gb)?;
        Ok(())
    }
}

/// The four samples of a group, on the 16-bit scale of ld-decode (and of the
/// Duplicator's `.s16`): the 10-bit value minus 512, times 64.
fn unpack_lds(b: &[u8]) -> [i32; 4] {
    [
        (((i32::from(b[0]) << 2) | (i32::from(b[1]) >> 6)) - LDS_ZERO) << 6,
        ((((i32::from(b[1]) & 0x3F) << 4) | (i32::from(b[2]) >> 4)) - LDS_ZERO) << 6,
        ((((i32::from(b[2]) & 0x0F) << 6) | (i32::from(b[3]) >> 2)) - LDS_ZERO) << 6,
        ((((i32::from(b[3]) & 0x03) << 8) | i32::from(b[4])) - LDS_ZERO) << 6,
    ]
}

/// Whether `v` is a sample an `.lds` file can hold.
fn lds_ok(v: i32) -> bool {
    v % 64 == 0 && (-512..512).contains(&(v / 64))
}

fn pack_lds(v: [i32; 4]) -> [u8; 5] {
    let t = v.map(|x| ((x >> 6) + LDS_ZERO) as u32 & 0x3FF);
    [(t[0] >> 2) as u8, (((t[0] & 3) << 6) | (t[1] >> 4)) as u8, (((t[1] & 15) << 4) | (t[2] >> 6)) as u8, (((t[2] & 63) << 2) | (t[3] >> 8)) as u8, (t[3] & 255) as u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lds_groups_roundtrip() {
        let mut x = 5u32;
        for _ in 0..1000 {
            let b: Vec<u8> = (0..5)
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    (x >> 24) as u8
                })
                .collect();
            assert_eq!(pack_lds(unpack_lds(&b)).to_vec(), b);
        }
    }

    #[test]
    fn every_format_loads_what_it_stored() {
        let dir = std::env::temp_dir().join(format!("rfx_fmt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for fmt in [Fmt::S16, Fmt::U16, Fmt::S8, Fmt::U8, Fmt::Lds] {
            let p = dir.join(format!("t.{}", fmt.name()));
            let f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&p).unwrap();
            let n = 4096usize;
            let (lo, hi) = match fmt {
                Fmt::S16 => (-32768, 32767),
                Fmt::U16 => (0, 65535),
                Fmt::S8 => (-128, 127),
                Fmt::U8 => (0, 255),
                Fmt::Lds => (-512, 511),
            };
            let mut x = 9u32;
            let vals: Vec<i32> = (0..n)
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    let v = lo + ((x >> 8) % (hi - lo + 1) as u32) as i32;
                    if fmt == Fmt::Lds {
                        v * 64
                    } else {
                        v
                    }
                })
                .collect();
            fmt.store(&f, 0, &vals[..2048]).unwrap();
            fmt.store(&f, 2048, &vals[2048..]).unwrap();
            let size = f.metadata().unwrap().len();
            assert_eq!(fmt.samples(size), n as u64);
            let mut back = vec![0i32; 100];
            fmt.load(&f, n as u64, 1001, &mut back).unwrap();
            assert_eq!(back, vals[1001..1101], "{}", fmt.name());
            let mut edge = vec![7i32; 10];
            fmt.load(&f, n as u64, n as i64 - 4, &mut edge).unwrap();
            assert_eq!(&edge[..4], &vals[n - 4..], "{}", fmt.name());
            assert_eq!(&edge[4..], &[0; 6]);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
