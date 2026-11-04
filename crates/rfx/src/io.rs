//! Positioned file access: the samples of a source are read and written by
//! index from many threads at once, so no handle has a cursor of its own.
use crate::fmt::Fmt;
use std::fs::File;
use std::io;

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, off)
}
#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, off)
}
#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], off: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(f, buf, off)
}
#[cfg(unix)]
fn write_at(f: &File, buf: &[u8], off: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(f, buf, off)
}

/// Fill `buf` from byte `off`, or fail at end of file.
pub fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = read_at(f, buf, off)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

pub fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let n = write_at(f, buf, off)?;
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}

/// `n` samples from sample index `start` (zero outside the file's `words`
/// samples), each shifted right by `shift`.
pub fn read_words(f: &File, fmt: Fmt, words: u64, start: i64, n: usize, shift: u32) -> io::Result<Vec<i32>> {
    let mut out = vec![0i32; n];
    fmt.load(f, words, start, &mut out)?;
    if shift > 0 {
        out.iter_mut().for_each(|v| *v >>= shift);
    }
    Ok(out)
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Like [`read_words`] into `out` (zeros outside the file), with `sub`
/// subtracted from each sample read.
pub fn read_words_into(f: &File, fmt: Fmt, words: u64, start: i64, out: &mut [i32], shift: u32, sub: i32) -> io::Result<()> {
    out.fill(0);
    let lo = start.max(0);
    let hi = (start + out.len() as i64).min(words as i64);
    if hi <= lo {
        return Ok(());
    }
    let base = (lo - start) as usize;
    let n = (hi - lo) as usize;
    fmt.load(f, words, lo, &mut out[base..base + n])?;
    for v in &mut out[base..base + n] {
        *v = (*v >> shift) - sub;
    }
    Ok(())
}

/// Add the samples `[lo, hi)` of the file, shifted right by `shift` and minus
/// `sub`, to `dst` (`dst[0]` for sample `lo`).
pub fn add_words(f: &File, fmt: Fmt, words: u64, lo: i64, hi: i64, shift: u32, sub: i32, dst: &mut [f32]) -> io::Result<()> {
    add_words_scaled(f, fmt, words, lo, hi, shift, sub, 1.0, dst)
}

/// Like [`add_words`] with each term multiplied by `scale` (a product rounded,
/// then added: no fused multiply-add).
#[allow(clippy::too_many_arguments)]
pub fn add_words_scaled(f: &File, fmt: Fmt, words: u64, lo: i64, hi: i64, shift: u32, sub: i32, scale: f32, dst: &mut [f32]) -> io::Result<()> {
    debug_assert!(lo >= 0 && hi - lo == dst.len() as i64);
    SCRATCH.with(|s| {
        let mut buf = s.borrow_mut();
        buf.resize(dst.len(), 0);
        fmt.load(f, words, lo, &mut buf)?;
        for (o, &w) in dst.iter_mut().zip(buf.iter()) {
            let v = (w >> shift) - sub;
            *o += scale * v as f32;
        }
        Ok(())
    })
}
