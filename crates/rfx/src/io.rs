//! Positioned file access: the samples of a source are read and written by
//! index from many threads at once, so no handle has a cursor of its own.
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

/// `n` 16-bit little-endian words from word index `start` (zero outside the
/// file's `words` words), each shifted right by `shift`.
pub fn read_words(f: &File, words: u64, start: i64, n: usize, shift: u32) -> io::Result<Vec<i32>> {
    let mut out = vec![0i32; n];
    let lo = start.max(0);
    let hi = (start + n as i64).min(words as i64);
    if hi <= lo {
        return Ok(out);
    }
    let mut buf = vec![0u8; (hi - lo) as usize * 2];
    read_exact_at(f, &mut buf, lo as u64 * 2)?;
    let base = (lo - start) as usize;
    for (o, w) in out[base..].iter_mut().zip(buf.chunks_exact(2)) {
        *o = (i16::from_le_bytes([w[0], w[1]]) >> shift) as i32;
    }
    Ok(out)
}
