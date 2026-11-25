//! A FLAC capture (`.flac`, or `.ldf` holding FLAC in Ogg or bare) read by
//! sample index, for `--tbc read`, which reads the RF at positions it picks.
//!
//! The file is scanned once for its frames (a few seconds a gigabyte): the
//! offset of each, and the frame's first sample is its number times the block
//! size (a fixed block size is required, which every FLAC encoder gives a
//! capture by default). A read decodes the frames that hold the samples it asks
//! for with claxon, and keeps the latest ones.
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

/// Frames kept decoded (a few thousand samples each).
const CACHE: usize = 4096;

/// Where a frame is: pieces of the file (more than one only when an Ogg packet
/// continues on the next page).
type Piece = (u64, u32);

pub struct FlacIndex {
    file: File,
    /// `fLaC` and the STREAMINFO block (marked last), put before a frame to decode it.
    header: Vec<u8>,
    /// Samples a frame holds, except the last one's.
    block: usize,
    total: u64,
    rate_hz: u32,
    frames: Vec<Piece>,
    /// The pieces after the first, of the frames that have more.
    more: HashMap<usize, Vec<Piece>>,
    cache: Mutex<HashMap<usize, Arc<Vec<i32>>>>,
}

fn crc8(bytes: &[u8]) -> u8 {
    let mut c = 0u8;
    for &b in bytes {
        c ^= b;
        for _ in 0..8 {
            c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
        }
    }
    c
}

/// A frame header at the start of `b`: its number (fixed block size) or first
/// sample (variable), and whether the block size is variable. The CRC-8 has to
/// match, which a false sync code seldom does.
fn frame_header(b: &[u8]) -> Option<(u64, bool)> {
    if b.len() < 16 || b[0] != 0xFF || b[1] & 0xFE != 0xF8 {
        return None;
    }
    let (size, rate) = (b[2] >> 4, b[2] & 15);
    if size == 0 || rate == 15 || b[3] & 1 != 0 || (b[3] >> 1) & 7 == 3 {
        return None;
    }
    let first = b[4];
    let (mut n, extra) = match first {
        0x00..=0x7F => (u64::from(first), 0),
        0xC0..=0xDF => (u64::from(first & 0x1F), 1),
        0xE0..=0xEF => (u64::from(first & 0x0F), 2),
        0xF0..=0xF7 => (u64::from(first & 0x07), 3),
        0xF8..=0xFB => (u64::from(first & 0x03), 4),
        0xFC..=0xFD => (u64::from(first & 0x01), 5),
        0xFE => (0, 6),
        _ => return None,
    };
    let mut at = 5;
    for _ in 0..extra {
        if b[at] & 0xC0 != 0x80 {
            return None;
        }
        n = (n << 6) | u64::from(b[at] & 0x3F);
        at += 1;
    }
    at += match size {
        6 => 1,
        7 => 2,
        _ => 0,
    };
    at += match rate {
        12 => 1,
        13 | 14 => 2,
        _ => 0,
    };
    (at < b.len() && crc8(&b[..at]) == b[at]).then_some((n, b[1] & 1 == 1))
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let got = crate::stack::read_at(file, &mut buf[filled..], offset + filled as u64)?;
        if got == 0 {
            bail!("the file ends inside a FLAC frame");
        }
        filled += got;
    }
    Ok(())
}

impl FlacIndex {
    /// Open `path`, whose container (`ogg` or bare FLAC) starts at `offset`.
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {path}"))?;
        let mut head = vec![0u8; 64 * 1024];
        let n = crate::stack::read_at(&file, &mut head, 0)?;
        head.truncate(n);
        let find = |magic: &[u8]| head.windows(magic.len()).position(|w| w == magic);
        let (ogg, flac) = (find(b"OggS"), find(b"fLaC"));
        let (is_ogg, offset) = match (ogg, flac) {
            (Some(o), Some(f)) if o <= f => (true, o),
            (Some(_), Some(f)) => (false, f),
            (Some(o), None) => (true, o),
            (None, Some(f)) => (false, f),
            (None, None) => bail!("{path}: no FLAC stream found"),
        };
        let (header, frames, more) = if is_ogg { scan_ogg(&file, path, offset as u64)? } else { scan_flac(path, offset as u64)? };
        // STREAMINFO: block sizes, then the rate, channels, bits and length.
        let d = header.get(8..42).filter(|d| d.len() == 34).context("the FLAC stream header is damaged")?;
        let (min, max) = (u16::from_be_bytes([d[0], d[1]]), u16::from_be_bytes([d[2], d[3]]));
        if min != max || min == 0 {
            bail!("{path}: the FLAC stream has a variable block size ({min} to {max}), which --tbc read does not read by index");
        }
        let rate_hz = (u32::from(d[10]) << 12) | (u32::from(d[11]) << 4) | u32::from(d[12] >> 4);
        let (channels, bits) = (((d[12] >> 1) & 7) + 1, (((d[12] & 1) << 4) | (d[13] >> 4)) + 1);
        if channels != 1 || bits > 16 {
            bail!("{path}: {channels} channels of {bits} bits, not a 16-bit mono capture");
        }
        let block = usize::from(min);
        let total = (u64::from(d[13] & 15) << 32) | u64::from(u32::from_be_bytes([d[14], d[15], d[16], d[17]]));
        let mut s = Self { file, header, block, total, rate_hz, frames, more, cache: Default::default() };
        if s.frames.is_empty() {
            bail!("{path}: no FLAC frame found");
        }
        if total == 0 {
            // The length was not written: the last frame says how long it is.
            let last = s.frames.len() - 1;
            let v = s.decode(last)?;
            s.total = (last * block + v.len()) as u64;
        } else if (s.frames.len() as u64) != total.div_ceil(block as u64) {
            bail!("{path}: {} frames for {total} samples of {block}: a damaged or cut file", s.frames.len());
        }
        Ok(s)
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    #[allow(dead_code)]
    pub fn rate_hz(&self) -> u32 {
        self.rate_hz
    }

    fn decode(&self, f: usize) -> Result<Arc<Vec<i32>>> {
        if let Some(v) = self.cache.lock().map_err(|_| anyhow::anyhow!("cache poisoned"))?.get(&f) {
            return Ok(Arc::clone(v));
        }
        let mut stream = self.header.clone();
        for &(off, len) in std::iter::once(&self.frames[f]).chain(self.more.get(&f).into_iter().flatten()) {
            let at = stream.len();
            stream.resize(at + len as usize, 0);
            read_exact_at(&self.file, &mut stream[at..], off)?;
        }
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(stream)).context("FLAC header")?;
        let block = reader.blocks().read_next_or_eof(Vec::new()).with_context(|| format!("FLAC frame {f}"))?.context("empty FLAC frame")?;
        let want = self.block.min((self.total as usize).saturating_sub(f * self.block));
        if (block.duration() as usize == self.block && block.time() != (f * self.block) as u64) || (self.total > 0 && block.duration() as usize != want) {
            bail!("FLAC frame {f} starts at sample {} and holds {}: not frame {f} of a fixed block size {}", block.time(), block.duration(), self.block);
        }
        let v = Arc::new(block.channel(0).to_vec());
        let mut c = self.cache.lock().map_err(|_| anyhow::anyhow!("cache poisoned"))?;
        if c.len() >= CACHE {
            c.clear();
        }
        c.insert(f, Arc::clone(&v));
        Ok(v)
    }

    /// Samples `[start, start + out.len())`; zero outside the capture.
    pub fn read(&self, start: i64, out: &mut [i32]) -> Result<()> {
        use rayon::prelude::*;
        out.fill(0);
        let (lo, hi) = (start.max(0), (start + out.len() as i64).min(self.total as i64));
        if hi <= lo {
            return Ok(());
        }
        let n = self.block as i64;
        let frames: Vec<usize> = ((lo / n) as usize..=((hi - 1) / n) as usize).collect();
        let decoded: Vec<Arc<Vec<i32>>> = frames.par_iter().map(|&f| self.decode(f)).collect::<Result<_>>()?;
        for (&f, v) in frames.iter().zip(&decoded) {
            let base = f as i64 * n;
            let (a, b) = (base.max(lo), (base + v.len() as i64).min(hi));
            for p in a..b {
                out[(p - start) as usize] = v[(p - base) as usize];
            }
        }
        Ok(())
    }
}

/// A FLAC capture the index cannot read (a variable block size, stereo, an
/// index that does not add up): read in chunks instead. Each chunk of about a
/// second of video is decoded in one go by the ordinary sequential source into a
/// buffer, and a few are kept, so what the correction asks for is served like a
/// raw file's. A read far from the decoder restarts it (slower, but it works).
pub struct FlacWindow {
    state: Mutex<WindowState>,
    total: u64,
}

struct WindowState {
    src: Box<dyn crate::reader::SampleSource>,
    /// The sample the source returns next.
    next: u64,
    chunks: HashMap<u64, Arc<Vec<i16>>>,
    /// Chunks in the order they were decoded, oldest first.
    order: std::collections::VecDeque<u64>,
}

/// Samples a chunk holds (about a second at 33 MHz, a third of a second at 100 MHz).
const WINDOW_CHUNK: u64 = 1 << 25;
/// Chunks kept.
const WINDOW_KEPT: usize = 4;

impl FlacWindow {
    pub fn open(path: &str, format: crate::reader::SampleFormat) -> Result<Self> {
        let mut src = crate::reader::open_input(path, format)?.source;
        let total = match streaminfo_total(path) {
            Some(t) if t > 0 => t,
            _ => {
                // The length was not written: count the samples once.
                tracing::info!("{path}: its length is not in the FLAC header, counting the samples");
                let mut buf = vec![0f32; 1 << 20];
                let mut n = 0u64;
                loop {
                    let got = src.read(&mut buf)?;
                    n += got as u64;
                    if got < buf.len() {
                        break;
                    }
                }
                src.seek_samples(0)?;
                n
            }
        };
        Ok(Self { state: Mutex::new(WindowState { src, next: 0, chunks: HashMap::new(), order: Default::default() }), total })
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    fn chunk(&self, c: u64) -> Result<Arc<Vec<i16>>> {
        let mut st = self.state.lock().map_err(|_| anyhow::anyhow!("window poisoned"))?;
        if let Some(v) = st.chunks.get(&c) {
            return Ok(Arc::clone(v));
        }
        let first = c * WINDOW_CHUNK;
        if st.next != first {
            st.src.seek_samples(first)?;
            st.next = first;
        }
        let want = WINDOW_CHUNK.min(self.total.saturating_sub(first)) as usize;
        let mut buf = vec![0f32; want];
        let mut filled = 0;
        while filled < want {
            let got = st.src.read(&mut buf[filled..])?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        st.next = first + filled as u64;
        let v = Arc::new(buf[..filled].iter().map(|&s| s as i16).collect::<Vec<i16>>());
        if st.order.len() >= WINDOW_KEPT {
            if let Some(old) = st.order.pop_front() {
                st.chunks.remove(&old);
            }
        }
        st.order.push_back(c);
        st.chunks.insert(c, Arc::clone(&v));
        Ok(v)
    }

    /// Samples `[start, start + out.len())`; zero outside the capture.
    pub fn read(&self, start: i64, out: &mut [i32]) -> Result<()> {
        out.fill(0);
        let (lo, hi) = (start.max(0) as u64, ((start + out.len() as i64).max(0) as u64).min(self.total));
        if hi <= lo {
            return Ok(());
        }
        for c in lo / WINDOW_CHUNK..=(hi - 1) / WINDOW_CHUNK {
            let v = self.chunk(c)?;
            let base = c * WINDOW_CHUNK;
            let (a, b) = (base.max(lo), (base + v.len() as u64).min(hi));
            for p in a..b {
                out[(p as i64 - start) as usize] = i32::from(v[(p - base) as usize]);
            }
        }
        Ok(())
    }
}

/// The length in the STREAMINFO of a FLAC (bare or in Ogg), 0 when not written.
fn streaminfo_total(path: &str) -> Option<u64> {
    let mut head = vec![0u8; 64 * 1024];
    let n = crate::stack::read_at(&File::open(path).ok()?, &mut head, 0).ok()?;
    let at = head[..n].windows(4).position(|w| w == b"fLaC")?;
    let d = head.get(at + 8..at + 42)?;
    Some((u64::from(d[13] & 15) << 32) | u64::from(u32::from_be_bytes([d[14], d[15], d[16], d[17]])))
}

/// A FLAC capture read by sample index: through the frame index when the file
/// allows it, in chunks when not.
pub enum FlacSource {
    Index(FlacIndex),
    Window(FlacWindow),
}

impl FlacSource {
    pub fn open(path: &str, format: crate::reader::SampleFormat) -> Result<Self> {
        // `LD_FLAC_WINDOW=1` reads in chunks even where the index would do (to test it).
        let index = if std::env::var_os("LD_FLAC_WINDOW").is_some() { Err(anyhow::anyhow!("LD_FLAC_WINDOW is set")) } else { FlacIndex::open(path) };
        match index {
            Ok(i) => Ok(Self::Index(i)),
            Err(e) => {
                tracing::warn!("{path}: {e:#}; reading it in chunks instead (slower)");
                Ok(Self::Window(FlacWindow::open(path, format)?))
            }
        }
    }

    pub fn len(&self) -> u64 {
        match self {
            Self::Index(i) => i.len(),
            Self::Window(w) => w.len(),
        }
    }

    pub fn read(&self, start: i64, out: &mut [i32]) -> Result<()> {
        match self {
            Self::Index(i) => i.read(start, out),
            Self::Window(w) => w.read(start, out),
        }
    }
}

/// The header to decode with (`fLaC` and STREAMINFO marked last) from the bytes
/// that follow the `fLaC` marker.
fn stream_header(after_marker: &[u8]) -> Result<Vec<u8>> {
    if after_marker.len() < 38 || after_marker[0] & 0x7F != 0 {
        bail!("the FLAC stream does not start with a STREAMINFO block");
    }
    let mut header = b"fLaC".to_vec();
    header.push(after_marker[0] | 0x80);
    header.extend_from_slice(&after_marker[1..38]);
    Ok(header)
}

type Scan = (Vec<u8>, Vec<Piece>, HashMap<usize, Vec<Piece>>);

/// A bare FLAC stream at `offset`: the metadata blocks, then the frames, found
/// by their sync code and header (numbered one after the other).
fn scan_flac(path: &str, offset: u64) -> Result<Scan> {
    let file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut r = BufReader::with_capacity(8 << 20, file);
    r.seek(SeekFrom::Start(offset + 4))?;
    let mut streaminfo = Vec::new();
    let mut at = offset + 4;
    loop {
        let mut h = [0u8; 4];
        r.read_exact(&mut h)?;
        let len = (u64::from(h[1]) << 16) | (u64::from(h[2]) << 8) | u64::from(h[3]);
        if streaminfo.is_empty() {
            streaminfo.extend_from_slice(&h);
            let mut body = vec![0u8; len as usize];
            r.read_exact(&mut body)?;
            streaminfo.extend_from_slice(&body);
        } else {
            r.seek_relative(len as i64)?;
        }
        at += 4 + len;
        if h[0] & 0x80 != 0 {
            break;
        }
    }
    let header = stream_header(&streaminfo)?;
    // The frames, to the end of the file.
    let mut frames: Vec<Piece> = Vec::new();
    let mut starts: Vec<u64> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut base = at; // file offset of buf[0]
    let mut chunk = vec![0u8; 8 << 20];
    let mut done = false;
    r.seek(SeekFrom::Start(at))?;
    let mut i = 0usize; // next byte of buf to look at
    while !done {
        let n = r.read(&mut chunk)?;
        if n == 0 {
            done = true;
        } else {
            buf.extend_from_slice(&chunk[..n]);
        }
        // Up to 16 bytes short of the end, a header is cut off (unless the end of the file).
        let end = if done { buf.len() } else { buf.len().saturating_sub(16) };
        while i < end {
            match buf[i..end].iter().position(|&b| b == 0xFF) {
                None => {
                    i = end;
                }
                Some(p) => {
                    i += p;
                    let tail = &buf[i..];
                    if let Some((num, variable)) = frame_header(tail) {
                        if variable {
                            bail!("the FLAC stream has a variable block size, which --tbc read does not read by index");
                        }
                        if num == starts.len() as u64 {
                            starts.push(base + i as u64);
                        }
                    }
                    i += 1;
                }
            }
        }
        // Drop what is looked at, keeping the cut-off tail.
        let drop = i.min(buf.len());
        buf.drain(..drop);
        base += drop as u64;
        i -= drop;
    }
    let file_len = std::fs::metadata(path)?.len();
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(file_len);
        frames.push((s, (e - s) as u32));
    }
    Ok((header, frames, HashMap::new()))
}

/// FLAC in Ogg, from the first page at `offset`: the packets of the logical
/// stream are the header, the metadata blocks, then a frame each.
fn scan_ogg(file: &File, path: &str, offset: u64) -> Result<Scan> {
    // Its own handle: reading at a position moves the cursor of a shared one (Windows).
    let own = File::open(path).with_context(|| format!("opening {path}"))?;
    let file_len = own.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, own);
    r.seek(SeekFrom::Start(offset))?;
    let mut at = offset;
    let mut serial: Option<u32> = None;
    let mut packet = 0usize; // packets finished
    let mut cur: Vec<Piece> = Vec::new();
    let (mut header, mut frames, mut more) = (Vec::new(), Vec::<Piece>::new(), HashMap::new());
    loop {
        let mut h = [0u8; 27];
        match r.read_exact(&mut h) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        if &h[..4] != b"OggS" {
            break;
        }
        let ser = u32::from_le_bytes([h[14], h[15], h[16], h[17]]);
        let mut lacing = vec![0u8; usize::from(h[26])];
        r.read_exact(&mut lacing)?;
        let data: u64 = lacing.iter().map(|&v| u64::from(v)).sum();
        if at + 27 + lacing.len() as u64 + data > file_len {
            break; // a cut-off last page
        }
        let own = *serial.get_or_insert(ser) == ser;
        if own {
            let mut pos = at + 27 + lacing.len() as u64; // file offset of the page's data
            let mut run = 0u32;
            let mut run_at = pos;
            for &v in &lacing {
                run += u32::from(v);
                pos += u64::from(v);
                if v < 255 {
                    cur.push((run_at, run));
                    run = 0;
                    run_at = pos;
                    // A packet is complete.
                    let first = cur[0];
                    if packet == 0 {
                        let mut p = vec![0u8; first.1 as usize];
                        read_exact_at(file, &mut p, first.0)?;
                        if p.len() < 13 + 38 || &p[..5] != b"\x7fFLAC" || &p[9..13] != b"fLaC" {
                            bail!("this Ogg stream is not FLAC");
                        }
                        header = stream_header(&p[13..])?;
                    } else if first.1 >= 2 {
                        let mut two = [0u8; 2];
                        read_exact_at(file, &mut two, first.0)?;
                        if two[0] == 0xFF && two[1] & 0xFC == 0xF8 {
                            let k = frames.len();
                            frames.push(cur[0]);
                            if cur.len() > 1 {
                                more.insert(k, cur[1..].to_vec());
                            }
                        }
                    }
                    packet += 1;
                    cur.clear();
                }
            }
            if run > 0 {
                // The packet goes on in the next page.
                cur.push((run_at, run));
            }
        }
        r.seek_relative(data as i64)?;
        at += 27 + lacing.len() as u64 + data;
    }
    if header.is_empty() {
        bail!("no FLAC header in the Ogg stream");
    }
    Ok((header, frames, more))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mono 16-bit sine-ish signal, as FLAC by ffmpeg (bare and in Ogg).
    pub(super) fn sample(i: usize) -> i16 {
        (((i as f64 * 0.01).sin() * 12000.0) + ((i * 7919) % 97) as f64) as i16
    }

    pub(super) fn make(ext: &[&str], name: &str, n: usize) -> Option<std::path::PathBuf> {
        let dir = std::env::temp_dir().join(format!("flacindex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let raw = dir.join(format!("{name}.s16"));
        let bytes: Vec<u8> = (0..n).flat_map(|i| sample(i).to_le_bytes()).collect();
        std::fs::write(&raw, bytes).ok()?;
        let out = dir.join(name);
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "s16le", "-ar", "40000", "-ac", "1", "-i"])
            .arg(&raw)
            .args(ext)
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    fn check(path: &std::path::Path, n: usize) {
        let idx = FlacIndex::open(path.to_str().unwrap()).unwrap();
        assert_eq!(idx.len(), n as u64);
        assert_eq!(idx.rate_hz(), 40000);
        // Reads that start, end and cross frames, and lie partly outside.
        for (start, len) in [(0i64, 100usize), (4090, 20), (10_000, 9000), (n as i64 - 50, 80), (-30, 60)] {
            let mut out = vec![7i32; len];
            idx.read(start, &mut out).unwrap();
            for (k, &v) in out.iter().enumerate() {
                let p = start + k as i64;
                let want = if p < 0 || p >= n as i64 { 0 } else { i32::from(sample(p as usize)) };
                assert_eq!(v, want, "sample {p}");
            }
        }
    }

    #[test]
    fn bare_flac_is_read_by_index() {
        let Some(p) = make(&["-c:a", "flac", "-frame_size", "4096", "-f", "flac"], "bare.flac", 50_001) else {
            eprintln!("no ffmpeg: skipped");
            return;
        };
        check(&p, 50_001);
    }

    #[test]
    fn ogg_flac_is_read_by_index() {
        let Some(p) = make(&["-c:a", "flac", "-frame_size", "4096", "-f", "ogg"], "in.ogg.ldf", 50_001) else {
            eprintln!("no ffmpeg: skipped");
            return;
        };
        check(&p, 50_001);
    }
}

#[cfg(test)]
mod window_tests {
    use super::tests::{make, sample};
    use super::*;
    use crate::reader::SampleFormat;

    fn check(path: &std::path::Path, format: SampleFormat, n: usize) {
        let w = FlacWindow::open(path.to_str().unwrap(), format).unwrap();
        assert_eq!(w.len(), n as u64);
        // Forward, then back (the decoder restarts), then forward again.
        for (start, len) in [(10_000i64, 9000usize), (0, 100), (n as i64 - 50, 80), (-30, 60), (4090, 20)] {
            let mut out = vec![7i32; len];
            w.read(start, &mut out).unwrap();
            for (k, &v) in out.iter().enumerate() {
                let p = start + k as i64;
                let want = if p < 0 || p >= n as i64 { 0 } else { i32::from(sample(p as usize)) };
                assert_eq!(v, want, "sample {p}");
            }
        }
    }

    #[test]
    fn bare_flac_is_read_in_chunks() {
        let Some(p) = make(&["-c:a", "flac", "-frame_size", "4096", "-f", "flac"], "wbare.flac", 50_001) else { return };
        check(&p, SampleFormat::Flac, 50_001);
    }

    #[test]
    fn ogg_flac_is_read_in_chunks() {
        let Some(p) = make(&["-c:a", "flac", "-frame_size", "4096", "-f", "ogg"], "wogg.ldf", 50_001) else { return };
        check(&p, SampleFormat::Ldf, 50_001);
    }
}
