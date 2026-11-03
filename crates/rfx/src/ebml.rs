//! The little of Matroska the archive uses, written and read by hand: an EBML
//! header (DocType `rfx`), one Segment of unknown size holding Info, Tracks
//! and Clusters, each Cluster holding one SimpleBlock. The codec of every
//! track is a private `D_RFX/...` id, so a general Matroska tool lists the
//! tracks and their blocks but cannot play anything.
use crate::io::read_exact_at;
use anyhow::{bail, Result};
use std::fs::File;

pub const ID_EBML: u32 = 0x1A45_DFA3;
pub const ID_SEGMENT: u32 = 0x1853_8067;
pub const ID_INFO: u32 = 0x1549_A966;
pub const ID_TRACKS: u32 = 0x1654_AE6B;
pub const ID_TRACK_ENTRY: u32 = 0xAE;
pub const ID_TRACK_NUMBER: u32 = 0xD7;
pub const ID_TRACK_UID: u32 = 0x73C5;
pub const ID_TRACK_TYPE: u32 = 0x83;
pub const ID_CODEC_ID: u32 = 0x86;
pub const ID_NAME: u32 = 0x536E;
pub const ID_CLUSTER: u32 = 0x1F43_B675;
pub const ID_TIMESTAMP: u32 = 0xE7;
pub const ID_SIMPLE_BLOCK: u32 = 0xA3;
pub const ID_TIMESTAMP_SCALE: u32 = 0x2A_D7B1;
pub const ID_MUXING_APP: u32 = 0x4D80;
pub const ID_WRITING_APP: u32 = 0x5741;

fn id_bytes(id: u32) -> Vec<u8> {
    let b = id.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count();
    b[skip..].to_vec()
}

/// EBML variable-length size, the shortest form.
pub fn vint(v: u64) -> Vec<u8> {
    let mut n = 1;
    while n < 8 && v >= (1u64 << (7 * n)) - 1 {
        n += 1;
    }
    let mut out = vec![0u8; n];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (v >> (8 * (n - 1 - i))) as u8;
    }
    out[0] |= 1 << (8 - n);
    out
}

pub fn elem(id: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend(vint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

pub fn uint(id: u32, v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count().min(7);
    elem(id, &b[skip..])
}

pub fn string(id: u32, s: &str) -> Vec<u8> {
    elem(id, s.as_bytes())
}

pub fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

/// EBML header and the start of the Segment (unknown size).
pub fn file_head() -> Vec<u8> {
    let header = concat(&[
        uint(0x4286, 1),
        uint(0x42F7, 1),
        uint(0x42F2, 4),
        uint(0x42F3, 8),
        string(0x4282, "rfx"),
        uint(0x4287, 1),
        uint(0x4285, 1),
    ]);
    let mut out = elem(ID_EBML, &header);
    out.extend(id_bytes(ID_SEGMENT));
    out.extend([0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    out
}

pub fn info() -> Vec<u8> {
    elem(
        ID_INFO,
        &concat(&[uint(ID_TIMESTAMP_SCALE, 1_000_000), string(ID_MUXING_APP, "rfx"), string(ID_WRITING_APP, "rfx")]),
    )
}

/// Track types: 0x21 is "metadata".
pub fn track_entry(number: u64, codec: &str, name: &str) -> Vec<u8> {
    elem(
        ID_TRACK_ENTRY,
        &concat(&[uint(ID_TRACK_NUMBER, number), uint(ID_TRACK_UID, number), uint(ID_TRACK_TYPE, 0x21), string(ID_CODEC_ID, codec), string(ID_NAME, name)]),
    )
}

pub fn tracks(entries: &[Vec<u8>]) -> Vec<u8> {
    elem(ID_TRACKS, &concat(entries))
}

/// A Cluster holding one SimpleBlock of `track` (timestamp = `ts`).
pub fn cluster(ts: u64, track: u64, payload: &[u8]) -> Vec<u8> {
    let mut block = vint(track);
    block.extend([0, 0, 0x80]);
    block.extend_from_slice(payload);
    elem(ID_CLUSTER, &concat(&[uint(ID_TIMESTAMP, ts), elem(ID_SIMPLE_BLOCK, &block)]))
}

// ---- reading ----

pub struct BlockRef {
    pub track: u64,
    pub ts: u64,
    pub off: u64,
    pub len: u64,
}

#[derive(Default)]
pub struct Index {
    /// Codec id of each track number.
    pub codecs: Vec<(u64, String)>,
    pub blocks: Vec<BlockRef>,
    /// CodecPrivate of each track number (when it has one).
    pub private: Vec<(u64, Vec<u8>)>,
    /// Attached files: name, offset and length of the data.
    pub attachments: Vec<(String, u64, u64)>,
}

struct Header {
    id: u32,
    size: Option<u64>,
    len: u64,
}

fn read_header(f: &File, pos: u64, file_len: u64) -> Result<Header> {
    let mut buf = [0u8; 12];
    let n = (file_len - pos).min(12) as usize;
    if n < 2 {
        bail!("truncated element at byte {pos}");
    }
    read_exact_at(f, &mut buf[..n], pos)?;
    let first = buf[0];
    if first == 0 {
        bail!("bad element id at byte {pos}");
    }
    let idl = first.leading_zeros() as usize + 1;
    if idl > 4 || idl >= n {
        bail!("bad element id at byte {pos}");
    }
    let id = buf[..idl].iter().fold(0u32, |a, &b| (a << 8) | u32::from(b));
    let s0 = buf[idl];
    if s0 == 0 {
        bail!("bad element size at byte {pos}");
    }
    let sl = s0.leading_zeros() as usize + 1;
    if idl + sl > n {
        bail!("truncated element at byte {pos}");
    }
    let mut v = u64::from(s0) & ((1u64 << (8 - sl)) - 1);
    let mut all_ones = v == (1u64 << (8 - sl)) - 1;
    for &b in &buf[idl + 1..idl + sl] {
        v = (v << 8) | u64::from(b);
        all_ones &= b == 0xFF;
    }
    Ok(Header { id, size: if all_ones { None } else { Some(v) }, len: (idl + sl) as u64 })
}

fn parse_uint(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, &x| (a << 8) | u64::from(x))
}

fn read_payload(f: &File, off: u64, len: u64) -> Result<Vec<u8>> {
    let mut v = vec![0u8; len as usize];
    read_exact_at(f, &mut v, off)?;
    Ok(v)
}

/// Children of an in-memory master element: (id, payload range).
fn children(buf: &[u8]) -> Result<Vec<(u32, std::ops::Range<usize>)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let first = buf[i];
        let idl = first.leading_zeros() as usize + 1;
        if first == 0 || idl > 4 || i + idl >= buf.len() {
            bail!("bad element in a master");
        }
        let id = buf[i..i + idl].iter().fold(0u32, |a, &b| (a << 8) | u32::from(b));
        let s0 = buf[i + idl];
        let sl = s0.leading_zeros() as usize + 1;
        if s0 == 0 || i + idl + sl > buf.len() {
            bail!("bad element size in a master");
        }
        let mut v = u64::from(s0) & ((1u64 << (8 - sl)) - 1);
        for &b in &buf[i + idl + 1..i + idl + sl] {
            v = (v << 8) | u64::from(b);
        }
        let start = i + idl + sl;
        let end = start + v as usize;
        if end > buf.len() {
            bail!("element runs past its master");
        }
        out.push((id, start..end));
        i = end;
    }
    Ok(out)
}

/// Walk the file once, reading element headers only (no payload but Tracks
/// and the first bytes of each block).
pub fn scan(f: &File) -> Result<Index> {
    scan_with(f, Some(b"rfx"))
}

/// [`scan`] for a Matroska file of any document type (or just `doctype`), as
/// a tool like ffmpeg writes it: SimpleBlocks or BlockGroups in the Clusters,
/// CodecPrivate in the track entries, Attachments.
pub fn scan_with(f: &File, doctype: Option<&[u8]>) -> Result<Index> {
    let file_len = f.metadata()?.len();
    let mut idx = Index::default();
    let h = read_header(f, 0, file_len)?;
    if h.id != ID_EBML {
        bail!("not an rfx file (no EBML header)");
    }
    let head = read_payload(f, h.len, h.size.unwrap_or(0))?;
    let doc = children(&head)?.into_iter().find(|(id, _)| *id == 0x4282).map(|(_, r)| head[r].to_vec());
    if doctype.is_some() && doc.as_deref() != doctype {
        bail!("not an rfx file (DocType is not rfx)");
    }
    let mut pos = h.len + h.size.unwrap_or(0);
    let seg = read_header(f, pos, file_len)?;
    if seg.id != ID_SEGMENT {
        bail!("no Segment after the EBML header");
    }
    pos += seg.len;
    let seg_end = seg.size.map_or(file_len, |s| (pos + s).min(file_len));
    while pos < seg_end {
        let h = read_header(f, pos, file_len)?;
        let size = match h.size {
            Some(s) => s,
            None => bail!("unknown-size element inside the Segment at byte {pos}"),
        };
        let body = pos + h.len;
        if body + size > file_len {
            bail!("the file is truncated (element at byte {pos} runs past the end)");
        }
        match h.id {
            ID_TRACKS => {
                let buf = read_payload(f, body, size)?;
                for (id, r) in children(&buf)? {
                    if id != ID_TRACK_ENTRY {
                        continue;
                    }
                    let (mut number, mut codec, mut private) = (0, String::new(), Vec::new());
                    for (cid, cr) in children(&buf[r.clone()])? {
                        let data = &buf[r.start + cr.start..r.start + cr.end];
                        match cid {
                            ID_TRACK_NUMBER => number = parse_uint(data),
                            ID_CODEC_ID => codec = String::from_utf8_lossy(data).into_owned(),
                            0x63A2 => private = data.to_vec(),
                            _ => {}
                        }
                    }
                    idx.codecs.push((number, codec));
                    if !private.is_empty() {
                        idx.private.push((number, private));
                    }
                }
            }
            ID_CLUSTER => {
                let (mut ts, mut p) = (0u64, body);
                while p < body + size {
                    let c = read_header(f, p, file_len)?;
                    let csize = c.size.unwrap_or(0);
                    match c.id {
                        ID_TIMESTAMP => ts = parse_uint(&read_payload(f, p + c.len, csize)?),
ID_SIMPLE_BLOCK => idx.blocks.push(block_ref(f, p + c.len, csize, ts)?),                        // A BlockGroup: its Block (a tool like ffmpeg writes a group for                        // the last packet of a track, or one with a duration).                        0xA0 => {                            let mut q = p + c.len;                            while q < p + c.len + csize {                                let g = read_header(f, q, file_len)?;                                let gs = g.size.unwrap_or(0);                                if g.id == 0xA1 {                                    idx.blocks.push(block_ref(f, q + g.len, gs, ts)?);                                }                                q += g.len + gs;                            }                        }
                        _ => {}
                    }
                    p += c.len + csize;
                }
            }
            // Attachments: AttachedFile (0x61A7) holding FileName (0x466E) and FileData (0x465C).
            0x1941_A469 => {
                if size > 1 << 28 {
                    bail!("the attachments are too large");
                }
                let buf = read_payload(f, body, size)?;
                for (id, r) in children(&buf)? {
                    if id != 0x61A7 {
                        continue;
                    }
                    let (mut name, mut data) = (String::new(), (0u64, 0u64));
                    for (cid, cr) in children(&buf[r.clone()])? {
                        let (a, b) = (r.start + cr.start, r.start + cr.end);
                        match cid {
                            0x466E => name = String::from_utf8_lossy(&buf[a..b]).into_owned(),
                            0x465C => data = (body + a as u64, (b - a) as u64),
                            _ => {}
                        }
                    }
                    idx.attachments.push((name, data.0, data.1));
                }
            }
            _ => {}
        }
        pos = body + size;
    }
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn vint_forms() {
        assert_eq!(vint(0), [0x80]);
        assert_eq!(vint(126), [0xFE]);
        assert_eq!(vint(127), [0x40, 0x7F]);
        assert_eq!(vint(16382), [0x7F, 0xFE]);
        assert_eq!(vint(16383), [0x20, 0x3F, 0xFF]);
    }

    #[test]
    fn write_and_scan() {
        let path = std::env::temp_dir().join("rfx_ebml_test.rfx");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&file_head()).unwrap();
        f.write_all(&info()).unwrap();
        f.write_all(&tracks(&[track_entry(1, "D_RFX/PRED", "a"), track_entry(2, "D_RFX/META", "m")])).unwrap();
        f.write_all(&cluster(0, 2, b"hello")).unwrap();
        f.write_all(&cluster(7, 1, &[1u8; 300])).unwrap();
        drop(f);
        let f = File::open(&path).unwrap();
        let idx = scan(&f).unwrap();
        assert_eq!(idx.codecs, vec![(1, "D_RFX/PRED".to_string()), (2, "D_RFX/META".to_string())]);
        assert_eq!(idx.blocks.len(), 2);
        assert_eq!((idx.blocks[0].track, idx.blocks[0].len), (2, 5));
        assert_eq!((idx.blocks[1].track, idx.blocks[1].ts, idx.blocks[1].len), (1, 7, 300));
        let mut b = [0u8; 5];
        read_exact_at(&f, &mut b, idx.blocks[0].off).unwrap();
        assert_eq!(&b, b"hello");
    }
}

/// The block whose element data (track number, timestamp offset, flags, then
/// the payload) is `len` bytes at `at`.
fn block_ref(f: &File, at: u64, len: u64, ts: u64) -> Result<BlockRef> {
    let mut head = [0u8; 12];
    let n = len.min(12) as usize;
    read_exact_at(f, &mut head[..n], at)?;
    let tl = head[0].leading_zeros() as usize + 1;
    let mut track = u64::from(head[0]) & ((1u64 << (8 - tl)) - 1);
    for &b in &head[1..tl] {
        track = (track << 8) | u64::from(b);
    }
    let skip = (tl + 3) as u64;
    Ok(BlockRef { track, ts, off: at + skip, len: len - skip })
}
