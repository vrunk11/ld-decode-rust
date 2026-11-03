//! Random access to the tracks of an archive, for a program that reads a
//! capture by sample index (ld-decode): the segment holding the samples asked
//! for is decoded when it is needed, with the segments of the reference tracks
//! it is predicted from, and kept in a small cache.
use crate::fmt::Fmt;
use crate::pack::{open, Opened, Refs, Store};
use crate::seg;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// Segments kept (each is a segment's worth of 32-bit samples).
const CACHE_SEGMENTS: usize = 384;

type Slot = Arc<OnceLock<std::result::Result<Arc<Vec<i32>>, String>>>;

#[derive(Default)]
struct Cache {
    slots: HashMap<(usize, usize), Slot>,
    order: VecDeque<(usize, usize)>,
}

/// An opened archive whose tracks can be read by sample index.
pub struct Archive {
    inner: Opened,
    cache: Mutex<Cache>,
    me: std::sync::Weak<Archive>,
}

/// Segments decoded ahead of the one being read (reads walk the track forward).
const PREFETCH: usize = 24;

impl Archive {
    pub fn open(path: &Path) -> Result<Arc<Archive>> {
        let inner = open(path)?;
        Ok(Arc::new_cyclic(|me| Archive { inner, cache: Mutex::new(Cache::default()), me: me.clone() }))
    }

    pub fn tracks(&self) -> usize {
        self.inner.meta.sources.len()
    }

    /// Name of the track as it was packed.
    pub fn name(&self, k: usize) -> &str {
        &self.inner.meta.sources[k].name
    }

    /// Samples in track `k`.
    pub fn len(&self, k: usize) -> u64 {
        self.inner.meta.sources[k].words
    }

    pub fn is_empty(&self, k: usize) -> bool {
        self.len(k) == 0
    }

    /// Format the track came in (its samples are on that format's scale).
    pub fn fmt(&self, k: usize) -> Fmt {
        self.inner.meta.sources[k].fmt()
    }

    /// Samples `[start, start + out.len())` of track `k` on the scale of the
    /// original file; zero outside the track.
    pub fn read(&self, k: usize, start: i64, out: &mut [i32]) -> Result<()> {
        if k >= self.tracks() {
            bail!("the archive has {} tracks: no track {k}", self.tracks());
        }
        out.fill(0);
        let words = self.len(k) as i64;
        let (lo, hi) = (start.max(0), (start + out.len() as i64).min(words));
        if hi <= lo {
            return Ok(());
        }
        let s = self.inner.meta.segment as i64;
        for q in (lo / s)..=((hi - 1) / s) {
            let seg = self.segment(k, q as usize)?;
            let (a, b) = ((q * s).max(lo), ((q + 1) * s).min(hi));
            out[(a - start) as usize..(b - start) as usize].copy_from_slice(&seg[(a - q * s) as usize..(b - q * s) as usize]);
        }
        self.prefetch(k, ((hi - 1) / s) as usize + 1);
        Ok(())
    }

    /// Start decoding the segments from `from` on, in the background, those not
    /// already wanted.
    fn prefetch(&self, k: usize, from: usize) {
        let nseg = (self.len(k) as usize).div_ceil(self.inner.meta.segment);
        for q in from..(from + PREFETCH).min(nseg) {
            let taken = self.cache.lock().map(|c| c.slots.contains_key(&(k, q))).unwrap_or(true);
            if taken {
                continue;
            }
            if let Some(me) = self.me.upgrade() {
                rayon::spawn(move || {
                    let _ = me.segment(k, q);
                });
            }
        }
    }

    /// The decoded segment `q` of track `k`.
    fn segment(&self, k: usize, q: usize) -> Result<Arc<Vec<i32>>> {
        let slot = {
            let mut c = self.cache.lock().map_err(|_| anyhow!("cache poisoned"))?;
            if let Some(s) = c.slots.get(&(k, q)) {
                Arc::clone(s)
            } else {
                let s: Slot = Arc::new(OnceLock::new());
                c.slots.insert((k, q), Arc::clone(&s));
                c.order.push_back((k, q));
                while c.order.len() > CACHE_SEGMENTS {
                    if let Some(old) = c.order.pop_front() {
                        c.slots.remove(&old);
                    }
                }
                s
            }
        };
        match slot.get_or_init(|| self.decode(k, q).map(Arc::new).map_err(|e| format!("{e:#}"))) {
            Ok(v) => Ok(Arc::clone(v)),
            Err(e) => Err(anyhow!("track {k}, segment {q}: {e}")),
        }
    }

    fn decode(&self, k: usize, q: usize) -> Result<Vec<i32>> {
        let meta = &self.inner.meta;
        let sm = &meta.sources[k];
        let s = meta.segment;
        let s0 = q * s;
        let n = s.min(sm.words as usize - s0);
        let &(off, len) = self.inner.blocks[k].get(&(q as u64)).with_context(|| format!("{}: segment {q} is missing", sm.name))?;
        let mut bytes = vec![0u8; len as usize];
        crate::io::read_exact_at(&self.inner.file, &mut bytes, off)?;
        let pre = sm.preset();
        let use_ref = !sm.refs.is_empty();
        let refs = Refs { files: &[], meta, tables: &self.inner.tables, store: Some(self) };
        let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
        let y = seg::decode(&bytes, n, &pre, use_ref, &win).with_context(|| format!("{}: segment {q} is corrupt", sm.name))?;
        Ok(y.into_iter().map(|v| (v + sm.mean) << sm.shift).collect())
    }
}

impl Store for Archive {
    fn load(&self, j: usize, lo: u64, out: &mut [i32]) -> Result<()> {
        self.read(j, lo as i64, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{pack, Device, OrderMode, PackOpts};

    #[test]
    fn reads_what_was_packed_at_any_position() {
        let dir = std::env::temp_dir().join(format!("rfx_reader_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (raw, out) = (dir.join("t.s16"), dir.join("t.rfx"));
        let n = 700_001usize;
        let mut x = 3u32;
        let mut s = Vec::with_capacity(n * 2);
        for i in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (((i as f64 * 0.05).sin() * 3000.0) as i32 + ((x >> 24) as i32 - 128)) as i16;
            s.extend_from_slice(&(v & !63).to_le_bytes());
        }
        std::fs::write(&raw, &s).unwrap();
        let o = PackOpts { order: OrderMode::Given, device: Device::Cpu, level: 3, segment: 262144, verify: true, pass0_dir: None, ld_decode: "ld-decode".into(), rate_mhz: 75.0, format: "auto".into() };
        pack(&[raw.clone()], &out, &o).unwrap();
        let a = Archive::open(&out).unwrap();
        assert_eq!((a.tracks(), a.len(0)), (1, n as u64));
        for (start, len) in [(0i64, 10usize), (262140, 20), (100_000, 500_000), (n as i64 - 5, 20), (-7, 12)] {
            let mut got = vec![7i32; len];
            a.read(0, start, &mut got).unwrap();
            for (i, g) in got.iter().enumerate() {
                let p = start + i as i64;
                let want = if p >= 0 && (p as usize) < n { i32::from(i16::from_le_bytes([s[2 * p as usize], s[2 * p as usize + 1]])) } else { 0 };
                assert_eq!(*g, want, "sample {p}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
