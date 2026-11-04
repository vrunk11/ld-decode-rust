//! Random access to the tracks of an archive, for a program that reads a
//! capture by sample index (ld-decode): the segment holding the samples asked
//! for is decoded when it is needed, with the segments of the reference tracks
//! it is predicted from, and kept in a small cache.
//!
//! Every track has a service thread that decodes what readers ask for. It
//! takes everything asked for at once (scattered reads from several threads
//! make one batch) and, when there is room in the batch, the segments a reader
//! walking the track is about to want, then decodes the batch together on the
//! GPU (or, with none, on a pool of its own). Readers wait for their segments;
//! they never decode, so the work is balanced by the queue and a segment is
//! decoded once. The services wait only for the services of the tracks they
//! are predicted from, which come before them: no cycle.
use crate::fmt::Fmt;
use crate::gpu::GpuSeg;
use crate::archive::{open, Opened, Refs, Store};
use crate::seg;
use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::Duration;

/// Segments kept (each is a segment's worth of 32-bit samples).
const CACHE_DEFAULT: usize = 2048;

/// Segments decoded ahead of the one being read (reads walk the track forward).
const PREFETCH_MAX: usize = 192;

/// Segments of a batch the GPU decodes in one go (more when readers ask for
/// more at once).
const BATCH_MAX: usize = 96;

/// Segments decoded on the GPU / on the CPU so far (see `RFX_READER_STATS`).
pub static STAT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

type Slot = Arc<OnceLock<std::result::Result<Arc<Vec<i32>>, String>>>;

#[derive(Default)]
struct Cache {
    slots: HashMap<(usize, usize), (Slot, u64)>,
    tick: u64,
}

/// What is waiting to be decoded for one track: wanted now, and read ahead.
#[derive(Default)]
struct Queue {
    urgent: Vec<(usize, Slot)>,
    ahead: Vec<(usize, Slot)>,
}

struct Svc {
    queue: Mutex<Queue>,
    cv: Condvar,
    started: AtomicBool,
}

/// An opened archive whose tracks can be read by sample index.
struct Pred {
    inner: Opened,
    cache: Mutex<Cache>,
    svc: Vec<Arc<Svc>>,
    me: Weak<Pred>,
}

impl Pred {
    /// A new reader of the predictor archive at `path`, its services started.
    fn open_pred(path: &Path) -> Result<Arc<Pred>> {
        let a = Self::open_new(path)?;
        for k in 0..a.tracks() {
            a.start(k);
        }
        Ok(a)
    }

    fn open_new(path: &Path) -> Result<Arc<Pred>> {
        let inner = open(path)?;
        let svc = (0..inner.meta.sources.len()).map(|_| Arc::new(Svc { queue: Mutex::default(), cv: Condvar::new(), started: AtomicBool::new(false) })).collect();
        Ok(Arc::new_cyclic(|me| Pred { inner, cache: Mutex::new(Cache::default()), svc, me: me.clone() }))
    }

    /// Sample rate of track `k` in Hz (0: not known).
    pub fn rate_hz(&self, k: usize) -> f64 {
        self.inner.meta.sources[k].rate_hz
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

    fn nseg(&self, k: usize) -> usize {
        (self.len(k) as usize).div_ceil(self.inner.meta.segment)
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
        let (first, last) = ((lo / s) as usize, ((hi - 1) / s) as usize);
        let qs: Vec<usize> = (first..=last).collect();
        let slots = self.ask(k, &qs, true)?;
        // A reader walking the track (the segment before is there) will want
        // the next ones.
        if first == 0 || self.done(k, first - 1) {
            let ahead: Vec<usize> = (last + 1..(last + 1 + limits().1).min(self.nseg(k))).collect();
            self.ask(k, &ahead, false)?;
        }
        for (&q, slot) in qs.iter().zip(&slots) {
            let seg = self.wait(k, q, slot)?;
            let q = q as i64;
            let (a, b) = ((q * s).max(lo), ((q + 1) * s).min(hi));
            out[(a - start) as usize..(b - start) as usize].copy_from_slice(&seg[(a - q * s) as usize..(b - q * s) as usize]);
        }
        Ok(())
    }

    fn done(&self, k: usize, q: usize) -> bool {
        self.cache.lock().map(|c| c.slots.get(&(k, q)).is_some_and(|(s, _)| s.get().is_some())).unwrap_or(true)
    }

    /// Mark segment `q` of track `k` as just used.
    fn touch(&self, k: usize, q: usize) {
        if let Ok(mut c) = self.cache.lock() {
            c.tick += 1;
            let t = c.tick;
            if let Some(e) = c.slots.get_mut(&(k, q)) {
                e.1 = t;
            }
        }
    }

    fn slot(&self, k: usize, q: usize) -> Result<Slot> {
        let mut c = self.cache.lock().map_err(|_| anyhow!("cache poisoned"))?;
        c.tick += 1;
        let t = c.tick;
        if let Some(e) = c.slots.get_mut(&(k, q)) {
            e.1 = t;
            return Ok(Arc::clone(&e.0));
        }
        let s: Slot = Arc::new(OnceLock::new());
        c.slots.insert((k, q), (Arc::clone(&s), t));
        // The least recently used decoded segment nobody holds is dropped: a
        // slot in a queue is the one its readers wait on, and a service holds
        // the references it decodes from.
        while c.slots.len() > limits().0 {
            let old = c.slots.iter().filter(|(_, (s, _))| s.get().is_some() && Arc::strong_count(s) == 1).min_by_key(|(_, (_, t))| *t).map(|(key, _)| *key);
            match old {
                Some(key) => {
                    c.slots.remove(&key);
                }
                None => break,
            }
        }
        Ok(s)
    }

    /// Put segments of track `k` in the queue of its service (wanted now, or
    /// to read ahead); returns their slots.
    fn ask(&self, k: usize, qs: &[usize], urgent: bool) -> Result<Vec<Slot>> {
        let svc = &self.svc[k];
        let mut slots = Vec::with_capacity(qs.len());
        let mut queued = false;
        {
            let mut g = svc.queue.lock().map_err(|_| anyhow!("queue poisoned"))?;
            for &q in qs {
                let slot = self.slot(k, q)?;
                if slot.get().is_none() {
                    let in_urgent = g.urgent.iter().any(|(p, _)| *p == q);
                    if urgent && !in_urgent {
                        g.ahead.retain(|(p, _)| *p != q);
                        g.urgent.push((q, Arc::clone(&slot)));
                        queued = true;
                    } else if !urgent && !in_urgent && !g.ahead.iter().any(|(p, _)| *p == q) {
                        g.ahead.push((q, Arc::clone(&slot)));
                        queued = true;
                    }
                }
                slots.push(slot);
            }
        }
        if queued {
            self.start(k);
            svc.cv.notify_all();
        }
        Ok(slots)
    }

    /// The decoded segment, once its service has it.
    fn wait(&self, k: usize, q: usize, slot: &Slot) -> Result<Arc<Vec<i32>>> {
        let svc = &self.svc[k];
        loop {
            match slot.get() {
                Some(Ok(v)) => return Ok(Arc::clone(v)),
                Some(Err(e)) => return Err(anyhow!("track {k}, segment {q}: {e}")),
                None => {}
            }
            let g = svc.queue.lock().map_err(|_| anyhow!("queue poisoned"))?;
            if slot.get().is_none() {
                let _ = svc.cv.wait_timeout(g, Duration::from_millis(200));
            }
        }
    }

    /// The segments of reference track `j` that the windows of segments `qs` of
    /// track `k` read.
    fn ref_segments(&self, k: usize, j: usize, qs: &[usize]) -> Vec<usize> {
        let meta = &self.inner.meta;
        let s = meta.segment;
        let refs = Refs { files: &[], meta, tables: &self.inner.tables, store: Some(self) };
        let mut out: Vec<usize> = Vec::new();
        for &q in qs {
            let n = s.min(meta.sources[k].words as usize - q * s);
            for (jj, lo, hi) in refs.spans(k, (q * s) as i64, n) {
                if jj == j {
                    out.extend(lo as usize / s..=(hi as usize - 1) / s);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Decode a batch of segments of track `k` (the service's work).
    fn run_batch(&self, k: usize, items: &[(usize, Slot)], gpu: &mut Option<Option<GpuSeg>>, pool: &mut Option<rayon::ThreadPool>) {
        let qs: Vec<usize> = items.iter().map(|i| i.0).collect();
        let sm = &self.inner.meta.sources[k];
        let result = (|| -> Result<Vec<Vec<i32>>> {
            // The reference tracks' segments the windows will want, asked of
            // their services in one go.
            let mut waits: Vec<(usize, usize, Slot)> = Vec::new();
            if !sm.refs.is_empty() {
                for j in sm.refs.iter().copied() {
                    let want = self.ref_segments(k, j, &qs);
                    for (&p, slot) in want.iter().zip(&self.ask(j, &want, true)?) {
                        waits.push((j, p, Arc::clone(slot)));
                    }
                    // The batch after this one, so that the references are
                    // decoded while this one is.
                    let next: Vec<usize> = (qs[qs.len() - 1] + 1..(qs[qs.len() - 1] + 1 + limits().2).min(self.nseg(k))).collect();
                    self.ask(j, &self.ref_segments(k, j, &next), false)?;
                }
                for (j, p, slot) in &waits {
                    self.wait(*j, *p, slot)?;
                }
            }
            let pool = pool.get_or_insert_with(|| {
                let n = std::thread::available_parallelism().map_or(8, |n| n.get()).max(4);
                rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("thread pool")
            });
            let g = gpu.get_or_insert_with(|| self.open_gpu(k));
            if let Some(gs) = g.as_mut() {
                if let Ok(v) = pool.install(|| self.decode_batch(k, gs, &qs)) {
                    STAT[1].fetch_add(qs.len() as u64, Ordering::Relaxed);
                    return Ok(v);
                }
            }
            STAT[0].fetch_add(qs.len() as u64, Ordering::Relaxed);
            pool.install(|| qs.par_iter().map(|&q| self.decode(k, q)).collect())
        })();
        match result {
            Ok(vs) => {
                for ((q, slot), v) in items.iter().zip(vs) {
                    let _ = slot.set(Ok(Arc::new(v)));
                    self.touch(k, *q);
                }
            }
            Err(e) => {
                let m = format!("{e:#}");
                for (_, slot) in items {
                    let _ = slot.set(Err(m.clone()));
                }
            }
        }
    }

    fn decode_batch(&self, k: usize, gs: &mut GpuSeg, qs: &[usize]) -> Result<Vec<Vec<i32>>> {
        let meta = &self.inner.meta;
        let sm = &meta.sources[k];
        let s = meta.segment;
        let pre = sm.preset();
        let use_ref = !sm.refs.is_empty();
        let l = s + 2 * pre.taps;
        let rw = pre.chans() * l;
        let lens: Vec<i32> = qs.iter().map(|&q| s.min(sm.words as usize - q * s) as i32).collect();
        let mut input = vec![0i32; qs.len() * s];
        let mut refw = if use_ref { vec![0f32; qs.len() * rw] } else { Vec::new() };
        let refs = Refs { files: &[], meta, tables: &self.inner.tables, store: Some(self) };
        let fill = |i: usize, inp: &mut [i32], win: &mut [f32]| -> Result<()> {
            let (q, n) = (qs[i], lens[i] as usize);
            let &(off, len) = self.inner.blocks[k].get(&(q as u64)).with_context(|| format!("{}: segment {q} is missing", sm.name))?;
            let mut bytes = vec![0u8; len as usize];
            crate::io::read_exact_at(&self.inner.file, &mut bytes, off)?;
            let res = seg::decode_residuals(&bytes, n).with_context(|| format!("{}: segment {q} is corrupt", sm.name))?;
            inp[..n].copy_from_slice(&res);
            if use_ref {
                refs.window_into(k, (q * s) as i64, n, l, win)?;
            }
            Ok(())
        };
        if use_ref {
            input.par_chunks_mut(s).zip(refw.par_chunks_mut(rw)).enumerate().try_for_each(|(i, (inp, win))| fill(i, inp, win))?;
        } else {
            input.par_chunks_mut(s).enumerate().try_for_each(|(i, inp)| fill(i, inp, &mut []))?;
        }
        let ys = gs.run(1, &input, &refw, &lens)?;
        Ok((0..qs.len()).map(|i| ys[i * s..i * s + lens[i] as usize].iter().map(|&v| (v + sm.mean) << sm.shift).collect()).collect())
    }

    /// One segment on the CPU.
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

/// The service of track `k`: decodes what is queued until the archive is gone.
fn serve(me: Weak<Pred>, svc: Arc<Svc>, k: usize) {
    let mut gpu: Option<Option<GpuSeg>> = me.upgrade().map(|a| a.open_gpu(k));
    let mut pool: Option<rayon::ThreadPool> = None;
    loop {
        let mut items: Vec<(usize, Slot)> = {
            let Ok(mut g) = svc.queue.lock() else { return };
            loop {
                g.urgent.retain(|(_, s)| s.get().is_none());
                g.ahead.retain(|(_, s)| s.get().is_none());
                if !g.urgent.is_empty() || !g.ahead.is_empty() {
                    break;
                }
                if me.strong_count() == 0 {
                    return;
                }
                g = match svc.cv.wait_timeout(g, Duration::from_secs(1)) {
                    Ok(r) => r.0,
                    Err(_) => return,
                };
            }
            let n = g.urgent.len().min(2 * limits().2);
            let mut items: Vec<_> = g.urgent.drain(..n).collect();
            let take = limits().2.saturating_sub(items.len()).min(g.ahead.len());
            items.extend(g.ahead.drain(..take));
            items
        };
        items.sort_by_key(|i| i.0);
        let Some(a) = me.upgrade() else { return };
        a.run_batch(k, &items, &mut gpu, &mut pool);
        drop(a);
        drop(svc.queue.lock());
        svc.cv.notify_all();
    }
}

impl Store for Pred {
    fn load(&self, j: usize, lo: u64, out: &mut [i32]) -> Result<()> {
        self.read(j, lo as i64, out)
    }
}

impl Drop for Pred {
    fn drop(&mut self) {
        if std::env::var_os("RFX_READER_STATS").is_some() {
            eprintln!("reader: {} segments decoded on the GPU, {} on the CPU", STAT[1].load(Ordering::Relaxed), STAT[0].load(Ordering::Relaxed));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tests/data/reader.rfx` was written by the `rfx` tool of RF-tools from the
    /// signal below (`rfx pack --segment 65536`): a file made by another
    /// implementation, read at positions that cross segments and the file's ends.
    #[test]
    fn reads_a_file_written_by_the_rfx_tool_at_any_position() {
        let n = 200_003usize;
        let mut x = 3u32;
        let mut s = Vec::with_capacity(n);
        for i in 0..n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (((i as f64 * 0.05).sin() * 3000.0) as i32 + ((x >> 24) as i32 - 128)) as i16;
            s.push(i32::from(v & !63));
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/reader.rfx");
        let a = Archive::open(&path).unwrap();
        assert_eq!((a.tracks(), a.len(0)), (1, n as u64));
        for (start, len) in [(0i64, 10usize), (65532, 20), (131068, 12), (30_000, 150_000), (n as i64 - 5, 20), (-7, 12)] {
            let mut got = vec![7i32; len];
            a.read(0, start, &mut got).unwrap();
            for (i, g) in got.iter().enumerate() {
                let p = start + i as i64;
                let want = if p >= 0 && (p as usize) < n { s[p as usize] } else { 0 };
                assert_eq!(*g, want, "sample {p}");
            }
        }
    }
}

impl Pred {
    /// Start the service of track `k` if it is not running: it builds the
    /// GPU kernel at once, while the reader is still setting up.
    fn start(&self, k: usize) {
        let svc = &self.svc[k];
        if !svc.started.swap(true, Ordering::AcqRel) {
            let (me, svc) = (self.me.clone(), Arc::clone(svc));
            std::thread::spawn(move || serve(me, svc, k));
        }
    }

    /// The GPU decoder of track `k` (`None`: the CPU decodes).
    fn open_gpu(&self, k: usize) -> Option<GpuSeg> {
        let sm = &self.inner.meta.sources[k];
        if std::env::var_os("RFX_READER_CPU").is_some() {
            return None;
        }
        GpuSeg::new(&sm.preset(), !sm.refs.is_empty(), self.inner.meta.segment).ok()
    }
}

/// (segments kept, read-ahead, batch): the cache is `RFX_READER_CACHE`
/// segments (a segment is 1 MB). The read-ahead of the tracks together must
/// fit in it, or what was decoded ahead is dropped before it is read.
fn limits() -> (usize, usize, usize) {
    static L: OnceLock<(usize, usize, usize)> = OnceLock::new();
    *L.get_or_init(|| {
        let cache = std::env::var("RFX_READER_CACHE").ok().and_then(|v| v.parse().ok()).unwrap_or(CACHE_DEFAULT).max(32);
        let ahead = (cache / 6).clamp(4, PREFETCH_MAX);
        (cache, ahead, (ahead / 2).clamp(2, BATCH_MAX))
    })
}

/// An opened archive whose tracks can be read by sample index: one packed
/// with the predictor (decoded by a service per track) or with `--flac`.
pub struct Archive {
    kind: Kind,
}

enum Kind {
    Pred(Arc<Pred>),
    Flac(crate::flacmode::FlacArchive),
}

impl Archive {
    /// The archive at `path`. One that is open already (a reader per track
    /// opens the same file) is shared, with its cache: a track predicted from
    /// others is decoded from their segments already there.
    pub fn open(path: &Path) -> Result<Arc<Archive>> {
        static OPEN: Mutex<Vec<(std::path::PathBuf, Weak<Archive>)>> = Mutex::new(Vec::new());
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut open_now = OPEN.lock().map_err(|_| anyhow!("archive list poisoned"))?;
        open_now.retain(|(_, w)| w.strong_count() > 0);
        if let Some(a) = open_now.iter().find(|(p, _)| *p == key).and_then(|(_, w)| w.upgrade()) {
            return Ok(a);
        }
        let kind = if crate::flacmode::is_flac_archive(path) { Kind::Flac(crate::flacmode::FlacArchive::open(path)?) } else { Kind::Pred(Pred::open_pred(path)?) };
        let a = Arc::new(Archive { kind });
        open_now.push((key, Arc::downgrade(&a)));
        Ok(a)
    }

    pub fn tracks(&self) -> usize {
        match &self.kind {
            Kind::Pred(p) => p.tracks(),
            Kind::Flac(f) => f.tracks(),
        }
    }

    /// Name of the track as it was packed.
    pub fn name(&self, k: usize) -> &str {
        match &self.kind {
            Kind::Pred(p) => p.name(k),
            Kind::Flac(f) => f.name(k),
        }
    }

    /// Samples in track `k`.
    pub fn len(&self, k: usize) -> u64 {
        match &self.kind {
            Kind::Pred(p) => p.len(k),
            Kind::Flac(f) => f.len(k),
        }
    }

    pub fn is_empty(&self, k: usize) -> bool {
        self.len(k) == 0
    }

    /// Format the track came in (its samples are on that format's scale).
    pub fn fmt(&self, k: usize) -> Fmt {
        match &self.kind {
            Kind::Pred(p) => p.fmt(k),
            Kind::Flac(f) => f.fmt(k),
        }
    }

    /// Sample rate of track `k` in Hz (0: not known).
    pub fn rate_hz(&self, k: usize) -> f64 {
        match &self.kind {
            Kind::Pred(p) => p.rate_hz(k),
            Kind::Flac(f) => f.rate_hz(k),
        }
    }

    /// Samples `[start, start + out.len())` of track `k` on the scale of the
    /// original file; zero outside the track.
    pub fn read(&self, k: usize, start: i64, out: &mut [i32]) -> Result<()> {
        match &self.kind {
            Kind::Pred(p) => p.read(k, start, out),
            Kind::Flac(f) => f.read(k, start, out),
        }
    }
}
