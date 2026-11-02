//! The GPU path of `pack` and `unpack` as a pipeline whose stages run at once
//! on successive chunks of segments, so neither the device nor the CPU waits
//! on the other:
//!
//! - packing: read samples and reference windows (CPU, parallel) -> upload ->
//!   predict -> download -> residuals, range coding, self-check and writing
//!   (CPU);
//! - unpacking: read blocks, parse residuals and build windows (CPU) ->
//!   upload -> reconstruct -> download -> write the words (CPU).
//!
//! Upload, kernel and download each have a command queue and a thread, so a
//! chunk can be going up while another runs and a third comes back; the
//! device buffers of a chunk travel with it and return to a small pool.
use crate::ebml;
use crate::gpu::{self, Dev, GpuSeg};
use crate::io::{read_exact_at, write_all_at};
use crate::pack::{Refs, SrcMeta};
use crate::pred::Preset;
use crate::seg;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

/// Segments per chunk.
const CHUNK: usize = 384;
/// Sets of device buffers: one uploading, one running, one coming back.
const DEV_SETS: usize = 3;

/// Busy time of the CPU stages, in microseconds (printed with RFX_TIMING).
static BUSY: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Adds the time it lives to a stage's busy total.
struct Busy(usize, Instant);

impl Busy {
    fn new(stage: usize) -> Self {
        Self(stage, Instant::now())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        BUSY[self.0].fetch_add(self.1.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

fn report(what: &str, wall: Instant) {
    if std::env::var_os("RFX_TIMING").is_some() {
        let b: Vec<f64> = BUSY.iter().map(|x| x.swap(0, Ordering::Relaxed) as f64 / 1e6).collect();
        let g = GpuSeg::times();
        GpuSeg::reset_times();
        eprintln!(
            "  {what}: wall {:.2} s, read+window {:.2} s, upload {:.2}, kernel {:.2}, download {:.2}, code+write {:.2} s",
            wall.elapsed().as_secs_f64(),
            b[0],
            g[0],
            g[1],
            g[2],
            b[1]
        );
    }
}

/// The host side of a chunk: its segments, real lengths, padded samples (or
/// residuals) and reference windows.
struct Chunk {
    segs: Vec<usize>,
    lens: Vec<i32>,
    input: Vec<i32>,
    refw: Vec<f32>,
}

/// Run `fill(q, segment_input, segment_window)` over the segments in parallel.
fn build_chunk(
    segs: Vec<usize>,
    s: usize,
    rw: usize,
    use_ref: bool,
    words: u64,
    fill: impl Fn(usize, &mut [i32], &mut [f32]) -> Result<()> + Sync,
) -> Result<Chunk> {
    let lens: Vec<i32> = segs.iter().map(|&q| s.min(words as usize - q * s) as i32).collect();
    let mut input = vec![0i32; segs.len() * s];
    let mut refw = if use_ref { vec![0f32; segs.len() * rw] } else { Vec::new() };
    if use_ref {
        input.par_chunks_mut(s).zip(refw.par_chunks_mut(rw)).enumerate().try_for_each(|(i, (inp, win))| fill(segs[i], inp, win))?;
    } else {
        input.par_chunks_mut(s).enumerate().try_for_each(|(i, inp)| fill(segs[i], inp, &mut []))?;
    }
    Ok(Chunk { segs, lens, input, refw })
}

fn chunks_of(nseg: usize) -> Vec<Vec<usize>> {
    (0..nseg).step_by(CHUNK).map(|f| (f..(f + CHUNK).min(nseg)).collect()).collect()
}

/// Channels and threads of the three device stages; `mode` is the kernel's.
/// Returns the receiver of the downloaded chunks.
#[allow(clippy::too_many_arguments)]
fn device_stages<'s, 'e>(
    sc: &'s std::thread::Scope<'s, 'e>,
    mode: i32,
    pre: &Preset,
    use_ref: bool,
    s: usize,
    first: std::sync::mpsc::Receiver<Result<Chunk>>,
) -> Result<std::sync::mpsc::Receiver<Result<(Chunk, Vec<i32>)>>> {
    let (up, mut run, down) = gpu::open(pre, use_ref, s)?;
    let (free_tx, free_rx) = sync_channel::<Dev>(DEV_SETS);
    for _ in 0..DEV_SETS {
        free_tx.send(up.dev()?).ok();
    }
    let (tx2, rx2) = sync_channel::<Result<(Chunk, Dev)>>(1);
    let (tx3, rx3) = sync_channel::<Result<(Chunk, Dev)>>(1);
    let (tx4, rx4) = sync_channel::<Result<(Chunk, Vec<i32>)>>(1);
    sc.spawn(move || {
        for c in first {
            let r = c.and_then(|c| {
                let mut dev = free_rx.recv().context("device buffers are gone")?;
                up.upload(&mut dev, &c.input, &c.refw, &c.lens)?;
                Ok((c, dev))
            });
            let stop = r.is_err();
            if tx2.send(r).is_err() || stop {
                return;
            }
        }
    });
    sc.spawn(move || {
        for r in rx2 {
            let r = r.and_then(|(c, dev)| {
                run.run(&dev, mode)?;
                Ok((c, dev))
            });
            let stop = r.is_err();
            if tx3.send(r).is_err() || stop {
                return;
            }
        }
    });
    sc.spawn(move || {
        for r in rx3 {
            let r = r.and_then(|(c, dev)| {
                let out = down.download(&dev)?;
                free_tx.send(dev).ok();
                Ok((c, out))
            });
            let stop = r.is_err();
            if tx4.send(r).is_err() || stop {
                return;
            }
        }
    });
    Ok(rx4)
}

/// Pack source `k`; returns the bytes written.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_source(
    k: usize,
    sm: &SrcMeta,
    pre: &Preset,
    s: usize,
    verify: bool,
    files: &[File],
    refs: &Refs,
    w: &mut impl Write,
) -> Result<u64> {
    let use_ref = !sm.refs.is_empty();
    let nseg = (sm.words as usize).div_ceil(s);
    let mut verifier = if verify { Some(GpuSeg::new(pre, use_ref, s)?) } else { None };
    let rw = s + 2 * pre.taps;
    let (tx1, rx1) = sync_channel::<Result<Chunk>>(1);
    let mut packed = 0u64;
    let wall = Instant::now();
    let res = std::thread::scope(|sc| -> Result<()> {
        let rx = device_stages(sc, 0, pre, use_ref, s, rx1)?;
        sc.spawn(move || {
            for segs in chunks_of(nseg) {
                let g1 = Busy::new(0);
                let c = build_chunk(segs, s, rw, use_ref, sm.words, |q, inp, win| {
                    let s0 = q * s;
                    let n = s.min(sm.words as usize - s0);
crate::io::read_words_into(&files[k], sm.words, s0 as i64, &mut inp[..n], sm.shift, sm.mean)?;
                    if use_ref {
refs.window_into(k, s0 as i64, n, win)?;
                    }
                    Ok(())
                });
                drop(g1);
                let stop = c.is_err();
                if tx1.send(c).is_err() || stop {
                    return;
                }
            }
        });
        for r in rx {
            let (c, preds) = r?;
            let _g = Busy::new(1);
            let b = c.segs.len();
            let coded: Vec<(Vec<u8>, Vec<i32>)> = (0..b)
                .into_par_iter()
                .map(|i| {
                    let n = c.lens[i] as usize;
                    let y = &c.input[i * s..i * s + n];
                    let res: Vec<i32> = y.iter().zip(&preds[i * s..]).map(|(a, b)| a - b).collect();
                    (seg::encode_residuals(&res), res)
                })
                .collect();
            if let Some(v) = verifier.as_mut() {
                let mut flat = vec![0i32; b * s];
                flat.par_chunks_mut(s).zip(coded.par_iter()).enumerate().try_for_each(|(i, (dst, (bytes, res)))| {
                    match seg::decode_residuals(bytes, c.lens[i] as usize) {
                        Some(d) if d == *res => {
                            dst[..d.len()].copy_from_slice(&d);
                            Ok(())
                        }
                        _ => bail!("self-check failed: segment {} of {} does not decode to its residuals", c.segs[i], sm.name),
                    }
                })?;
                let back = v.run(1, &flat, &c.refw, &c.lens)?;
                let bad = (0..b).into_par_iter().find_any(|&i| {
                    let n = c.lens[i] as usize;
                    back[i * s..i * s + n] != c.input[i * s..i * s + n]
                });
                if let Some(i) = bad {
                    bail!("self-check failed: segment {} of {} does not decode to itself on the GPU", c.segs[i], sm.name);
                }
            }
            for (q, (bytes, _)) in c.segs.iter().zip(&coded) {
                packed += bytes.len() as u64;
                w.write_all(&ebml::cluster(*q as u64, k as u64 + 1, bytes))?;
            }
        }
        Ok(())
    });
    report("pack", wall);
    res?;
    Ok(packed)
}

/// Unpack source `k` into `out` (word `q * s` of segment `q`); the reference
/// sources are read from `refs` (already written).
#[allow(clippy::too_many_arguments)]
pub(crate) fn unpack_source(
    k: usize,
    sm: &SrcMeta,
    pre: &Preset,
    s: usize,
    blocks: &HashMap<u64, (u64, u64)>,
    archive: &File,
    refs: &Refs,
    out: &File,
) -> Result<()> {
    let use_ref = !sm.refs.is_empty();
    let nseg = (sm.words as usize).div_ceil(s);
    let rw = s + 2 * pre.taps;
    let (tx1, rx1) = sync_channel::<Result<Chunk>>(1);
    let wall = Instant::now();
    let res = std::thread::scope(|sc| -> Result<()> {
        let rx = device_stages(sc, 1, pre, use_ref, s, rx1)?;
        sc.spawn(move || {
            for segs in chunks_of(nseg) {
                let g1 = Busy::new(0);
                let c = build_chunk(segs, s, rw, use_ref, sm.words, |q, inp, win| {
                    let s0 = q * s;
                    let n = s.min(sm.words as usize - s0);
                    let &(off, len) = blocks.get(&(q as u64)).with_context(|| format!("{}: segment {q} is missing", sm.name))?;
                    let mut bytes = vec![0u8; len as usize];
                    read_exact_at(archive, &mut bytes, off)?;
                    let res = seg::decode_residuals(&bytes, n).with_context(|| format!("{}: segment {q} is corrupt", sm.name))?;
                    inp[..n].copy_from_slice(&res);
                    if use_ref {
refs.window_into(k, s0 as i64, n, win)?;
                    }
                    Ok(())
                });
                drop(g1);
                let stop = c.is_err();
                if tx1.send(c).is_err() || stop {
                    return;
                }
            }
        });
        for r in rx {
            let (c, ys) = r?;
            let _g = Busy::new(1);
            c.segs.par_iter().enumerate().try_for_each(|(i, &q)| -> Result<()> {
                let n = c.lens[i] as usize;
                let mut raw = Vec::with_capacity(n * 2);
                for &v in &ys[i * s..i * s + n] {
                    raw.extend_from_slice(&(((v + sm.mean) << sm.shift) as i16).to_le_bytes());
                }
                write_all_at(out, &raw, (q * s) as u64 * 2)?;
                Ok(())
            })?;
        }
        Ok(())
    });
    report("unpack", wall);
    res
}
