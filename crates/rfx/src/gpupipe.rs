//! The GPU path of `pack` and `unpack` as a pipeline of three stages that run
//! at once on successive chunks of segments, so the device is not left idle
//! while the CPU reads, windows and entropy-codes:
//!
//! - packing: read samples and reference windows (CPU, parallel) -> predict
//!   (GPU) -> residuals, range coding, self-check and writing (CPU);
//! - unpacking: read blocks, parse residuals and build windows (CPU) ->
//!   reconstruct (GPU) -> write the words (CPU).
use crate::ebml;
use crate::gpu::GpuSeg;
use crate::io::{read_exact_at, read_words, write_all_at};
use crate::pack::{Refs, SrcMeta};
use crate::pred::Preset;
use crate::seg;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::mpsc::sync_channel;

/// What the first stage hands on: `segs.len()` segments of `s` values
/// (padded), their reference windows and real lengths.
struct Chunk {
    segs: Vec<usize>,
    lens: Vec<i32>,
    input: Vec<i32>,
    refw: Vec<f32>,
}

fn lens_of(segs: &[usize], s: usize, words: u64) -> Vec<i32> {
    segs.iter().map(|&q| s.min(words as usize - q * s) as i32).collect()
}

/// Run `fill(i, segment_input, segment_window)` over the chunk in parallel.
fn build_chunk(
    segs: Vec<usize>,
    s: usize,
    rw: usize,
    use_ref: bool,
    words: u64,
    fill: impl Fn(usize, usize, &mut [i32], &mut [f32]) -> Result<()> + Sync,
) -> Result<Chunk> {
    let lens = lens_of(&segs, s, words);
    let mut input = vec![0i32; segs.len() * s];
    let mut refw = if use_ref { vec![0f32; segs.len() * rw] } else { Vec::new() };
    if use_ref {
        input
            .par_chunks_mut(s)
            .zip(refw.par_chunks_mut(rw))
            .enumerate()
            .try_for_each(|(i, (inp, win))| fill(i, segs[i], inp, win))?;
    } else {
        input.par_chunks_mut(s).enumerate().try_for_each(|(i, inp)| fill(i, segs[i], inp, &mut []))?;
    }
    Ok(Chunk { segs, lens, input, refw })
}

fn chunks_of(nseg: usize, chunk: usize) -> Vec<Vec<usize>> {
    (0..nseg).step_by(chunk).map(|f| (f..(f + chunk).min(nseg)).collect()).collect()
}

/// Pack source `k`; returns the bytes written.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_source(
    k: usize,
    sm: &SrcMeta,
    pre: &Preset,
    s: usize,
    chunk: usize,
    verify: bool,
    files: &[File],
    refs: &Refs,
    w: &mut impl Write,
) -> Result<u64> {
    let use_ref = !sm.refs.is_empty();
    let nseg = (sm.words as usize).div_ceil(s);
    let mut predictor = GpuSeg::new(pre, use_ref, s)?;
    let mut verifier = if verify { Some(GpuSeg::new(pre, use_ref, s)?) } else { None };
    let rw = predictor.window_len();
    let (tx1, rx1) = sync_channel::<Result<Chunk>>(1);
    let (tx2, rx2) = sync_channel::<Result<(Chunk, Vec<i32>)>>(1);
    let mut packed = 0u64;
    std::thread::scope(|sc| -> Result<()> {
        sc.spawn(move || {
            for segs in chunks_of(nseg, chunk) {
                let c = build_chunk(segs, s, rw, use_ref, sm.words, |_, q, inp, win| {
                    let s0 = q * s;
                    let n = s.min(sm.words as usize - s0);
                    let y = read_words(&files[k], sm.words, s0 as i64, n, sm.shift)?;
                    for (o, v) in inp.iter_mut().zip(y) {
                        *o = v - sm.mean;
                    }
                    if use_ref {
                        let wv = refs.window(k, s0 as i64, n)?;
                        win[..wv.len()].copy_from_slice(&wv);
                    }
                    Ok(())
                });
                let stop = c.is_err();
                if tx1.send(c).is_err() || stop {
                    return;
                }
            }
        });
        sc.spawn(move || {
            for c in rx1 {
                let r = c.and_then(|c| {
                    let preds = predictor.run(0, &c.input, &c.refw, &c.lens)?;
                    Ok((c, preds))
                });
                let stop = r.is_err();
                if tx2.send(r).is_err() || stop {
                    return;
                }
            }
        });
        for r in rx2 {
            let (c, preds) = r?;
            let coded: Vec<(Vec<u8>, Vec<i32>)> = (0..c.segs.len())
                .into_par_iter()
                .map(|i| {
                    let n = c.lens[i] as usize;
                    let y = &c.input[i * s..i * s + n];
                    let res: Vec<i32> = y.iter().zip(&preds[i * s..]).map(|(a, b)| a - b).collect();
                    (seg::encode_residuals(&res), res)
                })
                .collect();
            if let Some(v) = verifier.as_mut() {
                let mut flat = vec![0i32; c.segs.len() * s];
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
                let bad = (0..c.segs.len()).into_par_iter().find_any(|&i| {
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
    })?;
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
    chunk: usize,
    blocks: &HashMap<u64, (u64, u64)>,
    archive: &File,
    refs: &Refs,
    out: &File,
) -> Result<()> {
    let use_ref = !sm.refs.is_empty();
    let nseg = (sm.words as usize).div_ceil(s);
    let mut gpu = GpuSeg::new(pre, use_ref, s)?;
    let rw = gpu.window_len();
    let (tx1, rx1) = sync_channel::<Result<Chunk>>(1);
    let (tx2, rx2) = sync_channel::<Result<(Chunk, Vec<i32>)>>(1);
    std::thread::scope(|sc| -> Result<()> {
        sc.spawn(move || {
            for segs in chunks_of(nseg, chunk) {
                let c = build_chunk(segs, s, rw, use_ref, sm.words, |_, q, inp, win| {
                    let s0 = q * s;
                    let n = s.min(sm.words as usize - s0);
                    let &(off, len) = blocks.get(&(q as u64)).with_context(|| format!("{}: segment {q} is missing", sm.name))?;
                    let mut bytes = vec![0u8; len as usize];
                    read_exact_at(archive, &mut bytes, off)?;
                    let res = seg::decode_residuals(&bytes, n).with_context(|| format!("{}: segment {q} is corrupt", sm.name))?;
                    inp[..n].copy_from_slice(&res);
                    if use_ref {
                        let wv = refs.window(k, s0 as i64, n)?;
                        win[..wv.len()].copy_from_slice(&wv);
                    }
                    Ok(())
                });
                let stop = c.is_err();
                if tx1.send(c).is_err() || stop {
                    return;
                }
            }
        });
        sc.spawn(move || {
            for c in rx1 {
                let r = c.and_then(|c| {
                    let ys = gpu.run(1, &c.input, &c.refw, &c.lens)?;
                    Ok((c, ys))
                });
                let stop = r.is_err();
                if tx2.send(r).is_err() || stop {
                    return;
                }
            }
        });
        for r in rx2 {
            let (c, ys) = r?;
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
    })
}
