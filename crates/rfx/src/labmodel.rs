//! Development: residual model experiments. The ideal code length of the same
//! residuals under several context sets, mixed in the logistic domain. The
//! numbers are what a model could reach; nothing here is part of the format.
use crate::io::read_words;
use crate::pack::{prepare, PackOpts, Refs};
use crate::seg;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

fn stretch(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn squash(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A table of adaptive bit probabilities: the first updates of a cell go
/// faster, down to a floor rate of `1 / limit`.
struct Tab {
    p: Vec<f32>,
    n: Vec<u16>,
    slots: usize,
    limit: f32,
}

impl Tab {
    fn new(ctx: usize, slots: usize, limit: f32) -> Self {
        Self { p: vec![0.5; ctx * slots], n: vec![0; ctx * slots], slots, limit }
    }
    fn get(&self, c: usize, s: usize) -> f64 {
        f64::from(self.p[c * self.slots + s]).clamp(1.0 / 4096.0, 1.0 - 1.0 / 4096.0)
    }
    fn upd(&mut self, c: usize, s: usize, bit: u32) {
        let i = c * self.slots + s;
        let r = (1.0 / (f32::from(self.n[i]) + 1.5)).max(1.0 / self.limit);
        self.p[i] += (bit as f32 - self.p[i]) * r;
        self.n[i] = self.n[i].saturating_add(1);
    }
}

fn log_bin(v: u32, max: usize) -> usize {
    let bl = 32 - v.leading_zeros();
    let second = if bl >= 2 { (v >> (bl - 2)) & 1 } else { 0 };
    ((2 * bl + second) as usize).min(max - 1)
}

const SLOTS: usize = 33 + 33 + 66;

fn zigzag(e: i32) -> u32 {
    ((e << 1) ^ (e >> 31)) as u32
}

/// Ideal bits of coding `res`; `ctx_of(i, past, out)` pushes one context per
/// input table.
fn model_bits(res: &[i32], sizes: &[usize], limit: f32, lr: f64, mut ctx_of: impl FnMut(usize, &[i32], &mut Vec<usize>)) -> f64 {
    let ni = sizes.len();
    let mut tabs: Vec<Tab> = sizes.iter().map(|&s| Tab::new(s, SLOTS, limit)).collect();
    let init = if ni > 1 { 1.0 / ni as f64 + 0.15 } else { 1.0 };
    let mut wts = vec![vec![init; ni + 1]; SLOTS];
    let mut bits = 0f64;
    let mut cx = Vec::new();
    let mut st = vec![0f64; ni + 1];
    for i in 0..res.len() {
        cx.clear();
        ctx_of(i, res, &mut cx);
        let v = zigzag(res[i]) + 1;
        let nb = 31 - v.leading_zeros();
        let mut code = |slot: usize, bit: u32, tabs: &mut Vec<Tab>, bits: &mut f64| {
            for (k, t) in tabs.iter().enumerate() {
                st[k] = stretch(t.get(cx[k], slot));
            }
            st[ni] = 0.3;
            let w = &mut wts[slot];
            let dot: f64 = if ni == 1 { st[0] } else { st.iter().zip(w.iter()).map(|(a, b)| a * b).sum() };
            let p = squash(dot).clamp(1.0 / 4096.0, 1.0 - 1.0 / 4096.0);
            *bits -= if bit == 1 { p.log2() } else { (1.0 - p).log2() };
            if ni > 1 {
                let err = f64::from(bit) - p;
                for (wk, sk) in w.iter_mut().zip(st.iter()) {
                    *wk += lr * err * sk;
                }
            }
            for (k, t) in tabs.iter_mut().enumerate() {
                t.upd(cx[k], slot, bit);
            }
        };
        for j in 0..nb as usize {
            code(j, 1, &mut tabs, &mut bits);
        }
        code(nb as usize, 0, &mut tabs, &mut bits);
        if nb >= 1 {
            let top = (v >> (nb - 1)) & 1;
            code(33 + nb as usize, top, &mut tabs, &mut bits);
            if nb >= 2 {
                let second = (v >> (nb - 2)) & 1;
                code(66 + nb as usize * 2 + top as usize, second, &mut tabs, &mut bits);
                if nb >= 3 {
                    bits += f64::from(nb - 2);
                }
            }
        }
    }
    bits
}

/// Residual model variants on the residuals of the production predictor.
pub fn model(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let refs = Refs { files: &p.files, meta: &p.meta, tables: &p.tables, store: None };
    let sm = &p.meta.sources[k];
    let pre = sm.preset();
    let (s, maxt) = (p.meta.segment, pre.taps);
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let use_ref = !sm.refs.is_empty();
    // (residuals, reference window) per segment.
    let data: Vec<(Vec<i32>, Vec<f32>)> = segs
        .par_iter()
        .map(|&q| -> Result<(Vec<i32>, Vec<f32>)> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&p.files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            Ok((y.iter().zip(&preds).map(|(a, b)| a - b).collect(), win))
        })
        .collect::<Result<_>>()?;
    let nsamp: usize = data.iter().map(|d| d.0.len()).sum();
    println!("target {} ({} refs), {} segments, {} samples", sm.name, sm.refs.len(), data.len(), nsamp);
    let real: f64 = data.iter().map(|d| seg::encode_residuals(&d.0).len() as f64 * 8.0).sum::<f64>() / nsamp as f64;
    println!("  actual range coder (rfx today)                 {:.4} bits/sample", real);
    let a = 48usize;
    let variants: Vec<(&str, Vec<usize>, f32, f64)> = vec![
        ("one context (avg 1/16), counted probs", vec![a], 32.0, 0.0),
        ("one context, slower adaptation", vec![a], 128.0, 0.0),
        ("avg 1/16 + avg 1/4 mixed", vec![a, a], 64.0, 0.003),
        ("avg 1/16 + 1/4 + 1/128 mixed", vec![a, a, a], 64.0, 0.003),
        ("... + last two residual classes", vec![a, a, a, 400], 64.0, 0.003),
        ("... + reference envelope", vec![a, a, a, 400, 24 * 8], 64.0, 0.003),
        ("... + reference slope", vec![a, a, a, 400, 24 * 8, 24 * 8], 64.0, 0.003),
    ];
    for (name, sizes, limit, lr) in &variants {
        let bits: f64 = data
            .par_iter()
            .map(|(res, win)| {
                let (mut a1, mut a2, mut a3) = (64u32, 64u32, 64u32);
                let mut last = [0u32; 2];
                model_bits(res, sizes, *limit, *lr, |i, past, cx| {
                    if i > 0 {
                        let u = zigzag(past[i - 1]);
                        a1 = a1 - (a1 >> 4) + u.min(1 << 20);
                        a2 = a2 - (a2 >> 2) + u.min(1 << 20);
                        a3 = a3 - (a3 >> 7) + u.min(1 << 20);
                        last = [u, last[0]];
                    }
                    cx.push(log_bin(a1, 48));
                    if sizes.len() > 1 {
                        cx.push(log_bin(a2, 48));
                    }
                    if sizes.len() > 2 {
                        cx.push(log_bin(a3, 48));
                    }
                    if sizes.len() > 3 {
                        cx.push(log_bin(last[0], 20) * 20 + log_bin(last[1], 20));
                    }
                    if sizes.len() > 4 {
                        let c = i + maxt;
                        let lo = c.saturating_sub(8);
                        let hi = (c + 9).min(win.len());
                        let env = win[lo..hi].iter().fold(0f32, |m, v| m.max(v.abs())) as u32;
                        cx.push(log_bin(env, 24) * 8 + (a1.min(255) >> 5) as usize);
                    }
                    if sizes.len() > 5 {
                        let c = i + maxt;
                        let sl = (win[c + 1] - win[c - 1]).abs() as u32;
                        cx.push(log_bin(sl, 24) * 8 + (a1.min(255) >> 5) as usize);
                    }
                })
            })
            .sum();
        println!("  {:<46} {:.4} bits/sample", name, bits / nsamp as f64);
    }
    Ok(())
}
