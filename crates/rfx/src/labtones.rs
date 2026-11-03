//! Development: subtract fitted tones (a clock, a spur) from the signal or
//! from the predictor's residual, rounded so that the decoder can add exactly
//! the same integers back, and count what the coder then needs, side
//! information included.
use crate::io::read_words;
use crate::pack::{prepare, PackOpts, Refs};
use crate::seg;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const BLOCK: usize = 16384;

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for s in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (s + k, s + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

/// The `k` strongest narrow lines of the averaged spectrum of `segs`, in
/// cycles per sample (to the bin).
fn find_lines(segs: &[Vec<f64>], k: usize) -> Vec<f64> {
    let win: Vec<f64> = (0..BLOCK).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / BLOCK as f64).cos()).collect();
    let mut acc = vec![0f64; BLOCK / 2 + 1];
    let mut blocks = 0usize;
    for s in segs {
        for b in s.chunks_exact(BLOCK) {
            let mut re: Vec<f64> = b.iter().zip(&win).map(|(a, w)| a * w).collect();
            let mut im = vec![0f64; BLOCK];
            fft(&mut re, &mut im);
            for i in 0..=BLOCK / 2 {
                acc[i] += re[i] * re[i] + im[i] * im[i];
            }
            blocks += 1;
        }
    }
    for a in acc.iter_mut() {
        *a /= blocks.max(1) as f64;
    }
    let mut cand: Vec<(usize, f64)> = Vec::new();
    for i in 8..BLOCK / 2 - 8 {
        let mut nb: Vec<f64> = (i.saturating_sub(40)..(i + 41).min(BLOCK / 2)).filter(|&q| (q as i64 - i as i64).abs() > 3).map(|q| acc[q]).collect();
        nb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = nb[nb.len() / 2];
        let local_max = (i - 2..=i + 2).all(|q| acc[q] <= acc[i]);
        if local_max && acc[i] > 6.0 * med {
            cand.push((i, acc[i] - med));
        }
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    cand.into_iter().take(k).map(|c| c.0 as f64 / BLOCK as f64).collect()
}

/// Power of `x` at frequency `f` (cycles per sample), and the cosine and sine
/// sums.
fn dft(x: &[f64], f: f64) -> (f64, f64, f64) {
    let w = 2.0 * std::f64::consts::PI * f;
    let (wr, wi) = (w.cos(), w.sin());
    let (mut cr, mut ci) = (1.0f64, 0.0f64);
    let (mut sc, mut ss) = (0f64, 0f64);
    for (i, &v) in x.iter().enumerate() {
        sc += v * cr;
        ss += v * ci;
        let nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
        if i & 1023 == 1023 {
            let a = w * (i + 1) as f64;
            cr = a.cos();
            ci = a.sin();
        }
    }
    (sc * sc + ss * ss, sc, ss)
}

/// Best frequency near `f0` (within one coarse bin) for `x`.
fn refine(x: &[f64], f0: f64) -> f64 {
    let (mut lo, mut hi) = (f0 - 1.0 / BLOCK as f64, f0 + 1.0 / BLOCK as f64);
    let g = 0.618_033_988_75;
    let (mut a, mut b) = (hi - g * (hi - lo), lo + g * (hi - lo));
    let (mut pa, mut pb) = (dft(x, a).0, dft(x, b).0);
    for _ in 0..22 {
        if pa > pb {
            hi = b;
            b = a;
            pb = pa;
            a = hi - g * (hi - lo);
            pa = dft(x, a).0;
        } else {
            lo = a;
            a = b;
            pa = pb;
            b = lo + g * (hi - lo);
            pb = dft(x, b).0;
        }
    }
    (lo + hi) / 2.0
}

/// Fit tones at (refined) frequencies `lines` to `x` one after another
/// (matching pursuit); returns the integer signal to subtract and the number
/// of tones kept.
fn fit(x: &[f64], lines: &[f64]) -> Vec<i32> {
    let n = x.len();
    let mut rest = x.to_vec();
    let mut tone = vec![0f64; n];
    for &f0 in lines {
        let f = refine(&rest, f0);
        let (_, sc, ss) = dft(&rest, f);
        let (a, b) = (2.0 * sc / n as f64, 2.0 * ss / n as f64);
        let w = 2.0 * std::f64::consts::PI * f;
        for i in 0..n {
            let t = a * (w * i as f64).cos() + b * (w * i as f64).sin();
            tone[i] += t;
            rest[i] -= t;
        }
    }
    tone.iter().map(|t| t.round() as i32).collect()
}

pub fn tones(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize, ntones: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let refs = Refs { files: &p.files, meta: &p.meta, tables: &p.tables };
    let sm = &p.meta.sources[k];
    let pre = sm.preset();
    let s = p.meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let use_ref = !sm.refs.is_empty();
    // (samples, residual, bits of the residual)
    let data: Vec<(Vec<i32>, Vec<i32>, f64)> = segs
        .par_iter()
        .map(|&q| -> Result<(Vec<i32>, Vec<i32>, f64)> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&p.files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            let e: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            let bits = seg::encode_residuals(&e).len() as f64 * 8.0 / n as f64;
            Ok((y, e, bits))
        })
        .collect::<Result<_>>()?;
    let base = data.iter().map(|d| d.2).sum::<f64>() / data.len() as f64;
    println!("capture {} ({} references), {} segments, {} tones: baseline {:.4} bits/sample", sm.name, sm.refs.len(), data.len(), ntones, base);
    // Side information: frequency (4 bytes), two amplitudes (4 bytes), per tone per segment.
    let side = |n: usize| (ntones * 8) as f64 * 8.0 / n as f64;

    // A: tones fitted to the residual of the predictor.
    let res: Vec<Vec<f64>> = data.iter().map(|d| d.1.iter().map(|&v| f64::from(v)).collect()).collect();
    let lines = find_lines(&res, ntones);
    let a: f64 = res
        .par_iter()
        .map(|e| {
            let t = fit(e, &lines);
            let e2: Vec<i32> = e.iter().zip(&t).map(|(v, t)| *v as i32 - t).collect();
            seg::encode_residuals(&e2).len() as f64 * 8.0 / e.len() as f64
        })
        .sum::<f64>()
        / res.len() as f64;
    println!("  A  tones taken off the residual:           {:.4} bits/sample + {:.4} side = {:.4} ({:+.2}%)", a, side(s), a + side(s), ((a + side(s)) / base - 1.0) * 100.0);

    // B: tones taken off the signal before the predictor (without references only).
    if !use_ref {
        let sig: Vec<Vec<f64>> = data.iter().map(|d| d.0.iter().map(|&v| f64::from(v)).collect()).collect();
        let lines = find_lines(&sig, ntones);
        let b: f64 = sig
            .par_iter()
            .map(|y| {
                let t = fit(y, &lines);
                let y2: Vec<i32> = y.iter().zip(&t).map(|(v, t)| *v as i32 - t).collect();
                let preds = seg::predict_all(&y2, &pre, false, &[]);
                let r: Vec<i32> = y2.iter().zip(&preds).map(|(a, b)| a - b).collect();
                seg::encode_residuals(&r).len() as f64 * 8.0 / y.len() as f64
            })
            .sum::<f64>()
            / sig.len() as f64;
        println!("  B  tones taken off the signal, then predicted: {:.4} + {:.4} side = {:.4} ({:+.2}%)", b, side(s), b + side(s), ((b + side(s)) / base - 1.0) * 100.0);
    }
    Ok(())
}
