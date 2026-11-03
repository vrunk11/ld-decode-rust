//! Development: is there anything narrowband and stationary (a clock tone, a
//! spur) left in the residual that a subtracted sinusoid could take out? The
//! averaged power spectrum of the signal and of the residual of the
//! production predictor, with the spectral flatness (the bits a perfect
//! linear predictor would still save) and the narrow lines found.
use crate::io::read_words;
use crate::pack::{prepare, PackOpts, Refs};
use crate::seg;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const N: usize = 16384;

pub(crate) fn fft(re: &mut [f64], im: &mut [f64]) {
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

/// Averaged power spectrum (bins 0..=N/2) of `x`, Hann blocks of N.
fn psd(x: &[f64]) -> (Vec<f64>, usize) {
    let win: Vec<f64> = (0..N).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos()).collect();
    let mut acc = vec![0f64; N / 2 + 1];
    let mut blocks = 0;
    for b in x.chunks_exact(N) {
        let mut re: Vec<f64> = b.iter().zip(&win).map(|(a, w)| a * w).collect();
        let mut im = vec![0f64; N];
        fft(&mut re, &mut im);
        for k in 0..=N / 2 {
            acc[k] += re[k] * re[k] + im[k] * im[k];
        }
        blocks += 1;
    }
    for a in acc.iter_mut() {
        *a /= blocks.max(1) as f64;
    }
    (acc, blocks)
}

fn report(what: &str, x: &[f64], rate_mhz: f64) {
    let (p, blocks) = psd(x);
    let bins: Vec<usize> = (4..N / 2 - 4).collect();
    let am = bins.iter().map(|&k| p[k]).sum::<f64>() / bins.len() as f64;
    let gm = (bins.iter().map(|&k| p[k].max(1e-30).ln()).sum::<f64>() / bins.len() as f64).exp();
    println!("{what}: {blocks} blocks of {N}; spectral flatness AM/GM {:.3} = {:.3} bits/sample a perfect linear predictor would still save", am / gm, 0.5 * (am / gm).log2());
    // Narrow lines: a bin well above the median of its neighbourhood.
    let mut lines: Vec<(usize, f64, f64)> = Vec::new();
    let mut excess = 0f64;
    for &k in &bins {
        let lo = k.saturating_sub(40);
        let hi = (k + 41).min(N / 2);
        let mut nb: Vec<f64> = (lo..hi).filter(|&q| (q as i64 - k as i64).abs() > 3).map(|q| p[q]).collect();
        nb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = nb[nb.len() / 2];
        if p[k] > 6.0 * med {
            lines.push((k, p[k] / med, p[k] - med));
            excess += p[k] - med;
        }
    }
    let total: f64 = bins.iter().map(|&k| p[k]).sum();
    println!("  {} bins stand 6x above their neighbourhood; they hold {:.3}% of the power; removing them would save at most {:.4} bits/sample", lines.len(), 100.0 * excess / total, -0.5 * (1.0 - excess / total).log2());
    lines.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (k, r, e) in lines.iter().take(8) {
        println!("    {:8.4} MHz  {:6.1}x the neighbourhood  {:.4}% of the power", *k as f64 * rate_mhz / N as f64, r, 100.0 * e / total);
    }
}

pub fn spectrum(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let refs = Refs { files: &p.files, meta: &p.meta, tables: &p.tables };
    let sm = &p.meta.sources[k];
    let pre = sm.preset();
    let s = p.meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let use_ref = !sm.refs.is_empty();
    let data: Vec<(Vec<f64>, Vec<f64>)> = segs
        .par_iter()
        .map(|&q| -> Result<(Vec<f64>, Vec<f64>)> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&p.files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            Ok((y.iter().map(|&v| f64::from(v)).collect(), y.iter().zip(&preds).map(|(a, b)| f64::from(a - b)).collect()))
        })
        .collect::<Result<_>>()?;
    let sig: Vec<f64> = data.iter().flat_map(|d| d.0.iter().copied()).collect();
    let res: Vec<f64> = data.iter().flat_map(|d| d.1.iter().copied()).collect();
    println!("capture {} ({} references), {} segments, sample rate taken as {} MHz", sm.name, sm.refs.len(), segs.len(), o.rate_mhz);
    report("signal  ", &sig, o.rate_mhz);
    report("residual", &res, o.rate_mhz);
    Ok(())
}

/// Exponentially weighted least squares predictor of `e[i]` from the `p`
/// samples before it (f64, no exactness: an experiment).
pub(crate) fn casc_predict(e: &[i32], p: usize, stride: usize, every: usize, mem: f64) -> Vec<i32> {
    let lam = (1.0f64 - 1.0 / mem).powi(stride as i32);
    let mut a = vec![0f64; p * p];
    let mut b = vec![0f64; p];
    let mut w = vec![0f64; p];
    let mut r = vec![0f64; p];
    let mut out = Vec::with_capacity(e.len());
    for i in 0..e.len() {
        let dot: f64 = r.iter().zip(&w).map(|(x, y)| x * y).sum();
        out.push((dot + 0.5).floor() as i32);
        let yf = f64::from(e[i]);
        if i & (stride - 1) == 0 {
            for x in 0..p {
                b[x] = lam * b[x] + r[x] * yf;
                for q in 0..=x {
                    a[x * p + q] = lam * a[x * p + q] + r[x] * r[q];
                }
            }
        }
        if i & (every - 1) == every - 1 && i > 4 * p {
            if let Some(s) = chol(&a, &b, p) {
                w = s;
            }
        }
        r.copy_within(0..p - 1, 1);
        r[0] = yf;
    }
    out
}

fn chol(a: &[f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    let tr: f64 = (0..d).map(|i| a[i * d + i]).sum();
    let ridge = tr / d as f64 * 1e-6 + 1e-6;
    let mut l = vec![0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut v = a[i * d + j] + if i == j { ridge } else { 0.0 };
            for k in 0..j {
                v -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if v <= 0.0 {
                    return None;
                }
                l[i * d + i] = v.sqrt();
            } else {
                l[i * d + j] = v / l[j * d + j];
            }
        }
    }
    let mut x = b.to_vec();
    for i in 0..d {
        for k in 0..i {
            x[i] -= l[i * d + k] * x[k];
        }
        x[i] /= l[i * d + i];
    }
    for i in (0..d).rev() {
        for k in i + 1..d {
            x[i] -= l[k * d + i] * x[k];
        }
        x[i] /= l[i * d + i];
    }
    Some(x)
}

/// A second predictor on the residual of the first (a cascade): what it saves.
pub fn cascade(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let refs = Refs { files: &p.files, meta: &p.meta, tables: &p.tables };
    let sm = &p.meta.sources[k];
    let pre = sm.preset();
    let s = p.meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let use_ref = !sm.refs.is_empty();
    let orders = [8usize, 16, 32];
    let rows: Vec<Vec<f64>> = segs
        .par_iter()
        .map(|&q| -> Result<Vec<f64>> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&p.files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            let e: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            let bits = |r: &[i32]| seg::encode_residuals(r).len() as f64 * 8.0 / n as f64;
            let mut c = vec![bits(&e)];
            for &ord in &orders {
                let p2 = casc_predict(&e, ord, 4, 64, 1024.0);
                let r2: Vec<i32> = e.iter().zip(&p2).map(|(a, b)| a - b).collect();
                c.push(bits(&r2));
            }
            Ok(c)
        })
        .collect::<Result<_>>()?;
    let m = |c: usize| rows.iter().map(|r| r[c]).sum::<f64>() / rows.len() as f64;
    println!("capture {} ({} references): residual {:.4} bits/sample", sm.name, sm.refs.len(), m(0));
    for (i, ord) in orders.iter().enumerate() {
        println!("  + second stage of order {:>2}: {:.4}  ({:+.2}%)", ord, m(i + 1), (m(i + 1) / m(0) - 1.0) * 100.0);
    }
    Ok(())
}
