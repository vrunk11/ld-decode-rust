//! Development: what is the residual of the predictor made of? Its
//! distribution against the Gaussian and Laplace shapes (with a causal local
//! scale, like the coder's), how its variance moves (slowly, or periodically
//! with the line), and how far the coder is from what those shapes allow.
use crate::io::read_words;
use crate::labspec::fft;
use crate::pack::{prepare, PackOpts, Refs};
use crate::seg;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn gauss_cdf(x: f64, s: f64) -> f64 {
    0.5 * (1.0 + erf(x / (s * std::f64::consts::SQRT_2)))
}

fn laplace_cdf(x: f64, s: f64) -> f64 {
    let b = s / std::f64::consts::SQRT_2;
    if x < 0.0 {
        0.5 * (x / b).exp()
    } else {
        1.0 - 0.5 * (-x / b).exp()
    }
}

/// Ideal bits per sample of `e` under a family whose scale is the causal
/// local one (exponential average of the squares, rate 1/`rate`).
fn ideal_bits(e: &[i32], rate: f64, cdf: impl Fn(f64, f64) -> f64) -> f64 {
    let mut s2 = 4.0;
    let mut bits = 0.0;
    for &v in e {
        let s = (s2 + 0.25f64).sqrt();
        let p = (cdf(f64::from(v) + 0.5, s) - cdf(f64::from(v) - 0.5, s)).max(1e-12);
        bits -= p.log2();
        s2 += (f64::from(v) * f64::from(v) - s2) / rate;
    }
    bits / e.len() as f64
}

pub fn residual(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let refs = Refs { files: &p.files, meta: &p.meta, tables: &p.tables };
    let sm = &p.meta.sources[k];
    let pre = sm.preset();
    let s = p.meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let use_ref = !sm.refs.is_empty();
    let data: Vec<Vec<i32>> = segs
        .par_iter()
        .map(|&q| -> Result<Vec<i32>> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            let y: Vec<i32> = read_words(&p.files[k], sm.fmt(), sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect();
            let win = if use_ref { refs.window(k, s0 as i64, n)? } else { Vec::new() };
            let preds = seg::predict_all(&y, &pre, use_ref, &win);
            Ok(y.iter().zip(&preds).map(|(a, b)| a - b).collect())
        })
        .collect::<Result<_>>()?;
    let all: Vec<f64> = data.iter().flatten().map(|&v| f64::from(v)).collect();
    let n = all.len() as f64;
    let mean = all.iter().sum::<f64>() / n;
    let var = all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = all.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    println!("capture {} ({} references), {} samples: mean {:.3}, sigma {:.3}, kurtosis {:.2} (Gaussian 3, Laplace 6)", sm.name, sm.refs.len(), all.len(), mean, var.sqrt(), m4 / var / var);

    let real = data.iter().map(|e| seg::encode_residuals(e).len() as f64 * 8.0).sum::<f64>() / n;
    println!("  coder today                                {:.4} bits/sample", real);
    for rate in [8.0, 16.0, 32.0, 64.0, 128.0] {
        let g = data.iter().map(|e| ideal_bits(e, rate, gauss_cdf) * e.len() as f64).sum::<f64>() / n;
        let l = data.iter().map(|e| ideal_bits(e, rate, laplace_cdf) * e.len() as f64).sum::<f64>() / n;
        println!("  ideal, local scale rate 1/{:<3}: Gaussian {:.4}  Laplace {:.4}", rate as u32, g, l);
    }

    // Standard deviation by block of 256 samples: how much the noise level moves.
    let mut sd: Vec<f64> = data.iter().flat_map(|e| e.chunks_exact(256).map(|c| (c.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 256.0).sqrt())).collect();
    sd.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| sd[((sd.len() - 1) as f64 * f) as usize];
    println!("  sigma per 256 samples: 5% {:.2}  25% {:.2}  median {:.2}  75% {:.2}  95% {:.2}  99.9% {:.2}", q(0.05), q(0.25), q(0.5), q(0.75), q(0.95), q(0.999));

    // Does the variance repeat? Spectrum of the energy by blocks of 32 samples.
    const B: usize = 32;
    let mut acc = vec![0f64; 4096 / 2 + 1];
    let mut cnt = 0;
    for e in &data {
        let nb = e.len() / B;
        if nb < 4096 {
            continue;
        }
        let v: Vec<f64> = e.chunks_exact(B).take(4096).map(|c| c.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / B as f64).collect();
        let m = v.iter().sum::<f64>() / 4096.0;
        let mut re: Vec<f64> = v.iter().map(|x| x - m).collect();
        let mut im = vec![0f64; 4096];
        fft(&mut re, &mut im);
        for i in 0..=2048 {
            acc[i] += re[i] * re[i] + im[i] * im[i];
        }
        cnt += 1;
    }
    if cnt > 0 {
        let tot: f64 = acc[1..].iter().sum();
        let mut pk: Vec<(usize, f64)> = (22..2020)
            .filter(|&i| acc[i] >= acc[i - 1] && acc[i] >= acc[i + 1] && acc[i] > 5.0 * acc[i.saturating_sub(20)..i + 20].iter().sum::<f64>() / 40.0)
            .map(|i| (i, acc[i]))
            .collect();
        pk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  periodic variance (energy by blocks of {B}), {cnt} segments; strongest periods:");
        for (i, v) in pk.iter().take(5) {
            println!("    period {:9.1} samples   {:.2}% of the variance-of-variance", 4096.0 * B as f64 / *i as f64, 100.0 * v / tot);
        }
        let low: f64 = acc[1..9].iter().sum::<f64>() / tot;
        println!("  slow drift (periods above {} samples): {:.1}% of the variance-of-variance", 4096 * B / 8, 100.0 * low);
    }
    Ok(())
}
