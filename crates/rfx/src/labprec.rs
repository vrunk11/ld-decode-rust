//! Development: for a source without references (own past only), what do a
//! longer history, f64 arithmetic and a more often updated fit save against
//! the f32 predictor of the codec? The f64 numbers are not bit exact; the
//! question is what a bigger variable could buy.
use crate::io::read_words;
use crate::labspec::casc_predict;
use crate::pack::{prepare, PackOpts};
use crate::pred::Preset;
use crate::seg;
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

pub fn precision(inputs: &[PathBuf], o: &PackOpts, k: usize, nseg: usize) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let sm = &p.meta.sources[k];
    if !sm.refs.is_empty() {
        bail!("{} has references: this experiment is for a source on its own past", sm.name);
    }
    let s = p.meta.segment;
    let total = (sm.words as usize).div_ceil(s);
    let segs: Vec<usize> = (0..nseg).map(|i| total / 10 + (total * 8 / 10) * i / nseg.max(1)).collect();
    let ys: Vec<Vec<i32>> = segs
        .par_iter()
        .map(|&q| -> Result<Vec<i32>> {
            let s0 = q * s;
            let n = s.min(sm.words as usize - s0);
            Ok(read_words(&p.files[k], sm.words, s0 as i64, n, sm.shift)?.into_iter().map(|v| v - sm.mean).collect())
        })
        .collect::<Result<_>>()?;
    let n: f64 = ys.iter().map(|y| y.len() as f64).sum();
    let bits = |r: &[i32]| seg::encode_residuals(r).len() as f64 * 8.0;
    let resid = |y: &[i32], pr: &[i32]| -> Vec<i32> { y.iter().zip(pr).map(|(a, b)| a - b).collect() };
    println!("{}: {} segments, own past only", sm.name, ys.len());
    println!("  {:<44} bits/sample", "predictor");
    let f32_run = |solo: usize| -> f64 {
        let pre = Preset { solo, ..sm.preset() };
        ys.par_iter().map(|y| bits(&resid(y, &seg::predict_all(y, &pre, false, &[])))).sum::<f64>() / n
    };
    let f64_run = |ord: usize, stride: usize, every: usize, mem: f64| -> f64 {
        ys.par_iter().map(|y| bits(&resid(y, &casc_predict(y, ord, stride, every, mem)))).sum::<f64>() / n
    };
    for ord in [16usize, 24, 32] {
        println!("  f32 codec, {ord} own samples, update 4 / solve 64: {:.4}", f32_run(ord));
        println!("  f64,       {ord} own samples, update 4 / solve 64: {:.4}", f64_run(ord, 4, 64, 1024.0));
    }
    println!("  f64, 32 own samples, update 1 / solve 16:      {:.4}", f64_run(32, 1, 16, 1024.0));
    println!("  f64, 32 own samples, memory 2048:              {:.4}", f64_run(32, 4, 64, 2048.0));
    println!("  f64, 32 own samples, memory 512:               {:.4}", f64_run(32, 4, 64, 512.0));
    println!("  f64, 48 own samples, update 4 / solve 64:      {:.4}", f64_run(48, 4, 64, 1024.0));
    println!("  f64, 64 own samples, update 4 / solve 64:      {:.4}", f64_run(64, 4, 64, 1024.0));
    println!("  f64, 48 own samples, update 2 / solve 32:      {:.4}", f64_run(48, 2, 32, 1024.0));
    println!("  f64, 64 own samples, update 2 / solve 32:      {:.4}", f64_run(64, 2, 32, 1024.0));
    println!("  f64, 64 own samples, update 4 / solve 16:      {:.4}", f64_run(64, 4, 16, 1024.0));
    println!("  f64, 96 own samples, update 4 / solve 64:      {:.4}", f64_run(96, 4, 64, 1024.0));
    Ok(())
}
