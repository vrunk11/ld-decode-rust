//! How noisy is each capture? Captures of one programme carry the same
//! picture, so on the same frames of the disc a capture that costs more bits
//! to code on its own past alone is the noisier one. The windows are taken at
//! frames (VBI codes, from pass 0) that every capture has, so the content
//! cancels out of the comparison.
use crate::align::Field;
use crate::fmt::Fmt;
use crate::io::read_words;
use crate::pred::Preset;
use crate::seg;
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

/// What was measured on one capture.
#[derive(Clone, Debug)]
pub struct Score {
    /// Mean cost of coding it from its own past, bits per sample.
    pub bits: f64,
    pub min: f64,
    pub max: f64,
    /// RMS of the samples around their mean (gain and amplitude).
    pub rms: f64,
    /// Windows measured.
    pub windows: usize,
}

type Key = (String, bool);

fn first_by_key(f: &[Field]) -> HashMap<Key, usize> {
    let mut m = HashMap::new();
    for (i, x) in f.iter().enumerate() {
        if let Some(c) = &x.code {
            m.entry((c.clone(), x.first)).or_insert(i);
        }
    }
    m
}

/// Cost of the first `nwin` windows of `win` samples at frames all captures
/// share (taken evenly between 20% and 80% of the first capture's frames).
pub fn scores(inputs: &[PathBuf], fmts: &[Fmt], fields: &[&[Field]], nwin: usize, win: usize) -> Result<Vec<Score>> {
    let maps: Vec<HashMap<Key, usize>> = fields.iter().map(|f| first_by_key(f)).collect();
    let mut common: Vec<(f64, Key)> = maps[0]
        .iter()
        .filter(|(k, _)| maps.iter().all(|m| m.contains_key(*k)))
        .map(|(k, &i)| (fields[0][i].lines.first().copied().unwrap_or(0.0), k.clone()))
        .collect();
    common.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if common.len() < nwin * 4 {
        bail!("the captures share only {} frames: not enough to compare their noise", common.len());
    }
    let (lo, hi) = (common.len() / 5, common.len() * 4 / 5);
    let picks: Vec<&Key> = (0..nwin).map(|w| &common[lo + (hi - lo) * w / nwin].1).collect();
    let pre = Preset::level(3).expect("level 3");
    let files: Vec<File> = inputs.iter().map(File::open).collect::<std::io::Result<_>>()?;
    let work: Vec<(usize, usize)> = (0..inputs.len()).flat_map(|i| (0..nwin).map(move |w| (i, w))).collect();
    let measured: Vec<Option<(f64, f64)>> = work
        .par_iter()
        .map(|&(i, w)| {
            let size = files[i].metadata().ok()?.len();
            let words = fmts[i].samples(size);
            let f = &fields[i][maps[i][picks[w]]];
            let start = *f.lines.first()? as i64;
            if start < 0 || (start as u64) + win as u64 > words {
                return None;
            }
            let probe = read_words(&files[i], fmts[i], words, start, win, 0).ok()?;
            let shift = probe.iter().fold(0i32, |a, &v| a | v).trailing_zeros().min(15);
            let y: Vec<i32> = probe.iter().map(|&v| v >> shift).collect();
            let mean = (y.iter().map(|&v| i64::from(v)).sum::<i64>() as f64 / win as f64).round() as i32;
            let y: Vec<i32> = y.iter().map(|&v| v - mean).collect();
            let rms = (y.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / win as f64).sqrt();
            let preds = seg::predict_all(&y, &pre, false, &[]);
            let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
            let bits = seg::encode_residuals(&res).len() as f64 * 8.0 / win as f64;
            Some((bits, rms))
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..inputs.len() {
        let m: Vec<(f64, f64)> = (0..nwin).filter_map(|w| measured[i * nwin + w]).collect();
        if m.is_empty() {
            bail!("{}: no window inside the file", inputs[i].display());
        }
        let n = m.len() as f64;
        out.push(Score {
            bits: m.iter().map(|x| x.0).sum::<f64>() / n,
            min: m.iter().map(|x| x.0).fold(f64::MAX, f64::min),
            max: m.iter().map(|x| x.0).fold(f64::MIN, f64::max),
            rms: m.iter().map(|x| x.1).sum::<f64>() / n,
            windows: m.len(),
        });
    }
    Ok(out)
}

impl Score {
    /// The cost normalised by the amplitude: the noise against the signal,
    /// smaller is cleaner. (The raw cost is wrong: a capture with a stronger
    /// signal costs more bits and is the better reference.)
    pub fn cost(&self) -> f64 {
        self.bits - self.rms.max(1e-9).log2()
    }
}

/// Frames two captures must share to count as the same programme.
const MIN_SHARED: usize = 50;

/// The order to pack `inputs` in: captures that share frames form a group;
/// groups keep the place of their first capture and each one is sorted from
/// the cleanest capture to the noisiest, so that the best reference comes
/// first. A group whose noise cannot be compared (too few frames in common
/// to all its captures) keeps the order given. Returns the permutation (the
/// original index of each place) and the scores by original index.
pub fn plan(inputs: &[PathBuf], fmts: &[Fmt], fields: &[Vec<Field>]) -> (Vec<usize>, Vec<Option<f64>>) {
    let n = inputs.len();
    let maps: Vec<HashMap<Key, usize>> = fields.iter().map(|f| first_by_key(f)).collect();
    let mut group: Vec<usize> = (0..n).collect();
    fn find(g: &mut Vec<usize>, i: usize) -> usize {
        if g[i] != i {
            let r = find(g, g[i]);
            g[i] = r;
        }
        g[i]
    }
    for i in 0..n {
        for j in i + 1..n {
            let shared = maps[i].keys().filter(|k| maps[j].contains_key(*k)).count();
            if shared >= MIN_SHARED {
                let (a, b) = (find(&mut group, i), find(&mut group, j));
                group[a.max(b)] = a.min(b);
            }
        }
    }
    let mut members: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let root = find(&mut group, i);
        if root == i {
            members.push(vec![i]);
        } else {
            members.iter_mut().find(|m| m[0] == root).expect("the root comes first").push(i);
        }
    }
    let mut order = Vec::new();
    let mut by_input = vec![None; n];
    for m in members {
        if m.len() > 1 {
            let sub_inputs: Vec<PathBuf> = m.iter().map(|&i| inputs[i].clone()).collect();
            let sub_fields: Vec<&[Field]> = m.iter().map(|&i| fields[i].as_slice()).collect();
            let sub_fmts: Vec<Fmt> = m.iter().map(|&i| fmts[i]).collect();
            match scores(&sub_inputs, &sub_fmts, &sub_fields, 8, 262144) {
                Ok(sc) => {
                    let mut ranked: Vec<(usize, f64)> = m.iter().zip(&sc).map(|(&i, s)| (i, s.cost())).collect();
                    for &(i, c) in &ranked {
                        by_input[i] = Some(c);
                    }
                    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    order.extend(ranked.into_iter().map(|r| r.0));
                    continue;
                }
                Err(e) => eprintln!("order: keeping the given order for {} captures: {e}", m.len()),
            }
        }
        order.extend(m);
    }
    (order, by_input)
}
