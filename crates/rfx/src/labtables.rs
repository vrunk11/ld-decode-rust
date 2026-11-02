//! Development: what the lag tables hold and what they could cost.
use crate::pack::{prepare, PackOpts};
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn log2_bin(v: u64) -> usize {
    (64 - v.leading_zeros()) as usize
}

/// Ideal bits of an adaptive frequency count over `alphabet` symbols.
struct Freq {
    c: Vec<f64>,
    total: f64,
}

impl Freq {
    fn new(n: usize) -> Self {
        Self { c: vec![0.4; n], total: 0.4 * n as f64 }
    }
    fn code(&mut self, s: usize) -> f64 {
        let b = -(self.c[s] / self.total).log2();
        self.c[s] += 1.0;
        self.total += 1.0;
        b
    }
}

pub fn tables(inputs: &[PathBuf], o: &PackOpts) -> Result<()> {
    let p = prepare(inputs, Path::new("lab.rfx"), o)?;
    let mut keys: Vec<_> = p.tables.keys().copied().collect();
    keys.sort();
    let (mut bytes, mut entries, mut ideal, mut ideal_b) = (0usize, 0usize, 0f64, 0f64);
    let mut dl_hist: BTreeMap<i64, usize> = BTreeMap::new();
    for (n, key) in keys.iter().enumerate() {
        let t = &p.tables[key];
        let e = t.encode().len();
        bytes += e;
        entries += t.pos.len();
        // Position gaps by log2 class (the mantissa bits raw), step as a
        // symbol (clamped), each with the previous step as its context.
        let mut gap = Freq::new(40);
        let mut step: Vec<Freq> = (0..9).map(|_| Freq::new(9)).collect();
        let mut dgap: Vec<Freq> = (0..40).map(|_| Freq::new(40)).collect();
        let mut bits_b = 0f64;
        let mut prev_gap = 0u64;
        let (mut pp, mut pl, mut ps) = (0u64, 0i64, 4usize);
        let mut bits = 0f64;
        for (&q, &l) in t.pos.iter().zip(&t.lag) {
            let g = q - pp;
            {
                let d = g as i64 - prev_gap as i64;
                let cls = log2_bin(d.unsigned_abs());
                bits_b += dgap[log2_bin(prev_gap)].code(cls) + cls.saturating_sub(1) as f64 + if d != 0 { 1.0 } else { 0.0 };
                prev_gap = g;
            }
            let cls = log2_bin(g);
            bits += gap.code(cls) + cls.saturating_sub(1) as f64;
            let d = (l - pl).clamp(-4, 4);
            let s = (d + 4) as usize;
            bits += step[ps].code(s);
            *dl_hist.entry(l - pl).or_default() += 1;
            ps = s;
            pp = q;
            pl = l;
        }
        ideal += bits / 8.0;
        ideal_b += bits_b / 8.0 + (t.pos.len() as f64 * 0.1 / 8.0);
        if n < 8 {
            let gaps: Vec<u64> = t.pos.windows(2).take(24).map(|w| w[1] - w[0]).collect();
            let lags: Vec<i64> = t.lag.iter().take(24).copied().collect();
            println!("  ({}, {}): {} entries, {} bytes; first gaps {:?}; first lags {:?}", key.0, key.1, t.pos.len(), e, gaps, lags);
        }
    }
    println!("gap-delta model: {:.0} bytes (step cost not included beyond 0.1 bit)", ideal_b);
    println!("{} tables, {} entries, {} bytes as stored, {:.0} bytes with an adaptive model", keys.len(), entries, bytes, ideal);
    let total: usize = dl_hist.values().sum();
    let mut top: Vec<_> = dl_hist.into_iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.1));
    for (d, c) in top.iter().take(8) {
        println!("  step {:+}: {} ({:.1}%)", d, c, 100.0 * *c as f64 / total as f64);
    }
    Ok(())
}
