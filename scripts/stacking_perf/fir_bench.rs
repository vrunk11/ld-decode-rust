// Micro-benchmark of the read-time FIR in stack.rs::compute_task (per input sample):
// 32-tap real sinc + 128-tap complex analytic band, at a slowly varying fractional phase.
use std::time::Instant;
const PH: usize = 1024;
const ST: usize = 32;
const HT: usize = 128;
#[derive(Clone, Copy)]
struct C { re: f64, im: f64 }

fn main() {
    let n = 1_000_000usize;
    let raw: Vec<f32> = (0..n + 1000).map(|i| ((i * 7919) % 20011) as f32 - 10000.0).collect();
    let sinc: Vec<f64> = (0..(PH + 1) * ST).map(|i| ((i * 31) % 97) as f64 * 1e-3 - 0.04).collect();
    let hi: Vec<C> = (0..(PH + 1) * HT).map(|i| C { re: ((i * 17) % 89) as f64 * 1e-3 - 0.04, im: ((i * 13) % 83) as f64 * 1e-3 - 0.04 }).collect();
    let sinc32: Vec<f32> = sinc.iter().map(|&v| v as f32).collect();
    let hre: Vec<f32> = hi.iter().map(|c| c.re as f32).collect();
    let him: Vec<f32> = hi.iter().map(|c| c.im as f32).collect();
    let slope = 1.0003f64;
    let pos = |j: usize| 200.0 + j as f64 * slope;
    let mut full = vec![0f64; n];
    let mut hr = vec![0f64; n];
    let mut hm = vec![0f64; n];

    // A: as in the branch (serial accumulation, f64, interleaved complex LUT)
    let t = Instant::now();
    for j in 0..n {
        let p = pos(j); let k0 = p.floor(); let ph = ((p - k0) * PH as f64).round() as usize; let k0 = k0 as usize;
        let xs = &raw[k0 - (ST / 2 - 1)..][..ST];
        full[j] = sinc[ph * ST..][..ST].iter().zip(xs).map(|(h, &x)| h * f64::from(x)).sum();
        let xs = &raw[k0 - (HT / 2 - 1)..][..HT];
        let (mut ar, mut ai) = (0.0, 0.0);
        for (h, &x) in hi[ph * HT..][..HT].iter().zip(xs) { ar += h.re * f64::from(x); ai += h.im * f64::from(x); }
        hr[j] = ar; hm[j] = ai;
    }
    let ta = t.elapsed().as_secs_f64();
    let chk_a = full.iter().sum::<f64>() + hr.iter().sum::<f64>() + hm.iter().sum::<f64>();

    // B: planar f32 LUTs, 8 lanes
    let mut full2 = vec![0f32; n];
    let mut hr2 = vec![0f32; n];
    let mut hm2 = vec![0f32; n];
    let t = Instant::now();
    for j in 0..n {
        let p = pos(j); let k0 = p.floor(); let ph = ((p - k0) * PH as f64).round() as usize; let k0 = k0 as usize;
        let xs = &raw[k0 - (ST / 2 - 1)..][..ST];
        let mut f8 = [0f32; 8];
        for (hc, xc) in sinc32[ph * ST..][..ST].chunks_exact(8).zip(xs.chunks_exact(8)) { for l in 0..8 { f8[l] += hc[l] * xc[l]; } }
        full2[j] = f8.iter().sum();
        let xs = &raw[k0 - (HT / 2 - 1)..][..HT];
        let (mut ar, mut ai) = ([0f32; 8], [0f32; 8]);
        let rr = &hre[ph * HT..][..HT]; let ii = &him[ph * HT..][..HT];
        for ((rc, ic), xc) in rr.chunks_exact(8).zip(ii.chunks_exact(8)).zip(xs.chunks_exact(8)) { for l in 0..8 { ar[l] += rc[l] * xc[l]; ai[l] += ic[l] * xc[l]; } }
        hr2[j] = ar.iter().sum(); hm2[j] = ai.iter().sum();
    }
    let tb = t.elapsed().as_secs_f64();
    let chk_b = full2.iter().map(|&v| v as f64).sum::<f64>() + hr2.iter().map(|&v| v as f64).sum::<f64>() + hm2.iter().map(|&v| v as f64).sum::<f64>();
    let maxrel = (0..n).map(|j| ((hr[j] - hr2[j] as f64).abs() / (hr[j].abs() + 1.0)).max((full[j] - full2[j] as f64).abs() / (full[j].abs() + 1.0))).fold(0.0, f64::max);
    println!("A branch (serial f64, interleaved): {:.1} ns/input-sample  ({:.1} M/s)", ta / n as f64 * 1e9, n as f64 / ta / 1e6);
    println!("B planar f32, 8 lanes             : {:.1} ns/input-sample  ({:.1} M/s)  speedup {:.1}x  max rel diff {:.2e}", tb / n as f64 * 1e9, n as f64 / tb / 1e6, ta / tb, maxrel);
    println!("(checksums {:.3e} {:.3e})", chk_a, chk_b);
}
