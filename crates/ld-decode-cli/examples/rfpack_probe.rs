// Probe for an RF archive that codes several captures of one disc as
// differences: how many bits does a capture cost when it is predicted from
// its own past and from the already-coded captures?
//
//   rfpack_probe <target.s16> <target pass0.txt> <ref.s16> <ref pass0.txt> [...]
//     [--field F] [--block B] [--own P] [--taps K] [--max N]
//
// The pass0 files come from `LD_DUMP_PASS0=<dir> ld-decode -i a -i b ...`; they
// give, for each capture, the position of every line of each field, so the
// coarse delay between two captures is read from the line positions. Within a
// block the delay is the nearest integer; the prediction filter (taps around
// that delay in each reference) absorbs the fractional part and the jitter.
//
// Cost = Rice-coded residual (FLAC's partitions of 256) plus 13 bits per
// coefficient per block. Least squares per block, forward adaptive: an upper
// bound on what a cheaper adaptive filter gives, not the codec itself.
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

struct Field {
    first: bool,
    code: Option<String>,
    lines: Vec<f64>,
}

fn parse_pass0(path: &str) -> Vec<Field> {
    let text = std::fs::read_to_string(path).expect("pass0");
    let mut out = Vec::new();
    for l in text.lines() {
        let (first, rest) = if let Some(r) = l.strip_prefix("true ") { (true, r) } else { (false, l.strip_prefix("false ").unwrap()) };
        let (code, rest) = if let Some(r) = rest.strip_prefix("None") {
            (None, r)
        } else {
            let e = rest.find("])").unwrap() + 2;
            (Some(rest[..e].to_string()), &rest[e..])
        };
        let lines = rest.split_whitespace().map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap())).collect();
        out.push(Field { first, code, lines });
    }
    out
}

fn key(f: &Field) -> Option<(String, bool)> {
    f.code.as_ref().map(|c| (c.clone(), f.first))
}

fn read_s16(path: &str, start: i64, n: usize, shift: u32) -> Vec<f64> {
    let mut f = File::open(path).unwrap();
    let len = f.metadata().unwrap().len() as i64 / 2;
    let mut out = vec![0.0; n];
    let lo = start.max(0);
    let hi = (start + n as i64).min(len);
    if hi <= lo {
        return out;
    }
    f.seek(SeekFrom::Start(lo as u64 * 2)).unwrap();
    let mut buf = vec![0u8; (hi - lo) as usize * 2];
    f.read_exact(&mut buf).unwrap();
    for (i, w) in buf.chunks_exact(2).enumerate() {
        out[(lo - start) as usize + i] = (i16::from_le_bytes([w[0], w[1]]) >> shift) as f64;
    }
    out
}

/// Piecewise-linear map target position -> reference position.
struct Delay(Vec<(f64, f64)>);
impl Delay {
    fn at(&self, p: f64) -> f64 {
        let v = &self.0;
        let i = v.partition_point(|&(a, _)| a < p).clamp(1, v.len() - 1);
        let (a0, b0) = v[i - 1];
        let (a1, b1) = v[i];
        b0 + (b1 - b0) * (p - a0) / (a1 - a0)
    }
}

fn rice_bits(e: &[i32]) -> f64 {
    let mut bits = 0f64;
    for part in e.chunks(256) {
        let mut best = u64::MAX;
        for k in 0..22u32 {
            let mut s = 0u64;
            for &v in part {
                let u = ((v << 1) ^ (v >> 31)) as u32 as u64;
                s += (u >> k) + 1 + k as u64;
            }
            best = best.min(s);
        }
        bits += best as f64 + 5.0;
    }
    bits
}

fn solve(a: &mut [f64], b: &mut [f64], d: usize) -> bool {
    let tr: f64 = (0..d).map(|i| a[i * d + i]).sum::<f64>() / d as f64;
    for i in 0..d {
        a[i * d + i] += tr * 1e-9 + 1e-9;
    }
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= a[i * d + k] * a[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                a[i * d + i] = s.sqrt();
            } else {
                a[i * d + j] = s / a[j * d + j];
            }
        }
    }
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * d + k] * b[k];
        }
        b[i] = s / a[i * d + i];
    }
    for i in (0..d).rev() {
        let mut s = b[i];
        for k in i + 1..d {
            s -= a[k * d + i] * b[k];
        }
        b[i] = s / a[i * d + i];
    }
    true
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut field_i, mut block, mut own, mut taps, mut maxn) = (150usize, 8192usize, 32usize, 16i64, usize::MAX);
    let mut aligned_mode = false;
    let mut adapt = false;
    let mut smooth = 0f64;
    let mut hyst = 0.25f64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--field" => field_i = it.next().unwrap().parse().unwrap(),
            "--block" => block = it.next().unwrap().parse().unwrap(),
            "--own" => own = it.next().unwrap().parse().unwrap(),
            "--taps" => taps = it.next().unwrap().parse().unwrap(),
            "--max" => maxn = it.next().unwrap().parse().unwrap(),
            "--aligned" => aligned_mode = true,
            "--adapt" => adapt = true,
            "--smooth" => smooth = it.next().unwrap().parse().unwrap(),
            "--hyst" => hyst = it.next().unwrap().parse().unwrap(),
            _ => pos.push(a.clone()),
        }
    }
    assert!(pos.len() >= 2 && pos.len() % 2 == 0);
    let srcs: Vec<(String, Vec<Field>)> = pos.chunks(2).map(|c| (c[0].clone(), parse_pass0(&c[1]))).collect();
    let tgt = &srcs[0].1;
    let refs = &srcs[1..];

    // Delay maps from every field of the target that all references share.
    let maps: Vec<HashMap<(String, bool), usize>> = refs
        .iter()
        .map(|(_, f)| {
            let mut m = HashMap::new();
            for (i, x) in f.iter().enumerate() {
                if let Some(k) = key(x) {
                    m.entry(k).or_insert(i);
                }
            }
            m
        })
        .collect();
    let mut delays: Vec<Delay> = (0..refs.len()).map(|_| Delay(Vec::new())).collect();
    let mut tg_first: HashMap<(String, bool), usize> = HashMap::new();
    for (i, f) in tgt.iter().enumerate() {
        if let Some(k) = key(f) {
            tg_first.entry(k).or_insert(i);
        }
    }
    let mut common = Vec::new();
    for (i, f) in tgt.iter().enumerate() {
        let Some(k) = key(f) else { continue };
        if tg_first[&k] != i || !maps.iter().all(|m| m.contains_key(&k)) {
            continue;
        }
        common.push(i);
        for (r, m) in maps.iter().enumerate() {
            let rf = &refs[r].1[m[&k]];
            for (a, b) in f.lines.iter().zip(&rf.lines) {
                delays[r].0.push((*a, *b));
            }
        }
    }
    for d in &mut delays {
        d.0.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.0.dedup_by(|a, b| a.0 == b.0);
    }
    eprintln!("{} fields shared by all inputs", common.len());
    let fi = common[field_i.min(common.len() - 1)];
    let n0 = tgt[fi].lines[2] as i64;
    let n1 = (tgt[fi].lines[240] as i64).min(n0 + maxn as i64);
    let n = ((n1 - n0) as usize / block) * block;
    eprintln!("window: field {fi}, samples {n0}..{} ({n} samples)", n0 + n as i64);

    let probe = read_s16(&srcs[0].0, n0, n, 0);
    let shift = probe.iter().fold(0i64, |a, &v| a | v as i64).trailing_zeros().min(15);
    eprintln!("wasted low bits: {shift}");
    let xt = read_s16(&srcs[0].0, n0, n, shift);
    let mean = (xt.iter().sum::<f64>() / n as f64).round();
    std::fs::write(
        std::env::temp_dir().join("rfpack_target.s16"),
        xt.iter().map(|&v| ((v as i32) << shift).clamp(-32768, 32767) as i16).flat_map(i16::to_le_bytes).collect::<Vec<u8>>(),
    )
    .unwrap();

    // Reference windows, read wide enough for their delay and the taps.
    let mut ref_x: Vec<(i64, Vec<f64>)> = Vec::new();
    for (r, (path, _)) in refs.iter().enumerate() {
        let a = delays[r].at(n0 as f64) as i64 - 4096;
        let b = delays[r].at((n0 + n as i64) as f64) as i64 + 4096;
        let lo = a.min(b) - 4096;
        let len = ((a.max(b) - lo) as usize) + n + 8192;
        ref_x.push((lo, read_s16(path, lo, len, shift)));
    }

    const PAD: usize = 64;
    let mut aligned: Vec<Vec<f64>> = Vec::new();
    for (j, (lo, xs)) in ref_x.iter().enumerate() {
        let mut al = vec![0f64; n + 2 * PAD];
        let m = n + 2 * PAD;
        let raw: Vec<f64> = (0..m).map(|i| { let p = (n0 + i as i64 - PAD as i64) as f64; delays[j].at(p) - p }).collect();
        let h = (smooth * 4767.0 / 2.0) as usize;
        let mut pre = vec![0f64; m + 1];
        for i in 0..m { pre[i + 1] = pre[i] + raw[i]; }
        let lv: Vec<f64> = (0..m).map(|i| { let a = i.saturating_sub(h); let b = (i + h + 1).min(m); (pre[b] - pre[a]) / (b - a) as f64 }).collect();
        let mut dcur: Option<i64> = None;
        let (mut changes, mut moved) = (0usize, 0i64);
        for i in 0..n + 2 * PAD {
            let p = (n0 + i as i64 - PAD as i64) as f64;
            let l = lv[i];
            let d = match dcur {
                Some(d) if (l - d as f64).abs() <= 0.5 + hyst => d,
                _ => l.round() as i64,
            };
            if let Some(old) = dcur {
                if old != d {
                    changes += 1;
                    moved += (d - old).abs();
                }
            }
            dcur = Some(d);
            let ix = p as i64 + d - lo;
            al[i] = if ix >= 0 && (ix as usize) < xs.len() { xs[ix as usize] } else { 0.0 };
        }
        eprintln!("ref {}: {} slips, {} samples skipped or repeated ({:.5} per sample)", j + 1, changes, moved, moved as f64 / n as f64);
        aligned.push(al);
    }
    println!("target {}  samples {n}  block {block}  {}", srcs[0].0, if aligned_mode { "aligned (integer address map)" } else { "block lag" });
    println!("{:<34} {:>9} {:>9} {:>9}", "variant", "bits/smp", "resid", "rms");
    let variants: Vec<(String, usize, i64)> = {
        let mut v = vec![("own only".to_string(), 0, 0)];
        for r in 1..=refs.len() {
            for &k in &[0i64, 2, taps] {
                v.push((format!("own + {r} ref(s), +-{k} taps"), r, k));
            }
        }
        v
    };
    let raw: Vec<i32> = xt.iter().map(|&v| (v - mean) as i32).collect();
    println!("{:<34} {:>9.3}", "no prediction (Rice)", rice_bits(&raw) / n as f64);
    for (name, nr, k) in variants {
        let per_ref = (2 * k + 1) as usize;
        let d = 1 + own + nr * per_ref;
        let mut resid: Vec<i32> = Vec::with_capacity(n);
        let mut coef_bits = 0f64;
        let mut r = vec![0f64; d];
        for s in (0..n).step_by(block) {
            let centre = (n0 + (s + block / 2) as i64) as f64;
            let dl: Vec<i64> = (0..nr).map(|j| (delays[j].at(centre) - centre).round() as i64).collect();
            let regress = |i: usize, r: &mut [f64]| {
                r[0] = 1.0;
                for p in 0..own {
                    let idx = s as i64 + i as i64 - 1 - p as i64;
                    r[1 + p] = if idx >= 0 { xt[idx as usize] - mean } else { 0.0 };
                }
                for j in 0..nr {
                    let (lo, ref xs) = ref_x[j];
                    for t in 0..per_ref {
                        if aligned_mode {
                            let ix = (s + i + PAD) as i64 + t as i64 - k;
                            r[1 + own + j * per_ref + t] = aligned[j][ix as usize] - mean;
                            continue;
                        }
                        let abs = n0 + (s + i) as i64 + dl[j] + t as i64 - k;
                        let ix = abs - lo;
                        r[1 + own + j * per_ref + t] = if ix >= 0 && (ix as usize) < xs.len() { xs[ix as usize] - mean } else { 0.0 };
                    }
                }
            };
            let mut a = vec![0f64; d * d];
            let mut b = vec![0f64; d];
            for i in 0..block {
                regress(i, &mut r);
                let y = xt[s + i] - mean;
                for p in 0..d {
                    let rp = r[p];
                    b[p] += rp * y;
                    for q in 0..=p {
                        a[p * d + q] += rp * r[q];
                    }
                }
            }
            for p in 0..d {
                for q in 0..p {
                    a[q * d + p] = a[p * d + q];
                }
            }
            let mut w = b.clone();
            let ok = solve(&mut a, &mut w, d);
            if !ok {
                w.iter_mut().for_each(|v| *v = 0.0);
            }
            coef_bits += d as f64 * 13.0;
            for i in 0..block {
                regress(i, &mut r);
                let pred = r.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>().round();
                resid.push((xt[s + i] - mean - pred) as i32);
            }
        }
        let rb = rice_bits(&resid);
        let rms = (resid.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
        println!("{:<34} {:>9.3} {:>9.3} {:>9.2}", name, (rb + coef_bits) / n as f64, rb / n as f64, rms);
    }
    if adapt {
        // The references are independent noisy copies: their mean is the
        // sufficient statistic, so one averaged channel can stand for all.
        let m = aligned[0].len();
        let mean_ref: Vec<f64> = (0..m).map(|i| aligned.iter().map(|a| a[i]).sum::<f64>() / aligned.len() as f64).collect();
        let mean_set = vec![mean_ref];
        println!("-- backward adaptive, cost vs bits ({} refs) --", refs.len());
        println!("{:<52} {:>8} {:>8} {:>10}", "variant", "bits/smp", "rms", "Msmp/s");
        // (label, averaged, own, taps, memory, stride, every)
        let runs: [(&str, bool, usize, i64, f64, usize, usize); 9] = [
            ("all refs, own 32, +-6, every sample", false, 32, 6, 1024.0, 1, 32),
            ("mean ref, own 32, +-6, every sample", true, 32, 6, 1024.0, 1, 32),
            ("mean ref, own 32, +-6, stride 2", true, 32, 6, 1024.0, 2, 32),
            ("mean ref, own 32, +-6, stride 4", true, 32, 6, 1024.0, 4, 64),
            ("mean ref, own 16, +-6, stride 4", true, 16, 6, 1024.0, 4, 64),
            ("mean ref, own 16, +-4, stride 4", true, 16, 4, 1024.0, 4, 64),
            ("mean ref, own 16, +-4, stride 8", true, 16, 4, 1024.0, 8, 128),
            ("mean ref, own 8, +-3, stride 8", true, 8, 3, 1024.0, 8, 128),
            ("mean ref, own 8, +-3, stride 8, memory 4096", true, 8, 3, 4096.0, 8, 128),
        ];
        for (name, avg, own_o, k, mem, stride, every) in runs {
            let t0 = std::time::Instant::now();
            let (set, nr): (&[Vec<f64>], usize) = if avg { (&mean_set, 1) } else { (&aligned, refs.len()) };
            let resid = if std::env::var_os("RFPACK_F32").is_some() { run_ls_f32(&xt, mean, set, nr, k, own_o, 1.0 - 1.0 / mem, every, stride) } else { run_ls(&xt, mean, set, nr, k, own_o, 1.0 - 1.0 / mem, every, stride) };
            let secs = t0.elapsed().as_secs_f64();
            let rms = (resid.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
            println!("{:<52} {:>8.3} {:>8.2} {:>10.2}", name, rice_bits(&resid) / n as f64, rms, n as f64 / secs / 1e6);
        }
    }
}

/// Exponentially weighted least squares on the past only (what a decoder can
/// redo without any coefficient being sent): the covariance is updated every
/// `stride` samples and the weights re-solved every `every` samples.
fn run_ls(xt: &[f64], mean: f64, aligned: &[Vec<f64>], nr: usize, k: i64, own: usize, lambda: f64, every: usize, stride: usize) -> Vec<i32> {
    const PAD: usize = 64;
    let n = xt.len();
    let per_ref = (2 * k + 1) as usize;
    let d = own + nr * per_ref;
    let lam = lambda.powi(stride as i32);
    let mut a = vec![0f64; d * d];
    let mut b = vec![0f64; d];
    let mut w = vec![0f64; d];
    let mut r = vec![0f64; d];
    let mut m = vec![0f64; d * d];
    let mut resid = Vec::with_capacity(n);
    for i in 0..n {
        for p in 0..own {
            r[p] = if i > p { xt[i - 1 - p] - mean } else { 0.0 };
        }
        for j in 0..nr {
            for t in 0..per_ref {
                r[own + j * per_ref + t] = aligned[j][i + PAD + t - k as usize] - mean;
            }
        }
        let y = xt[i] - mean;
        let pred = r.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>().round();
        resid.push((y - pred) as i32);
        if i % stride == 0 {
            for p in 0..d {
                let rp = r[p];
                b[p] = lam * b[p] + rp * y;
                let row = &mut a[p * d..p * d + p + 1];
                for (q, v) in row.iter_mut().enumerate() {
                    *v = lam * *v + rp * r[q];
                }
            }
        }
        if i % every == every - 1 && i > 4 * d {
            for p in 0..d {
                for q in 0..=p {
                    m[p * d + q] = a[p * d + q];
                    m[q * d + p] = a[p * d + q];
                }
            }
            let mut s = b.clone();
            if solve(&mut m, &mut s, d) {
                w = s;
            }
        }
    }
    resid
}
fn solve32(a: &mut [f32], b: &mut [f32], d: usize) -> bool {
    let tr: f32 = (0..d).map(|i| a[i * d + i]).sum::<f32>() / d as f32;
    for i in 0..d {
        a[i * d + i] += tr * 1e-6 + 1e-6;
    }
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= a[i * d + k] * a[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                a[i * d + i] = s.sqrt();
            } else {
                a[i * d + j] = s / a[j * d + j];
            }
        }
    }
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * d + k] * b[k];
        }
        b[i] = s / a[i * d + i];
    }
    for i in (0..d).rev() {
        let mut s = b[i];
        for k in i + 1..d {
            s -= a[k * d + i] * b[k];
        }
        b[i] = s / a[i * d + i];
    }
    true
}
fn run_ls32(xt: &[f32], mean: f32, aligned: &[Vec<f32>], nr: usize, k: i64, own: usize, lambda: f32, every: usize, stride: usize) -> Vec<i32> {
    const PAD: usize = 64;
    let n = xt.len();
    let per_ref = (2 * k + 1) as usize;
    let d = own + nr * per_ref;
    let lam = lambda.powi(stride as i32);
    let mut a = vec![0f32; d * d];
    let mut b = vec![0f32; d];
    let mut w = vec![0f32; d];
    let mut r = vec![0f32; d];
    let mut m = vec![0f32; d * d];
    let mut resid = Vec::with_capacity(n);
    for i in 0..n {
        for p in 0..own {
            r[p] = if i > p { xt[i - 1 - p] - mean } else { 0.0 };
        }
        for j in 0..nr {
            for t in 0..per_ref {
                r[own + j * per_ref + t] = aligned[j][i + PAD + t - k as usize] - mean;
            }
        }
        let y = xt[i] - mean;
        let pred = r.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>().round();
        resid.push((y - pred) as i32);
        if i % stride == 0 {
            for p in 0..d {
                let rp = r[p];
                b[p] = lam * b[p] + rp * y;
                let row = &mut a[p * d..p * d + p + 1];
                for (q, v) in row.iter_mut().enumerate() {
                    *v = lam * *v + rp * r[q];
                }
            }
        }
        if i % every == every - 1 && i > 4 * d {
            for p in 0..d {
                for q in 0..=p {
                    m[p * d + q] = a[p * d + q];
                    m[q * d + p] = a[p * d + q];
                }
            }
            let mut s = b.clone();
            if solve32(&mut m, &mut s, d) {
                w = s;
            }
        }
    }
    resid
}

fn run_ls_f32(xt: &[f64], mean: f64, aligned: &[Vec<f64>], nr: usize, k: i64, own: usize, lambda: f64, every: usize, stride: usize) -> Vec<i32> {
    let x32: Vec<f32> = xt.iter().map(|&v| v as f32).collect();
    let al32: Vec<Vec<f32>> = aligned.iter().map(|a| a.iter().map(|&v| v as f32).collect()).collect();
    run_ls32(&x32, mean as f32, &al32, nr, k, own, lambda as f32, every, stride)
}
