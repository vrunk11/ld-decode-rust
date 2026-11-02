//! Coarse alignment between captures, to the sample: a table of integer lags
//! (`lag(p)`: where, in an earlier source, the position `p` of this one
//! is) that changes by whole samples where the drift between the two
//! captures adds one. It is what `--tbc read` does with the line positions of
//! pass 0, without the fraction: the predictor's taps take what is left.
//!
//! The line positions come from ld-decode's pass 0 (`LD_DUMP_PASS0`): for
//! each field its frame code (VBI) and the sample position of every line.
//! The tables are stored in the file, so unpacking needs no pass 0.
use crate::rc::{Decoder, Encoder, P2};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Field {
    pub first: bool,
    pub code: Option<String>,
    pub lines: Vec<f64>,
}

pub fn parse_pass0(path: &Path) -> Result<Vec<Field>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for l in text.lines() {
        let (first, rest) = if let Some(r) = l.strip_prefix("true ") {
            (true, r)
        } else if let Some(r) = l.strip_prefix("false ") {
            (false, r)
        } else {
            bail!("{}: unexpected line in the pass 0 dump", path.display());
        };
        let (code, rest) = if let Some(r) = rest.strip_prefix("None") {
            (None, r)
        } else {
            let e = rest.find("])").context("bad frame code in the pass 0 dump")? + 2;
            (Some(rest[..e].to_string()), &rest[e..])
        };
        let mut lines = Vec::new();
        for h in rest.split_whitespace() {
            lines.push(f64::from_bits(u64::from_str_radix(h, 16).context("bad line position")?));
        }
        out.push(Field { first, code, lines });
    }
    Ok(out)
}

/// Run ld-decode's pass 0 on the captures and read back the dumps.
pub fn run_pass0(ld_decode: &Path, inputs: &[PathBuf], rate_mhz: f64, format: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut cmd = std::process::Command::new(ld_decode);
    cmd.env("LD_DUMP_PASS0", dir).env("LD_PASS0_ONLY", "1").args(["--format", format, "--inputfreq", &rate_mhz.to_string()]);
    for i in inputs {
        cmd.arg("-i").arg(i);
    }
    cmd.args(["-l", "1", "-o"]).arg(dir.join("pass0_probe"));
    let out = cmd.output().with_context(|| format!("starting {}", ld_decode.display()))?;
    if !out.status.success() {
        bail!("ld-decode failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(())
}

/// Integer lag of a source against an earlier one, as a step function of the
/// position: `lag[i]` holds from `pos[i]` up to `pos[i + 1]`.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct LagTable {
    pub pos: Vec<u64>,
    pub lag: Vec<i64>,
}

impl LagTable {
    pub fn lag_at(&self, p: u64) -> i64 {
        let i = self.pos.partition_point(|&q| q <= p);
        self.lag[i.saturating_sub(1)]
    }

    /// Smallest and largest lag over positions `[a, b)`.
    pub fn range(&self, a: u64, b: u64) -> (i64, i64) {
        let first = self.pos.partition_point(|&q| q <= a).saturating_sub(1);
        let (mut lo, mut hi) = (self.lag[first], self.lag[first]);
        let mut i = first + 1;
        while i < self.pos.len() && self.pos[i] < b {
            lo = lo.min(self.lag[i]);
            hi = hi.max(self.lag[i]);
            i += 1;
        }
        (lo, hi)
    }

    /// Compact form: a few varints, then a range-coded stream. The positions
    /// are multiples of a unit (the grid), their gaps are coded against the
    /// previous gap (the drift between two captures changes slowly), and a
    /// step of the lag is nearly always +-1 in the direction of the last one.
    pub fn encode(&self) -> Vec<u8> {
        let n = self.pos.len();
        let unit = self.pos.windows(2).map(|w| w[1] - w[0]).fold(0, gcd).max(1);
        let mut out = Vec::new();
        put_varint(&mut out, n as u64);
        put_varint(&mut out, unit);
        if n == 0 {
            return out;
        }
        put_varint(&mut out, self.pos[0]);
        put_varint(&mut out, zigzag(self.lag[0]));
        let mut enc = Encoder::new();
        let mut m = TableModel::new();
        for i in 1..n {
            let gap = (self.pos[i] - self.pos[i - 1]) / unit;
            m.put_gap(&mut enc, gap);
            m.put_step(&mut enc, self.lag[i] - self.lag[i - 1]);
        }
        out.extend(enc.finish());
        out
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut at = 0;
        let n = get_varint(buf, &mut at)? as usize;
        let unit = get_varint(buf, &mut at)?;
        let mut t = Self::default();
        if n == 0 {
            bail!("a lag table must start at position 0");
        }
        let p0 = get_varint(buf, &mut at)?;
        let l0 = unzigzag(get_varint(buf, &mut at)?);
        if p0 != 0 {
            bail!("a lag table must start at position 0");
        }
        t.pos.push(p0);
        t.lag.push(l0);
        let mut dec = Decoder::new(&buf[at..]);
        let mut m = TableModel::new();
        for _ in 1..n {
            let gap = m.get_gap(&mut dec);
            let step = m.get_step(&mut dec);
            if dec.overrun() {
                bail!("truncated lag table");
            }
            t.pos.push(t.pos[t.pos.len() - 1] + gap * unit);
            t.lag.push(t.lag[t.lag.len() - 1].wrapping_add(step));
        }
        Ok(t)
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn bits_of(v: u64) -> usize {
    (64 - v.leading_zeros()) as usize
}

/// Adaptive model of a lag table's gaps and steps (see [`LagTable::encode`]).
struct TableModel {
    cls: Vec<[P2; 65]>,
    mant: [P2; 65],
    sign: [P2; 3],
    esc: P2,
    same: [P2; 2],
    smag: [[P2; 3]; 4],
    prev_mag: usize,
    prev_gap: u64,
    prev_sign: usize,
    prev_up: usize,
}

impl TableModel {
    fn new() -> Self {
        Self { cls: vec![[P2::INIT; 65]; 24], mant: [P2::INIT; 65], sign: [P2::INIT; 3], esc: P2::INIT, same: [P2::INIT; 2], smag: [[P2::INIT; 3]; 4], prev_mag: 0, prev_gap: 1, prev_sign: 0, prev_up: 0 }
    }

    fn ctx(&self) -> usize {
        bits_of(self.prev_gap).min(23)
    }

    fn put_gap(&mut self, enc: &mut Encoder, gap: u64) {
        let d = gap as i64 - self.prev_gap as i64;
        let mag = d.unsigned_abs();
        let nb = bits_of(mag);
        let c = self.ctx();
        for j in 0..nb {
            enc.bit2(&mut self.cls[c][j], 1);
        }
        if nb < 64 {
            enc.bit2(&mut self.cls[c][nb], 0);
        }
        if nb >= 2 {
            enc.bit2(&mut self.mant[nb], ((mag >> (nb - 2)) & 1) as u32);
            put_direct(enc, mag & ((1u64 << (nb - 2)) - 1), nb - 2);
        }
        if nb >= 1 {
            let neg = u32::from(d < 0);
            enc.bit2(&mut self.sign[self.prev_sign], neg);
            self.prev_sign = 1 + neg as usize;
        } else {
            self.prev_sign = 0;
        }
        self.prev_gap = gap;
    }

    fn get_gap(&mut self, dec: &mut Decoder) -> u64 {
        let c = self.ctx();
        let mut nb = 0;
        while nb < 64 && dec.bit2(&mut self.cls[c][nb]) == 1 {
            nb += 1;
        }
        let mut mag = u64::from(nb >= 1);
        if nb >= 2 {
            mag = (mag << 1) | u64::from(dec.bit2(&mut self.mant[nb]));
            mag = (mag << (nb - 2)) | get_direct(dec, nb - 2);
        }
        let mut d = mag as i64;
        if nb >= 1 {
            let neg = dec.bit2(&mut self.sign[self.prev_sign]);
            if neg == 1 {
                d = d.wrapping_neg();
            }
            self.prev_sign = 1 + neg as usize;
        } else {
            self.prev_sign = 0;
        }
        self.prev_gap = (self.prev_gap as i64).wrapping_add(d) as u64;
        self.prev_gap
    }

    fn put_step(&mut self, enc: &mut Encoder, step: i64) {
        let mag = step.unsigned_abs();
        if (1..=4).contains(&mag) {
            enc.bit2(&mut self.esc, 0);
            for j in 1..mag {
                enc.bit2(&mut self.smag[self.prev_mag][j as usize - 1], 1);
            }
            if mag < 4 {
                enc.bit2(&mut self.smag[self.prev_mag][mag as usize - 1], 0);
            }
            let up = u32::from(step > 0);
            enc.bit2(&mut self.same[self.prev_up], up);
            self.prev_up = up as usize;
            self.prev_mag = mag as usize - 1;
        } else {
            enc.bit2(&mut self.esc, 1);
            let z = zigzag(step);
            enc.direct(bits_of(z) as u32, 7);
            put_direct(enc, z, bits_of(z));
        }
    }

    fn get_step(&mut self, dec: &mut Decoder) -> i64 {
        if dec.bit2(&mut self.esc) == 0 {
            let mut mag = 1usize;
            while mag < 4 && dec.bit2(&mut self.smag[self.prev_mag][mag - 1]) == 1 {
                mag += 1;
            }
            let up = dec.bit2(&mut self.same[self.prev_up]);
            self.prev_up = up as usize;
            self.prev_mag = mag - 1;
            if up == 1 {
                mag as i64
            } else {
                -(mag as i64)
            }
        } else {
            let nb = (dec.direct(7) as usize).min(64);
            unzigzag(get_direct(dec, nb))
        }
    }
}

/// `n` (<= 64) equiprobable bits of `v`, high first.
fn put_direct(enc: &mut Encoder, v: u64, n: usize) {
    if n > 32 {
        enc.direct((v >> 32) as u32 & ((1u64 << (n - 32)) - 1) as u32, (n - 32) as u32);
        enc.direct(v as u32, 32);
    } else if n > 0 {
        enc.direct(v as u32 & (((1u64 << n) - 1) as u32), n as u32);
    }
}

fn get_direct(dec: &mut Decoder, n: usize) -> u64 {
    if n > 32 {
        let hi = u64::from(dec.direct((n - 32) as u32));
        (hi << 32) | u64::from(dec.direct(32))
    } else if n > 0 {
        u64::from(dec.direct(n as u32))
    } else {
        0
    }
}

pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn get_varint(buf: &[u8], at: &mut usize) -> Result<u64> {
    let (mut v, mut shift) = (0u64, 0);
    loop {
        let b = *buf.get(*at).context("truncated varint")?;
        *at += 1;
        v |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
        if shift > 63 {
            bail!("varint too long");
        }
    }
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

const GRID: u64 = 64;
/// The lag must leave the nearest integer by more than this to change.
pub const HYST_FINE: f64 = 0.75;
/// Coarser tables (half as many steps, a fifth of the bytes) for captures whose
/// predictor has a channel for each of its best references, which absorb the
/// misalignment.
pub const HYST_COARSE: f64 = 1.5;
/// Lines over which the line-to-line lag is averaged.
const SMOOTH_LINES: f64 = 16.0;

/// Lag of `tgt` (sample positions in the later source) against `rf`, from
/// the fields both located with the same frame code. `None` when they share
/// no field.
pub fn build_table(tgt: &[Field], rf: &[Field], tgt_words: u64, hyst: f64) -> Option<LagTable> {
    let mut first_rf: HashMap<(&str, bool), usize> = HashMap::new();
    for (i, f) in rf.iter().enumerate() {
        if let Some(c) = &f.code {
            first_rf.entry((c.as_str(), f.first)).or_insert(i);
        }
    }
    let mut seen: HashMap<(&str, bool), ()> = HashMap::new();
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for f in tgt {
        let Some(c) = &f.code else { continue };
        let key = (c.as_str(), f.first);
        if seen.insert(key, ()).is_some() {
            continue;
        }
        if let Some(&j) = first_rf.get(&key) {
            for (a, b) in f.lines.iter().zip(&rf[j].lines) {
                pts.push((*a, *b));
            }
        }
    }
    if pts.len() < 2 {
        return None;
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pts.dedup_by(|a, b| a.0 == b.0);
    let line_len = {
        let l = &tgt.iter().find(|f| f.lines.len() > 10)?.lines;
        (l[l.len() - 1] - l[0]) / (l.len() - 1) as f64
    };
    let ng = (tgt_words / GRID + 2) as usize;
    let at = |p: f64| -> f64 {
        if p <= pts[0].0 {
            return pts[0].1 - pts[0].0;
        }
        let last = pts[pts.len() - 1];
        if p >= last.0 {
            return last.1 - last.0;
        }
        let i = pts.partition_point(|&(a, _)| a < p).clamp(1, pts.len() - 1);
        let (a0, b0) = pts[i - 1];
        let (a1, b1) = pts[i];
        b0 + (b1 - b0) * (p - a0) / (a1 - a0) - p
    };
    let raw: Vec<f64> = (0..ng).map(|g| at((g as u64 * GRID) as f64)).collect();
    let mut pre = vec![0f64; ng + 1];
    for g in 0..ng {
        pre[g + 1] = pre[g] + raw[g];
    }
    let h = (SMOOTH_LINES * line_len / 2.0 / GRID as f64) as usize;
    let mut t = LagTable::default();
    let mut d = 0i64;
    for g in 0..ng {
        let (a, b) = (g.saturating_sub(h), (g + h + 1).min(ng));
        let l = (pre[b] - pre[a]) / (b - a) as f64;
        if g == 0 || (l - d as f64).abs() > hyst {
            let nd = l.round() as i64;
            if g == 0 || nd != d {
                t.pos.push(g as u64 * GRID);
                t.lag.push(nd);
                d = nd;
            }
        }
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_roundtrip_and_lookup() {
        let t = LagTable { pos: vec![0, 100, 5000, 5001], lag: vec![-7, -6, 400000, -3] };
        assert_eq!(LagTable::decode(&t.encode()).unwrap(), t);
        assert_eq!(t.lag_at(0), -7);
        assert_eq!(t.lag_at(99), -7);
        assert_eq!(t.lag_at(100), -6);
        assert_eq!(t.lag_at(6000), -3);
        assert_eq!(t.range(50, 5001), (-7, 400000));
        assert_eq!(t.range(200, 300), (-6, -6));
    }

    #[test]
    fn coded_table_roundtrip_with_drift_and_jumps() {
        let mut t = LagTable { pos: vec![0], lag: vec![-123456789012] };
        let (mut p, mut l, mut x) = (0u64, t.lag[0], 7u32);
        for i in 0..5000u64 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            p += 64 * (90 + (i % 40) + u64::from(x >> 29));
            l += if i % 700 == 699 { 1 << 33 } else if i % 900 == 5 { -77 } else if (i / 300) % 2 == 0 { 1 } else { -1 };
            t.pos.push(p);
            t.lag.push(l);
        }
        let bytes = t.encode();
        assert_eq!(LagTable::decode(&bytes).unwrap(), t);
        assert!(bytes.len() < 5000 * 2, "{} bytes", bytes.len());
    }

    #[test]
    fn steady_drift_gives_one_slip_per_sample_of_drift() {
        // The later source runs 1e-3 slower: 100 lines of 1000 samples.
        let mk = |scale: f64, off: f64| Field {
            first: true,
            code: Some("[1]".into()),
            lines: (0..100).map(|i| off + i as f64 * 1000.0 * scale).collect(),
        };
        let tgt = [mk(1.0, 0.0)];
        let rf = [mk(1.001, 50.0)];
        let t = build_table(&tgt, &rf, 100_000, HYST_FINE).unwrap();
        // The lag grows from 50 to about 50 + 99 (the ends are averaged inward).
        assert!((50..=60).contains(&t.lag[0]), "{}", t.lag[0]);
        let last = *t.lag.last().unwrap();
        assert!((140..=150).contains(&last), "{last}");
    }
}
