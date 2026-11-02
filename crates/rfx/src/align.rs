//! Coarse alignment between captures, to the sample: a table of integer lags
//! (`lag(p)`: where, in an earlier source, the position `p` of this one
//! is) that changes by whole samples where the drift between the two
//! captures adds one. It is what `--tbc read` does with the line positions of
//! pass 0, without the fraction: the predictor's taps take what is left.
//!
//! The line positions come from ld-decode's pass 0 (`LD_DUMP_PASS0`): for
//! each field its frame code (VBI) and the sample position of every line.
//! The tables are stored in the file, so unpacking needs no pass 0.
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

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_varint(&mut out, self.pos.len() as u64);
        let (mut pp, mut pl) = (0u64, 0i64);
        for (&p, &l) in self.pos.iter().zip(&self.lag) {
            put_varint(&mut out, p - pp);
            put_varint(&mut out, zigzag(l - pl));
            pp = p;
            pl = l;
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut at = 0;
        let n = get_varint(buf, &mut at)? as usize;
        let (mut t, mut p, mut l) = (Self::default(), 0u64, 0i64);
        for _ in 0..n {
            p += get_varint(buf, &mut at)?;
            l += unzigzag(get_varint(buf, &mut at)?);
            t.pos.push(p);
            t.lag.push(l);
        }
        if n == 0 || t.pos[0] != 0 {
            bail!("a lag table must start at position 0");
        }
        Ok(t)
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
const HYSTERESIS: f64 = 0.75;
/// Lines over which the line-to-line lag is averaged.
const SMOOTH_LINES: f64 = 16.0;

/// Lag of `tgt` (sample positions in the later source) against `rf`, from
/// the fields both located with the same frame code. `None` when they share
/// no field.
pub fn build_table(tgt: &[Field], rf: &[Field], tgt_words: u64) -> Option<LagTable> {
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
        if g == 0 || (l - d as f64).abs() > HYSTERESIS {
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
    fn steady_drift_gives_one_slip_per_sample_of_drift() {
        // The later source runs 1e-3 slower: 100 lines of 1000 samples.
        let mk = |scale: f64, off: f64| Field {
            first: true,
            code: Some("[1]".into()),
            lines: (0..100).map(|i| off + i as f64 * 1000.0 * scale).collect(),
        };
        let tgt = [mk(1.0, 0.0)];
        let rf = [mk(1.001, 50.0)];
        let t = build_table(&tgt, &rf, 100_000).unwrap();
        // The lag grows from 50 to about 50 + 99 (the ends are averaged inward).
        assert!((50..=60).contains(&t.lag[0]), "{}", t.lag[0]);
        let last = *t.lag.last().unwrap();
        assert!((140..=150).contains(&last), "{last}");
    }
}
