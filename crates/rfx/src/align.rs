//! The lag tables of an archive: an integer lag of a track against an earlier
//! one (`lag(p)`: where, in the earlier track, the position `p` of this one
//! is) that changes by whole samples where the drift between the two captures
//! adds one. The tables are stored in the file, range coded.
use crate::rc::{Decoder, Encoder, P2};
use anyhow::{bail, Context, Result};


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
}
