//! Residual model: a prediction error is zigzagged, shifted by one and sent
//! as an Elias-gamma class (adaptive unary, one probability set per context)
//! and the bits under the leading one (the first two adaptive, the rest
//! raw). The context is the magnitude of the recent residuals, so the cost
//! follows the noise level without any side information, and the model
//! depends on past residuals only (a decoder can parse every residual
//! before it knows a single prediction).
use crate::rc::{Decoder, Encoder, P2};

const NCTX: usize = 48;
const NCLS: usize = 33;

pub struct Model {
    unary: Box<[[P2; NCLS]; NCTX]>,
    m1: [P2; NCLS],
    m2: [[P2; 2]; NCLS],
    avg: u32,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self { unary: Box::new([[P2::INIT; NCLS]; NCTX]), m1: [P2::INIT; NCLS], m2: [[P2::INIT; 2]; NCLS], avg: 64 }
    }

    fn ctx(&self) -> usize {
        let bl = 32 - self.avg.leading_zeros();
        let second = if bl >= 2 { (self.avg >> (bl - 2)) & 1 } else { 0 };
        ((2 * bl + second) as usize).min(NCTX - 1)
    }

    fn adapt(&mut self, u: u32) {
        self.avg = self.avg - (self.avg >> 4) + u.min(1 << 20);
    }

    pub fn encode(&mut self, enc: &mut Encoder, e: i32) {
        let u = ((e << 1) ^ (e >> 31)) as u32;
        let v = u + 1;
        let nb = 31 - v.leading_zeros();
        let c = self.ctx();
        for i in 0..nb as usize {
            enc.bit2(&mut self.unary[c][i], 1);
        }
        enc.bit2(&mut self.unary[c][nb as usize], 0);
        if nb >= 1 {
            let top = (v >> (nb - 1)) & 1;
            enc.bit2(&mut self.m1[nb as usize], top);
            if nb >= 2 {
                let second = (v >> (nb - 2)) & 1;
                enc.bit2(&mut self.m2[nb as usize][top as usize], second);
                if nb >= 3 {
                    enc.direct(v & ((1 << (nb - 2)) - 1), nb - 2);
                }
            }
        }
        self.adapt(u);
    }

    pub fn decode(&mut self, dec: &mut Decoder) -> i32 {
        let c = self.ctx();
        let mut nb = 0usize;
        while nb < NCLS - 1 && dec.bit2(&mut self.unary[c][nb]) == 1 {
            nb += 1;
        }
        let mut v = 1u32;
        if nb >= 1 {
            let top = dec.bit2(&mut self.m1[nb]);
            v = (v << 1) | top;
            if nb >= 2 {
                let second = dec.bit2(&mut self.m2[nb][top as usize]);
                v = (v << 1) | second;
                if nb >= 3 {
                    v = (v << (nb - 2)) | dec.direct(nb as u32 - 2);
                }
            }
        }
        let u = v - 1;
        self.adapt(u);
        ((u >> 1) as i32) ^ -((u & 1) as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residuals_roundtrip() {
        let mut x = 7u32;
        let vals: Vec<i32> = (0..50000)
            .map(|i| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                let s = if i % 5000 < 2500 { 3 } else { 400 };
                ((x >> 16) as i32 % (2 * s + 1)) - s
            })
            .chain([0, 1, -1, 1 << 20, -(1 << 20)])
            .collect();
        let mut m = Model::new();
        let mut enc = Encoder::new();
        for &v in &vals {
            m.encode(&mut enc, v);
        }
        let bytes = enc.finish();
        let mut m = Model::new();
        let mut dec = Decoder::new(&bytes);
        for &v in &vals {
            assert_eq!(m.decode(&mut dec), v);
        }
    }
}
