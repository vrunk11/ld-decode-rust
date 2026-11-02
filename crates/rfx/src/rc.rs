//! Adaptive binary range coder (the LZMA construction): 11-bit
//! probabilities, 32-bit range, carry through a cached byte. Integer only,
//! so an encoder and a decoder agree on any machine.

pub const PROB_INIT: u16 = 1024;
const TOP: u32 = 1 << 24;

pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    pub out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    #[inline]
    pub fn bit(&mut self, prob: &mut u16, bit: u32) {
        let bound = (self.range >> 11) * u32::from(*prob);
        if bit == 0 {
            self.range = bound;
            *prob += (2048 - *prob) >> 5;
        } else {
            self.low += u64::from(bound);
            self.range -= bound;
            *prob -= *prob >> 5;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// `n` equiprobable bits, high first.
    pub fn direct(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.range >>= 1;
            if (value >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self { data, pos: 1, range: 0xFFFF_FFFF, code: 0 };
        for _ in 0..4 {
            d.code = (d.code << 8) | u32::from(d.next());
        }
        d
    }

    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    pub fn bit(&mut self, prob: &mut u16) -> u32 {
        let bound = (self.range >> 11) * u32::from(*prob);
        let bit;
        if self.code < bound {
            self.range = bound;
            *prob += (2048 - *prob) >> 5;
            bit = 0;
        } else {
            self.code -= bound;
            self.range -= bound;
            *prob -= *prob >> 5;
            bit = 1;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next());
        }
        bit
    }

    pub fn direct(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            self.range >>= 1;
            let b = if self.code >= self.range {
                self.code -= self.range;
                1
            } else {
                0
            };
            v = (v << 1) | b;
            while self.range < TOP {
                self.range <<= 8;
                self.code = (self.code << 8) | u32::from(self.next());
            }
        }
        v
    }

    /// A corrupt stream runs far past the end of its data.
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() + 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut probs = [PROB_INIT; 4];
        let mut e = Encoder::new();
        let mut x = 12345u32;
        let mut want = Vec::new();
        for i in 0..20000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = u32::from((x >> 20) % 7 == 0);
            want.push((b, i % 4, x >> 28));
            e.bit(&mut probs[i % 4], b);
            e.direct(x >> 28, 4);
        }
        let bytes = e.finish();
        let mut probs = [PROB_INIT; 4];
        let mut d = Decoder::new(&bytes);
        for (b, p, v) in want {
            assert_eq!(d.bit(&mut probs[p]), b);
            assert_eq!(d.direct(4), v);
        }
    }
}
