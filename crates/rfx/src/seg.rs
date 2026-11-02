//! One segment of one source: its samples against the predictor, the
//! residuals through the range coder. A segment starts from nothing (empty
//! history, zero weights, fresh model), so segments are independent and are
//! coded and decoded in parallel.
use crate::model::Model;
use crate::pred::{Pred, Preset};
use crate::rc::{Decoder, Encoder};

/// Length the reference window must have for `n` samples.
pub fn refwin_len(n: usize, pre: &Preset) -> usize {
    n + 2 * pre.taps
}

fn run(n: usize, pre: &Preset, use_ref: bool, refwin: &[f32], mut step: impl FnMut(usize, i32) -> i32) -> Vec<i32> {
    assert!(!use_ref || refwin.len() >= refwin_len(n, pre));
    let mut pred = Pred::new(*pre, use_ref);
    let mut ys: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        let p = pred.predict(refwin, i);
        let y = step(i, p);
        ys.push(y);
        pred.update(i, y);
    }
    ys
}

/// Prediction of every sample of `y` from the samples before it.
pub fn predict_all(y: &[i32], pre: &Preset, use_ref: bool, refwin: &[f32]) -> Vec<i32> {
    let mut preds = Vec::with_capacity(y.len());
    run(y.len(), pre, use_ref, refwin, |i, p| {
        preds.push(p);
        y[i]
    });
    preds
}

/// The samples whose residuals (error against the prediction) are `res`.
pub fn reconstruct(res: &[i32], pre: &Preset, use_ref: bool, refwin: &[f32]) -> Vec<i32> {
    run(res.len(), pre, use_ref, refwin, |i, p| p + res[i])
}

/// Residuals to bytes. The model reads past residuals only, so the bytes
/// can be parsed without knowing any prediction.
pub fn encode_residuals(res: &[i32]) -> Vec<u8> {
    let mut enc = Encoder::new();
    let mut model = Model::new();
    for &e in res {
        model.encode(&mut enc, e);
    }
    enc.finish()
}

/// The inverse of [`encode_residuals`]; `None` if the bytes are not valid.
pub fn decode_residuals(bytes: &[u8], n: usize) -> Option<Vec<i32>> {
    let mut dec = Decoder::new(bytes);
    let mut model = Model::new();
    let res: Vec<i32> = (0..n).map(|_| model.decode(&mut dec)).collect();
    if dec.overrun() {
        None
    } else {
        Some(res)
    }
}

/// Samples `y` (shifted and mean-removed) to bytes.
pub fn encode(y: &[i32], pre: &Preset, use_ref: bool, refwin: &[f32]) -> Vec<u8> {
    let preds = predict_all(y, pre, use_ref, refwin);
    let res: Vec<i32> = y.iter().zip(&preds).map(|(a, b)| a - b).collect();
    encode_residuals(&res)
}

/// The inverse of [`encode`]; `None` if the bytes are not a valid segment.
pub fn decode(bytes: &[u8], n: usize, pre: &Preset, use_ref: bool, refwin: &[f32]) -> Option<Vec<i32>> {
    let res = decode_residuals(bytes, n)?;
    Some(reconstruct(&res, pre, use_ref, refwin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_roundtrip_with_and_without_reference() {
        let n = 5000;
        let mut x = 1u32;
        let mut noise = move || {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 24) as i32 % 5) - 2
        };
        let mut s = 99u32;
        let sig: Vec<i32> = (0..n + 16)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 20) as i32 % 201) - 100
            })
            .collect();
        let y: Vec<i32> = (0..n).map(|i| sig[i + 8] + noise()).collect();
        for level in 1..=4 {
            let pre = Preset::level(level).unwrap();
            let refwin: Vec<f32> = (0..refwin_len(n, &pre)).map(|i| (sig[i + 8 - pre.taps] + noise()) as f32).collect();
            for use_ref in [false, true] {
                let bytes = encode(&y, &pre, use_ref, &refwin);
                let back = decode(&bytes, n, &pre, use_ref, &refwin).unwrap();
                assert_eq!(back, y, "level {level} ref {use_ref}");
            }
            let with = encode(&y, &pre, true, &refwin).len();
            let without = encode(&y, &pre, false, &refwin).len();
            assert!(with < without, "the reference should help: {with} vs {without}");
        }
    }
}
