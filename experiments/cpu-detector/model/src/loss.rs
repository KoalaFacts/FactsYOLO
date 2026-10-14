// SPDX-License-Identifier: Apache-2.0
use crate::{sigmoid, Sample};
pub fn loss_gradient(logits: &[f32], labels: &[Sample]) -> Result<(f32, Vec<f32>), String> {
    if labels.is_empty()
        || labels.len().checked_mul(7) != Some(logits.len())
        || logits.iter().any(|v| !v.is_finite())
    {
        return Err("invalid loss input".into());
    }
    let mut loss = 0.0;
    let mut gradient = vec![0.0; logits.len()];
    let scale = 1.0 / labels.len() as f32;
    for ((out, g), label) in logits
        .chunks_exact(7)
        .zip(gradient.chunks_exact_mut(7))
        .zip(labels)
    {
        let target = match label.class {
            None => 0,
            Some(0) => 1,
            Some(1) => 2,
            _ => return Err("invalid class".into()),
        };
        let max = out[..3].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = out[..3].iter().map(|x| (x - max).exp()).sum::<f32>();
        loss += (max - out[target] + sum.ln()) * scale;
        for i in 0..3 {
            g[i] = ((out[i] - max).exp() / sum - if i == target { 1.0 } else { 0.0 }) * scale;
        }
        if target != 0 {
            for i in 0..4 {
                if !label.box_xywh[i].is_finite() || !(0.0..=1.0).contains(&label.box_xywh[i]) {
                    return Err("invalid target box".into());
                }
                let p = sigmoid(out[3 + i]);
                let e = p - label.box_xywh[i];
                loss += 5.0 * e * e * 0.25 * scale;
                g[3 + i] = 2.5 * e * p * (1.0 - p) * scale;
            }
        }
    }
    if !loss.is_finite() {
        return Err("nonfinite loss".into());
    }
    Ok((loss, gradient))
}
