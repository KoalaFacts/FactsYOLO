// SPDX-License-Identifier: Apache-2.0
use crate::{sigmoid, Sample};

// Evaluation needs identical scalar validation, but not an output gradient
// buffer. Keep its objective arithmetic in the same worker as training.
pub fn loss_value(logits: &[f32], labels: &[Sample]) -> Result<f32, String> {
    calculate_loss(logits, labels, false).map(|(loss, _)| loss)
}

pub fn loss_gradient(logits: &[f32], labels: &[Sample]) -> Result<(f32, Vec<f32>), String> {
    calculate_loss(logits, labels, true)
}

fn calculate_loss(
    logits: &[f32],
    labels: &[Sample],
    compute_gradient: bool,
) -> Result<(f32, Vec<f32>), String> {
    if labels.is_empty()
        || labels.len().checked_mul(7) != Some(logits.len())
        || logits.iter().any(|v| !v.is_finite())
    {
        return Err("invalid loss input".into());
    }
    let mut loss = 0.0;
    let mut gradient = if compute_gradient {
        vec![0.0; logits.len()]
    } else {
        Vec::new()
    };
    let scale = 1.0 / labels.len() as f32;
    for (index, (out, label)) in logits.chunks_exact(7).zip(labels).enumerate() {
        let mut g = if compute_gradient {
            Some(&mut gradient[index * 7..(index + 1) * 7])
        } else {
            None
        };
        let target = match label.class {
            None => 0,
            Some(0) => 1,
            Some(1) => 2,
            _ => return Err("invalid class".into()),
        };
        let max = out[..3].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = out[..3].iter().map(|x| (x - max).exp()).sum::<f32>();
        loss += (max - out[target] + sum.ln()) * scale;
        if let Some(g) = g.as_deref_mut() {
            for i in 0..3 {
                g[i] = ((out[i] - max).exp() / sum - if i == target { 1.0 } else { 0.0 }) * scale;
            }
        }
        // The box objective masks background regression only; all box
        // metadata remains subject to exactly the same validation.
        if label
            .box_xywh
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("invalid target box".into());
        }
        if target != 0 {
            // TD1 positive targets describe a nonempty box wholly inside the
            // 64x64 image. Zero-area and out-of-frame boxes can never be
            // faithfully returned by decode(), which drops or clips them.
            let [cx, cy, w, h] = label.box_xywh;
            if w <= 0.0
                || h <= 0.0
                || cx - w / 2.0 < 0.0
                || cx + w / 2.0 > 1.0
                || cy - h / 2.0 < 0.0
                || cy + h / 2.0 > 1.0
            {
                return Err("invalid positive target box geometry".into());
            }
            for i in 0..4 {
                let p = sigmoid(out[3 + i]);
                let e = p - label.box_xywh[i];
                loss += 5.0 * e * e * 0.25 * scale;
                if let Some(g) = g.as_deref_mut() {
                    g[3 + i] = 2.5 * e * p * (1.0 - p) * scale;
                }
            }
        }
    }
    if !loss.is_finite() {
        return Err("nonfinite loss".into());
    }
    Ok((loss, gradient))
}
