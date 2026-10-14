// SPDX-License-Identifier: Apache-2.0
use fy_tiny_model::Sample;
#[derive(Debug, serde::Serialize)]
pub struct Metrics {
    pub positives: usize,
    pub negatives: usize,
    pub class_accuracy: f32,
    pub recall_iou50: f32,
    pub mean_iou: f32,
    pub false_positive_rate: f32,
}
pub fn target_corners(sample: &Sample) -> [f32; 4] {
    let [x, y, w, h] = sample.box_xywh;
    [
        (x - w / 2.0) * 64.0,
        (y - h / 2.0) * 64.0,
        (x + w / 2.0) * 64.0,
        (y + h / 2.0) * 64.0,
    ]
}
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let union = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)
        + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
        - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
pub fn measure(outputs: &[f32], samples: &[Sample]) -> Result<Metrics, String> {
    // Share validation with the objective; no NaN/invalid target can pass a metric gate.
    fy_tiny_model::loss_gradient(outputs, samples)?;
    let mut m = Metrics {
        positives: 0,
        negatives: 0,
        class_accuracy: 0.0,
        recall_iou50: 0.0,
        mean_iou: 0.0,
        false_positive_rate: 0.0,
    };
    for (out, sample) in outputs.as_chunks::<7>().0.iter().zip(samples) {
        let detection = fy_tiny_model::decode(out)?;
        if let Some(class) = sample.class {
            m.positives += 1;
            if let Some(d) = detection {
                let overlap = iou(d.corners, target_corners(sample));
                m.mean_iou += overlap;
                if d.class == class {
                    m.class_accuracy += 1.0;
                    if overlap >= 0.5 {
                        m.recall_iou50 += 1.0;
                    }
                }
            }
        } else {
            m.negatives += 1;
            if detection.is_some() {
                m.false_positive_rate += 1.0;
            }
        }
    }
    if m.positives == 0 || m.negatives == 0 {
        return Err("metrics require positive and background samples".into());
    }
    m.class_accuracy /= m.positives as f32;
    m.recall_iou50 /= m.positives as f32;
    m.mean_iou /= m.positives as f32;
    m.false_positive_rate /= m.negatives as f32;
    Ok(m)
}
impl Metrics {
    pub fn passes(&self) -> bool {
        self.class_accuracy >= 0.95
            && self.recall_iou50 >= 0.85
            && self.mean_iou >= 0.70
            && self.false_positive_rate <= 0.05
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_penalize_absent_boxes_and_background_false_positives() {
        let samples = vec![
            Sample {
                pixels: vec![],
                class: Some(0),
                box_xywh: [0.5; 4],
            },
            Sample {
                pixels: vec![],
                class: None,
                box_xywh: [0.0; 4],
            },
        ];
        let perfect = [
            0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let m = measure(&perfect, &samples).unwrap();
        assert_eq!(m.mean_iou, 1.0);
        assert_eq!(m.recall_iou50, 1.0);
        assert_eq!(m.class_accuracy, 1.0);
        assert_eq!(m.false_positive_rate, 0.0);
        let swapped = [
            10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let m = measure(&swapped, &samples).unwrap();
        assert_eq!(m.mean_iou, 0.0);
        assert_eq!(m.recall_iou50, 0.0);
        assert_eq!(m.class_accuracy, 0.0);
        assert_eq!(m.false_positive_rate, 1.0);
        assert!(measure(&[f32::NAN; 14], &samples).is_err());
        assert!(measure(&perfect[..13], &samples).is_err());
    }
}
