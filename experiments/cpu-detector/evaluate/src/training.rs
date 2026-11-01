// SPDX-License-Identifier: Apache-2.0
use fy_tiny_model::{Rng, Sample, INPUTS};

// Reuse the 64-sample feature and label buffers across all 2,000 training steps.
// The sample draw order, payload bytes and labels remain unchanged.
pub fn fill_training_batch(
    train: &[Sample],
    cached: &[f32],
    rng: &mut Rng,
    batch: &mut Vec<f32>,
    labels: &mut Vec<Sample>,
) {
    batch.clear();
    labels.clear();
    for _ in 0..64 {
        let index = rng.next_u32() as usize % train.len();
        batch.extend_from_slice(&cached[index * INPUTS..(index + 1) * INPUTS]);
        labels.push(Sample {
            pixels: vec![],
            class: train[index].class,
            box_xywh: train[index].box_xywh,
        });
    }
}

/// Mirror only the four ground-truth edge-positive reserved slots of a TD1
/// training batch. Slot 0 stays unchanged, 16 flips horizontally, 32 vertically,
/// and 48 flips both axes. This does not draw RNG, change sample identities,
/// allocate an image buffer, or change any of the other 60 training slots.
///
/// Check all inputs first so malformed labels cannot leave a partially
/// transformed batch. This is an opt-in experiment, not deployed inference.
pub fn mirror_reserved_edge_batch(batch: &mut [f32], labels: &mut [Sample]) -> Result<(), String> {
    const SLOTS: [usize; 4] = [0, 16, 32, 48];
    if batch.len() != 64 * INPUTS || labels.len() != 64 {
        return Err("mirroring requires a complete 64-sample TD1 batch".into());
    }
    for slot in SLOTS {
        let label = &labels[slot];
        let [x, y, w, h] = label.box_xywh;
        if label.class.is_none()
            || [x, y, w, h].iter().any(|n| !n.is_finite())
            || w <= 0.0
            || h <= 0.0
            || x - w / 2.0 < 0.0
            || x + w / 2.0 > 1.0
            || y - h / 2.0 < 0.0
            || y + h / 2.0 > 1.0
            || batch[slot * INPUTS..(slot + 1) * INPUTS]
                .iter()
                .any(|n| !n.is_finite() || !(0.0..=1.0).contains(n))
        {
            return Err("invalid reserved TD1 edge sample for mirroring".into());
        }
    }

    for (orientation, slot) in SLOTS.into_iter().enumerate() {
        let horizontal = orientation & 1 != 0;
        let vertical = orientation & 2 != 0;
        if !horizontal && !vertical {
            continue;
        }
        let start = slot * INPUTS;
        let mut original = [0.0_f32; INPUTS];
        original.copy_from_slice(&batch[start..start + INPUTS]);
        // An 8x8 block contains exactly 8x8 RGB pixels; reflection can
        // permute already averaged F32 features without any rounding.
        for y in 0..8 {
            for x in 0..8 {
                let source_x = if horizontal { 7 - x } else { x };
                let source_y = if vertical { 7 - y } else { y };
                for channel in 0..3 {
                    batch[start + (y * 8 + x) * 3 + channel] =
                        original[(source_y * 8 + source_x) * 3 + channel];
                }
            }
        }
        if horizontal {
            labels[slot].box_xywh[0] = 1.0 - labels[slot].box_xywh[0];
        }
        if vertical {
            labels[slot].box_xywh[1] = 1.0 - labels[slot].box_xywh[1];
        }
    }
    Ok(())
}

/// Opt-in TD1 feature-space illumination experiment. Model inputs are the
/// established 8x8 RGB block means, so changing them here cannot change the
/// original images, the sampler/RNG stream, targets or native inference ABI.
///
/// Phase selection uses only the fixed step and batch slot, never the class,
/// geometry or split identity. A shared additive channel lift preserves the
/// red/blue dominance of each object while exposing the network to varying
/// brightness, checkerboard and horizontal/vertical backgrounds. The identity
/// phase deliberately retains a quarter of the historical features.
///
/// Validate the entire batch before writing: a corrupted last feature must
/// not partially transform the earlier slots.
pub fn illuminate_training_batch(
    batch: &mut [f32],
    labels: &[Sample],
    step: usize,
) -> Result<(), String> {
    if batch.len() != 64 * INPUTS
        || labels.len() != 64
        || !(1..=2000).contains(&step)
        || batch
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("invalid TD1 batch for feature illumination".into());
    }

    for (slot, features) in batch.as_chunks_mut::<INPUTS>().0.iter_mut().enumerate() {
        let phase = (step + slot) % 4;
        for y in 0..8 {
            for x in 0..8 {
                let lift = match phase {
                    0 => 0.0_f32,
                    1 if (x + y) % 2 == 0 => 0.15_f32,
                    1 => 0.08_f32,
                    2 => 0.07_f32 + 0.02_f32 * x as f32,
                    _ => 0.07_f32 + 0.02_f32 * y as f32,
                };
                for channel in 0..3 {
                    let index = (y * 8 + x) * 3 + channel;
                    let value = features[index];
                    features[index] = value + (1.0 - value) * lift;
                }
            }
        }
    }
    Ok(())
}

// Preserve the original operation order in both the validation and commit pass.
fn candidate(weight: f32, gradient: f32, m: f32, v: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let nm = 0.9 * m + 0.1 * gradient;
    let nv = 0.999 * v + 0.001 * gradient * gradient;
    let nw = weight - 0.003 * (nm / a) / ((nv / b).sqrt() + 1e-8);
    (nw, nm, nv)
}

pub fn adam_step(
    weights: &mut [f32],
    gradient: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: usize,
) -> Result<(), String> {
    if step == 0
        || step > 2000
        || weights.is_empty()
        || [gradient.len(), m.len(), v.len()]
            .iter()
            .any(|n| *n != weights.len())
        || weights
            .iter()
            .chain(gradient)
            .chain(m.iter())
            .chain(v.iter())
            .any(|x| !x.is_finite())
        || v.iter().any(|x| *x < 0.0)
    {
        return Err("invalid Adam state/gradient/step".into());
    }
    let a = 1.0 - 0.9_f32.powi(step as i32);
    let b = 1.0 - 0.999_f32.powi(step as i32);
    // Two passes avoid staging one (weight, first-moment, second-moment)
    // tuple per parameter. Every proposed state is checked before any mutation.
    // Each parameter update depends only on that index's previous values.
    for i in 0..weights.len() {
        let (nw, nm, nv) = candidate(weights[i], gradient[i], m[i], v[i], a, b);
        if !nw.is_finite() || nw.abs() > 10000.0 || !nm.is_finite() || !nv.is_finite() {
            return Err("nonfinite/out-of-range Adam update".into());
        }
    }
    for i in 0..weights.len() {
        let (nw, nm, nv) = candidate(weights[i], gradient[i], m[i], v[i], a, b);
        weights[i] = nw;
        m[i] = nm;
        v[i] = nv;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_pass_update_matches_historical_staging_bit_for_bit() {
        // This is an independent copy of the prior Vec-staging algorithm,
        // including its original f32 expression ordering.
        fn historical(
            weights: &mut [f32],
            gradient: &[f32],
            m: &mut [f32],
            v: &mut [f32],
            step: usize,
        ) {
            let a = 1.0 - 0.9_f32.powi(step as i32);
            let b = 1.0 - 0.999_f32.powi(step as i32);
            let next = (0..weights.len())
                .map(|i| {
                    let nm = 0.9 * m[i] + 0.1 * gradient[i];
                    let nv = 0.999 * v[i] + 0.001 * gradient[i] * gradient[i];
                    let nw = weights[i] - 0.003 * (nm / a) / ((nv / b).sqrt() + 1e-8);
                    (nw, nm, nv)
                })
                .collect::<Vec<_>>();
            for (i, (nw, nm, nv)) in next.into_iter().enumerate() {
                weights[i] = nw;
                m[i] = nm;
                v[i] = nv;
            }
        }

        let n = fy_tiny_model::PARAMETERS;
        let mut weights = (0..n)
            .map(|i| (i % 83) as f32 / 100.0 - 0.41)
            .collect::<Vec<_>>();
        let gradient = (0..n)
            .map(|i| (i % 57) as f32 / 500.0 - 0.056)
            .collect::<Vec<_>>();
        let mut m = (0..n)
            .map(|i| (i % 29) as f32 / 1000.0 - 0.014)
            .collect::<Vec<_>>();
        let mut v = (0..n)
            .map(|i| (i % 31) as f32 / 10_000.0 + 0.001)
            .collect::<Vec<_>>();
        let (mut expected_weights, mut expected_m, mut expected_v) =
            (weights.clone(), m.clone(), v.clone());
        for step in [1, 2, 17, 100, 2000] {
            historical(
                &mut expected_weights,
                &gradient,
                &mut expected_m,
                &mut expected_v,
                step,
            );
            adam_step(&mut weights, &gradient, &mut m, &mut v, step).unwrap();
            for (actual, expected) in [
                (&weights[..], &expected_weights[..]),
                (&m[..], &expected_m[..]),
                (&v[..], &expected_v[..]),
            ] {
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "Adam state changed at step {step}"
                );
            }
        }
    }

    #[test]
    fn invalid_last_parameter_keeps_entire_optimizer_state_unchanged() {
        let n = fy_tiny_model::PARAMETERS;
        let mut weights = vec![0.5; n];
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut gradient = vec![0.01; n];
        gradient[n - 1] = f32::MAX;
        let before = (weights.clone(), m.clone(), v.clone());
        assert!(adam_step(&mut weights, &gradient, &mut m, &mut v, 1).is_err());
        assert_eq!((weights, m, v), before);
    }

    #[test]
    fn adam_changes_real_parameters_with_the_pinned_update_and_rejects_nan() {
        let mut weights = [1.0, -1.0];
        let mut m = [0.0; 2];
        let mut v = [0.0; 2];
        adam_step(&mut weights, &[2.0, -4.0], &mut m, &mut v, 1).unwrap();
        assert!((weights[0] - 0.997).abs() < 1e-6);
        assert!((weights[1] + 0.997).abs() < 1e-6);
        let before = (weights, m, v);
        assert!(adam_step(&mut weights, &[f32::NAN, 0.0], &mut m, &mut v, 2).is_err());
        assert_eq!((weights, m, v), before);
        assert!(adam_step(&mut weights, &[f32::MAX, 0.0], &mut m, &mut v, 2).is_err());
        assert_eq!((weights, m, v), before);
    }

    #[test]
    fn illuminated_batch_covers_four_phases_without_changing_targets() {
        let original = vec![0.25_f32; 64 * INPUTS];
        let mut batch = original.clone();
        let labels = (0..64)
            .map(|slot| Sample {
                pixels: vec![],
                class: if slot % 5 == 0 { None } else { Some(0) },
                box_xywh: [0.5, 0.5, 0.25, 0.25],
            })
            .collect::<Vec<_>>();
        let original_labels = labels
            .iter()
            .map(|sample| (sample.class, sample.box_xywh))
            .collect::<Vec<_>>();

        illuminate_training_batch(&mut batch, &labels, 1).unwrap();
        let expected = |lift: f32| 0.25 + 0.75 * lift;
        assert!((batch[0] - expected(0.15)).abs() < 1e-6);
        assert!((batch[3] - expected(0.08)).abs() < 1e-6);
        assert!((batch[INPUTS + 7 * 3] - expected(0.21)).abs() < 1e-6);
        assert!((batch[2 * INPUTS + 7 * 8 * 3] - expected(0.21)).abs() < 1e-6);
        assert_eq!(
            &batch[3 * INPUTS..4 * INPUTS],
            &original[3 * INPUTS..4 * INPUTS]
        );
        assert!(batch
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
        assert_eq!(
            labels
                .iter()
                .map(|sample| (sample.class, sample.box_xywh))
                .collect::<Vec<_>>(),
            original_labels
        );

        let mut replay = original;
        illuminate_training_batch(&mut replay, &labels, 1).unwrap();
        assert_eq!(
            batch
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            replay
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_ne!(&batch[0..INPUTS], &batch[3 * INPUTS..4 * INPUTS]);
    }

    #[test]
    fn invalid_illumination_input_is_rejected_before_any_mutation() {
        let labels = vec![
            Sample {
                pixels: vec![],
                class: Some(1),
                box_xywh: [0.5, 0.5, 0.25, 0.25],
            };
            64
        ];
        for invalid in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            let mut batch = vec![0.25_f32; 64 * INPUTS];
            let last = batch.len() - 1;
            batch[last] = invalid;
            let before = batch.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert!(illuminate_training_batch(&mut batch, &labels, 1).is_err());
            assert_eq!(
                batch.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                before
            );
        }
        let mut batch = vec![0.25_f32; 64 * INPUTS];
        let original = batch.clone();
        for step in [0, 2001] {
            assert!(illuminate_training_batch(&mut batch, &labels, step).is_err());
            assert_eq!(batch, original);
        }
        let short_len = batch.len() - 1;
        assert!(illuminate_training_batch(&mut batch[..short_len], &labels, 1).is_err());
        assert!(illuminate_training_batch(&mut batch, &labels[..63], 1).is_err());
        assert_eq!(batch, original);
    }

    #[test]
    fn mirrored_edge_slots_match_full_resolution_pixel_reflections() {
        let sample = fy_tiny_model::dataset(101, 2).pop().unwrap();
        assert!(sample.class.is_some());
        let features = fy_tiny_model::features(&sample.pixels).unwrap();
        let mut input = Vec::with_capacity(64 * INPUTS);
        let mut labels = Vec::with_capacity(64);
        for _ in 0..64 {
            input.extend_from_slice(&features);
            labels.push(Sample {
                pixels: vec![],
                class: sample.class,
                box_xywh: sample.box_xywh,
            });
        }
        mirror_reserved_edge_batch(&mut input, &mut labels).unwrap();

        for slot in 0..64 {
            let (horizontal, vertical) = match slot {
                16 => (true, false),
                32 => (false, true),
                48 => (true, true),
                _ => (false, false),
            };
            let mut pixels = vec![0_u8; 64 * 64 * 3];
            for y in 0..64 {
                for x in 0..64 {
                    let source_x = if horizontal { 63 - x } else { x };
                    let source_y = if vertical { 63 - y } else { y };
                    let src = (source_y * 64 + source_x) * 3;
                    let dst = (y * 64 + x) * 3;
                    pixels[dst..dst + 3].copy_from_slice(&sample.pixels[src..src + 3]);
                }
            }
            let expected = fy_tiny_model::features(&pixels).unwrap();
            assert_eq!(
                input[slot * INPUTS..(slot + 1) * INPUTS]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "reflected features differ in slot {slot}"
            );
            let mut expected_box = sample.box_xywh;
            if horizontal {
                expected_box[0] = 1.0 - expected_box[0];
            }
            if vertical {
                expected_box[1] = 1.0 - expected_box[1];
            }
            assert_eq!(
                labels[slot].box_xywh.map(f32::to_bits),
                expected_box.map(f32::to_bits),
                "reflected target differs in slot {slot}"
            );
            assert_eq!(labels[slot].class, sample.class);
            assert!(labels[slot].pixels.is_empty());
        }
    }

    #[test]
    fn invalid_reserved_sample_does_not_partially_mirror_batch() {
        let mut input = vec![0.25_f32; 64 * INPUTS];
        let mut labels = vec![
            Sample {
                pixels: vec![],
                class: Some(0),
                box_xywh: [0.5, 0.5, 0.25, 0.25],
            };
            64
        ];
        labels[48].class = None;
        let original_input = input.clone();
        let original_boxes = labels.iter().map(|s| s.box_xywh).collect::<Vec<_>>();
        assert!(mirror_reserved_edge_batch(&mut input, &mut labels).is_err());
        assert_eq!(input, original_input);
        assert_eq!(
            labels.iter().map(|s| s.box_xywh).collect::<Vec<_>>(),
            original_boxes
        );
    }
}
