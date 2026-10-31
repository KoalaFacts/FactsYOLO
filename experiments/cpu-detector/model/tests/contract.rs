// SPDX-License-Identifier: Apache-2.0
use fy_tiny_model::*;
use sha2::{Digest, Sha256};

#[test]
fn preprocessing_preserves_spatial_and_channel_order() {
    let mut pixels = vec![0; 64 * 64 * 3];
    for y in 0..8 {
        for x in 8..16 {
            pixels[(y * 64 + x) * 3 + 2] = 255;
        }
    }
    let f = features(&pixels).unwrap();
    assert_eq!(f.len(), 192);
    assert_eq!(f[5], 1.0);
    assert_eq!(f.iter().sum::<f32>(), 1.0);
    assert!(features(&pixels[..pixels.len() - 1]).is_err());
}

#[test]
fn original_data_is_reproducible_disjoint_and_geometrically_labeled() {
    let a = dataset(101, 4096);
    let again = dataset(101, 4096);
    let b = dataset(202, 128);
    let c = dataset(909, 128);
    assert_eq!(a.len(), 4096);
    assert_eq!(b.len(), 128);
    assert_eq!(c.len(), 128);
    let mut hashes = std::collections::HashSet::new();
    for (i, s) in a.iter().chain(&b).chain(&c).enumerate() {
        assert_eq!(s.pixels.len(), 64 * 64 * 3);
        assert!(
            hashes.insert(Sha256::digest(&s.pixels).to_vec()),
            "duplicate image {i}"
        );
        if let Some(class) = s.class {
            assert!(class < 2);
            let [cx, cy, w, h] = s.box_xywh;
            let x = ((cx - w / 2.0) * 64.0).round() as usize;
            let y = ((cy - h / 2.0) * 64.0).round() as usize;
            assert!((2..=41).contains(&x));
            assert!((2..=41).contains(&y));
            assert!((12.0..=22.0).contains(&(w * 64.0)));
            assert!((12.0..=22.0).contains(&(h * 64.0)));
            assert!(s.pixels[(y * 64 + x) * 3 + class * 2] > 150);
            assert!(s.pixels[0] < 20);
        }
    }
    for (a, b) in a.iter().zip(again) {
        assert_eq!(a.pixels, b.pixels);
        assert_eq!(a.class, b.class);
        assert_eq!(a.box_xywh, b.box_xywh);
    }
    assert!(a.iter().any(|s| s.class.is_none()));
}

#[test]
fn real_parameters_control_class_box_and_background() {
    let mut m = Model {
        weights: vec![0.0; PARAMETERS],
    };
    m.weights[PARAMETERS - 6] = 8.0;
    let d = m.detect(&vec![0; 64 * 64 * 3]).unwrap().unwrap();
    assert_eq!(d.class, 0);
    assert!(d.score > 0.99);
    assert_eq!(d.corners, [16.0, 16.0, 48.0, 48.0]);
    m.weights[PARAMETERS - 7] = 16.0;
    assert!(m.detect(&vec![0; 64 * 64 * 3]).unwrap().is_none());
    assert!(m.forward(&[0.0; 191]).is_err());
    assert!(m.detect(&[]).is_err());
}

#[test]
fn loss_gradient_matches_finite_differences_and_masks_background_boxes() {
    let positive = Sample {
        pixels: vec![],
        class: Some(1),
        box_xywh: [0.3, 0.7, 0.2, 0.4],
    };
    let logits = vec![0.2, -0.3, 0.4, -0.2, 0.7, -0.6, 0.1];
    let (loss, g) = loss_gradient(&logits, std::slice::from_ref(&positive)).unwrap();
    assert!(loss > 0.0 && loss.is_finite());
    assert_eq!(g.len(), 7);
    for i in 0..7 {
        let mut p = logits.clone();
        let mut n = logits.clone();
        p[i] += 1e-3;
        n[i] -= 1e-3;
        let fd = (loss_gradient(&p, std::slice::from_ref(&positive))
            .unwrap()
            .0
            - loss_gradient(&n, std::slice::from_ref(&positive))
                .unwrap()
                .0)
            / 0.002;
        assert!((fd - g[i]).abs() < 0.002, "output {i}: {fd} vs {}", g[i]);
    }
    let negative = Sample {
        class: None,
        ..positive
    };
    let (_, g) = loss_gradient(&logits, &[negative]).unwrap();
    assert_eq!(&g[3..], &[0.0; 4]);
    assert!(loss_gradient(&[f32::NAN; 7], &[]).is_err());
}

#[test]
fn background_boxes_cannot_hide_invalid_targets() {
    let background = Sample {
        pixels: vec![],
        class: None,
        box_xywh: [0.0; 4],
    };
    let logits = [2.0, 0.0, -1.0, 0.1, -0.2, 0.3, -0.4];
    let (reference_loss, reference_gradient) =
        loss_gradient(&logits, std::slice::from_ref(&background)).unwrap();
    assert_eq!(&reference_gradient[3..], &[0.0; 4]);

    // Background coordinates are ignored by the box objective, not by input validation.
    let valid = Sample {
        box_xywh: [0.5; 4],
        ..background.clone()
    };
    let (valid_loss, valid_gradient) =
        loss_gradient(&logits, std::slice::from_ref(&valid)).unwrap();
    assert_eq!(reference_loss.to_bits(), valid_loss.to_bits());
    assert_eq!(reference_gradient, valid_gradient);

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        for axis in 0..4 {
            let mut bad = background.clone();
            bad.box_xywh[axis] = invalid;
            assert!(
                loss_gradient(&logits, std::slice::from_ref(&bad)).is_err(),
                "invalid background coordinate {axis} unexpectedly accepted: {invalid:?}"
            );
        }
    }
}

#[test]
fn positive_boxes_require_nonempty_in_bounds_geometry() {
    // Invalid positive labels otherwise train a target that decode() must
    // discard (zero area) or clip (outside image), corrupting the objective.
    let logits = [0.0_f32; 7];
    let base = Sample {
        pixels: vec![],
        class: Some(0),
        box_xywh: [0.5, 0.5, 1.0, 1.0],
    };
    for valid in [[0.5, 0.5, 1.0, 1.0], [0.125, 0.125, 0.25, 0.25]] {
        let sample = Sample {
            box_xywh: valid,
            ..base.clone()
        };
        assert!(loss_value(&logits, std::slice::from_ref(&sample)).is_ok());
        assert!(loss_gradient(&logits, std::slice::from_ref(&sample)).is_ok());
    }
    for invalid in [
        [0.5, 0.5, 0.0, 0.5],
        [0.5, 0.5, 0.5, 0.0],
        [0.1, 0.5, 0.5, 0.5],
        [0.9, 0.5, 0.5, 0.5],
        [0.5, 0.1, 0.5, 0.5],
        [0.5, 0.9, 0.5, 0.5],
        // Positive but subpixel widths/heights can collapse to exactly zero
        // when TD1 computes F32 corners, despite passing w > 0 / h > 0.
        [0.5, 0.5, 1e-9, 0.25],
        [0.5, 0.5, 0.25, 1e-9],
    ] {
        let sample = Sample {
            box_xywh: invalid,
            ..base.clone()
        };
        assert!(
            loss_value(&logits, std::slice::from_ref(&sample)).is_err(),
            "scalar loss accepted invalid positive box {invalid:?}"
        );
        assert!(
            loss_gradient(&logits, std::slice::from_ref(&sample)).is_err(),
            "gradient loss accepted invalid positive box {invalid:?}"
        );
        // Background geometry is intentionally not a regression target;
        // its normalized (but degenerate) metadata remains acceptable.
        let background = Sample {
            class: None,
            ..sample
        };
        assert!(loss_value(&logits, std::slice::from_ref(&background)).is_ok());
        assert!(loss_gradient(&logits, std::slice::from_ref(&background)).is_ok());
    }
}

#[test]
fn scalar_loss_matches_gradient_loss_and_rejects_corrupt_inputs() {
    let samples = [
        Sample {
            pixels: vec![],
            class: Some(1),
            box_xywh: [0.3, 0.7, 0.2, 0.4],
        },
        Sample {
            pixels: vec![],
            class: None,
            box_xywh: [0.0; 4],
        },
    ];
    let logits = [
        0.2, -0.3, 0.4, -0.2, 0.7, -0.6, 0.1, 2.0, 0.0, -1.0, 0.1, -0.2, 0.3, -0.4,
    ];
    let expected = loss_gradient(&logits, &samples).unwrap().0;
    assert_eq!(
        loss_value(&logits, &samples).unwrap().to_bits(),
        expected.to_bits(),
        "scalar validation must use exactly the training loss"
    );

    let background_only = [samples[1].clone()];
    let background_logits = &logits[7..];
    assert_eq!(
        loss_value(background_logits, &background_only)
            .unwrap()
            .to_bits(),
        loss_gradient(background_logits, &background_only)
            .unwrap()
            .0
            .to_bits()
    );

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        for axis in 0..4 {
            let mut corrupted = samples[1].clone();
            corrupted.box_xywh[axis] = invalid;
            assert!(loss_value(background_logits, &[corrupted]).is_err());
        }
    }
    assert!(loss_value(&logits[..13], &samples).is_err());
    assert!(loss_value(&[f32::NAN; 14], &samples).is_err());
    assert!(loss_value(&[], &[]).is_err());
}

fn rehash(bytes: &mut [u8]) {
    let digest = Sha256::new()
        .chain_update(&bytes[..24])
        .chain_update(&bytes[56..])
        .finalize();
    bytes[24..56].copy_from_slice(&digest);
}
#[test]
fn artifact_roundtrip_is_exact_and_rejects_corruption_and_invalid_values() {
    let model = Model::initialized(404);
    assert_eq!(model.weights.len(), PARAMETERS);
    assert_ne!(model.weights, Model::initialized(405).weights);
    let bytes = model.to_bytes().unwrap();
    assert_eq!(bytes.len(), 56 + PARAMETERS * 4);
    let loaded = Model::from_bytes(&bytes).unwrap();
    assert!(model
        .weights
        .iter()
        .zip(&loaded.weights)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(
        model.forward(&[0.25; 192]).unwrap(),
        loaded.forward(&[0.25; 192]).unwrap()
    );
    for len in [0, 7, 23, 55, bytes.len() - 1] {
        assert!(Model::from_bytes(&bytes[..len]).is_err());
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(Model::from_bytes(&extra).is_err());
    for index in [0, 8, 12, 16, 24, 56] {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert!(Model::from_bytes(&bad).is_err(), "index{index}");
    }
    for value in [f32::NAN, f32::INFINITY, 10001.0] {
        let mut bad = bytes.clone();
        bad[56..60].copy_from_slice(&value.to_le_bytes());
        rehash(&mut bad);
        assert!(Model::from_bytes(&bad).is_err());
    }
    assert!(Model { weights: vec![] }.to_bytes().is_err());
    assert!(Model::from_bytes(&vec![0; 65537]).is_err());
}
