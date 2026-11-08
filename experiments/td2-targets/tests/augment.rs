// SPDX-License-Identifier: Apache-2.0
//! Hand-computed cases for the frozen TD2 flip and photometric augmentation
//! (D2b2b1).

use td2_targets::GtBox;
use td2_targets::augment::{
    AugmentError, BRIGHTNESS, CONTRAST, FLIP_PROBABILITY, GRAYSCALE_PROBABILITY, Image,
    LOW_LIGHT_GAIN, LOW_LIGHT_PROBABILITY, LUMA, Params, SATURATION, apply, sample,
};

fn image(width: usize, height: usize, rgb: &[u8]) -> Image {
    Image {
        width,
        height,
        rgb: rgb.to_vec(),
    }
}

fn gt(left: f64, top: f64, right: f64, bottom: f64) -> GtBox {
    GtBox {
        left,
        top,
        right,
        bottom,
        class: 3,
    }
}

/// Deterministic original test pattern (no dataset pixels).
fn pattern(width: usize, height: usize, seed: u32) -> Image {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for i in 0..width * height {
        let v = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
        rgb.extend_from_slice(&[(v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8]);
    }
    image(width, height, &rgb)
}

#[test]
fn frozen_constants() {
    assert_eq!(FLIP_PROBABILITY, 0.5);
    assert_eq!(LOW_LIGHT_PROBABILITY, 0.2);
    assert_eq!(LOW_LIGHT_GAIN, (0.15, 0.5));
    assert_eq!(BRIGHTNESS, (0.75, 1.25));
    assert_eq!(CONTRAST, (0.75, 1.25));
    assert_eq!(SATURATION, (0.6, 1.4));
    assert_eq!(GRAYSCALE_PROBABILITY, 0.1);
    assert_eq!(LUMA, [0.299, 0.587, 0.114]);
}

#[test]
fn identity_leaves_pixels_and_boxes_unchanged() {
    let img = pattern(7, 5, 1);
    let boxes = [gt(0.0, 0.0, 7.0, 5.0), gt(1.5, 2.0, 3.25, 4.0)];
    let (out, out_boxes) = apply(&img, &boxes, &Params::IDENTITY).unwrap();
    assert_eq!(out, img);
    assert_eq!(out_boxes, boxes);
}

#[test]
fn flip_mirrors_rows_and_boxes() {
    // Two rows of three pixels.
    let rgb: Vec<u8> = (0..18).collect();
    let img = image(3, 2, &rgb);
    let p = Params {
        flip: true,
        ..Params::IDENTITY
    };
    let (out, boxes) = apply(&img, &[gt(0.5, 0.0, 2.0, 1.0)], &p).unwrap();
    assert_eq!(
        out.rgb,
        [6, 7, 8, 3, 4, 5, 0, 1, 2, 15, 16, 17, 12, 13, 14, 9, 10, 11]
    );
    assert_eq!(boxes, [gt(1.0, 0.0, 2.5, 1.0)]);
    // Flipping twice restores the input exactly.
    let (back, back_boxes) = apply(&out, &boxes, &p).unwrap();
    assert_eq!(back, img);
    assert_eq!(back_boxes, [gt(0.5, 0.0, 2.0, 1.0)]);
}

#[test]
fn gain_and_brightness_scale_then_clamp_and_round() {
    let img = image(2, 1, &[100, 200, 3, 255, 0, 51]);
    let p = Params {
        low_light_gain: Some(0.25),
        brightness: 1.2,
        ..Params::IDENTITY
    };
    // 100·0.25·1.2 = 30; 200 → 60; 3 → 0.9 → 1; 255 → 76.5 → 77 (half away
    // from zero); 51 → 15.3 → 15.
    let (out, _) = apply(&img, &[], &p).unwrap();
    assert_eq!(out.rgb, [30, 60, 1, 77, 0, 15]);
    // Brightness alone clamps at 255.
    let bright = Params {
        brightness: 1.25,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&image(1, 1, &[250, 204, 0]), &[], &bright).unwrap();
    assert_eq!(out.rgb, [255, 255, 0]);
}

#[test]
fn contrast_pivots_on_mean_luma() {
    // Gray pixels 50 and 150: mean luma 100. Factor 0.8 → 60 and 140.
    let img = image(2, 1, &[50, 50, 50, 150, 150, 150]);
    let p = Params {
        contrast: 0.8,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&img, &[], &p).unwrap();
    assert_eq!(out.rgb, [60, 60, 60, 140, 140, 140]);
    // Factor 1.25 → 37.5 → 38 and 162.5 → 163.
    let p = Params {
        contrast: 1.25,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&img, &[], &p).unwrap();
    assert_eq!(out.rgb, [38, 38, 38, 163, 163, 163]);
}

#[test]
fn saturation_and_grayscale_blend_towards_luma() {
    // Pure red: luma 0.299·200 = 59.8.
    let img = image(1, 1, &[200, 0, 0]);
    let gray = Params {
        grayscale: true,
        saturation: 1.3,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&img, &[], &gray).unwrap();
    assert_eq!(out.rgb, [60, 60, 60]);
    // Saturation 0.6: 59.8 + 140.2·0.6 = 143.92 → 144; 59.8·0.4 = 23.92 → 24.
    let p = Params {
        saturation: 0.6,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&img, &[], &p).unwrap();
    assert_eq!(out.rgb, [144, 24, 24]);
    // Saturation 1.4 overshoots and clamps: 59.8 + 140.2·1.4 = 256.08 → 255;
    // 59.8 − 59.8·1.4 < 0 → 0.
    let p = Params {
        saturation: 1.4,
        ..Params::IDENTITY
    };
    let (out, _) = apply(&img, &[], &p).unwrap();
    assert_eq!(out.rgb, [255, 0, 0]);
}

#[test]
fn photometric_operations_never_move_boxes() {
    let img = pattern(16, 9, 7);
    let boxes = [gt(1.0, 2.0, 15.0, 9.0)];
    for seed in 0..200 {
        let p = Params {
            flip: false,
            ..sample(seed)
        };
        let (out, out_boxes) = apply(&img, &boxes, &p).unwrap();
        assert_eq!(out_boxes, boxes);
        assert_eq!((out.width, out.height, out.rgb.len()), (16, 9, 16 * 9 * 3));
    }
}

#[test]
fn sampling_is_deterministic_and_frequencies_match() {
    assert_eq!(sample(42), sample(42));
    assert_ne!(sample(42), sample(43));
    let n = 20_000;
    let (mut flips, mut lows, mut grays) = (0, 0, 0);
    for seed in 0..n {
        let p = sample(seed);
        flips += usize::from(p.flip);
        grays += usize::from(p.grayscale);
        if let Some(g) = p.low_light_gain {
            lows += 1;
            assert!((LOW_LIGHT_GAIN.0..=LOW_LIGHT_GAIN.1).contains(&g));
        }
        for (v, (lo, hi)) in [
            (p.brightness, BRIGHTNESS),
            (p.contrast, CONTRAST),
            (p.saturation, SATURATION),
        ] {
            assert!(
                (lo..=hi).contains(&v),
                "seed {seed}: {v} not in [{lo}, {hi}]"
            );
        }
        // Every sampled augmentation is accepted by `apply`.
        apply(&image(1, 1, &[1, 2, 3]), &[], &p).unwrap();
    }
    // 4.5 standard deviations of a binomial count.
    let near = |count: usize, prob: f64| {
        let mean = n as f64 * prob;
        (count as f64 - mean).abs() <= 4.5 * (mean * (1.0 - prob)).sqrt()
    };
    assert!(near(flips, FLIP_PROBABILITY), "flips {flips}");
    assert!(near(lows, LOW_LIGHT_PROBABILITY), "low-light {lows}");
    assert!(near(grays, GRAYSCALE_PROBABILITY), "grayscale {grays}");
}

#[test]
fn sample_draw_order_is_frozen() {
    // Pins the SplitMix64 stream and the draw order, so a reordered or
    // skipped draw changes these parameters.
    let p = sample(0);
    let q = sample(14);
    let fingerprint = |p: &Params| {
        (
            p.flip,
            p.low_light_gain.map(f64::to_bits),
            p.brightness.to_bits(),
            p.contrast.to_bits(),
            p.saturation.to_bits(),
            p.grayscale,
        )
    };
    assert_eq!(fingerprint(&p), FROZEN_SEED_0);
    assert_eq!(fingerprint(&q), FROZEN_SEED_14);
}

type Fingerprint = (bool, Option<u64>, u64, u64, u64, bool);
const FROZEN_SEED_0: Fingerprint = (
    false,
    None,
    4608242750750691940,
    4605409561906846343,
    4605938169801739433,
    false,
);
const FROZEN_SEED_14: Fingerprint = (
    true,
    Some(4594760639552223007),
    4607543129586450986,
    4608083554919126025,
    4605965500638201736,
    true,
);

#[test]
fn rejects_bad_shapes_params_and_boxes() {
    let img = image(2, 2, &[0; 12]);
    let id = Params::IDENTITY;
    assert_eq!(apply(&image(0, 2, &[]), &[], &id), Err(AugmentError::Shape));
    assert_eq!(
        apply(&image(2, 2, &[0; 11]), &[], &id),
        Err(AugmentError::Shape)
    );
    assert_eq!(
        apply(&image(usize::MAX, 2, &[]), &[], &id),
        Err(AugmentError::Shape)
    );
    for p in [
        Params {
            brightness: f64::NAN,
            ..id
        },
        Params {
            brightness: 1.3,
            ..id
        },
        Params {
            contrast: 0.5,
            ..id
        },
        Params {
            saturation: 1.5,
            ..id
        },
        Params {
            low_light_gain: Some(0.6),
            ..id
        },
        Params {
            low_light_gain: Some(f64::INFINITY),
            ..id
        },
    ] {
        assert_eq!(apply(&img, &[], &p), Err(AugmentError::Param), "{p:?}");
    }
    for (index, b) in [
        gt(0.0, 0.0, 2.5, 1.0),
        gt(-0.1, 0.0, 1.0, 1.0),
        gt(1.0, 0.0, 1.0, 1.0),
        gt(0.0, 1.5, 1.0, 1.0),
        gt(0.0, 0.0, f64::NAN, 1.0),
        gt(0.0, 0.0, 1.0, 2.0001),
    ]
    .into_iter()
    .enumerate()
    {
        let boxes = [gt(0.0, 0.0, 1.0, 1.0), b];
        assert_eq!(
            apply(&img, &boxes, &id),
            Err(AugmentError::Box { index: 1 }),
            "case {index}"
        );
    }
}
