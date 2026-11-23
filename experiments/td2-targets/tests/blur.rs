// SPDX-License-Identifier: Apache-2.0
//! Hand-computed cases for the frozen TD2 blur (D2b2b2a).

use td2_targets::augment::{
    AugmentError, BLUR_DOMAIN, BLUR_PROBABILITY, BLUR_WIDE_PROBABILITY, BlurKernel, Image, blur,
    sample_blur,
};

fn gray(width: usize, height: usize, values: &[u8]) -> Image {
    Image {
        width,
        height,
        rgb: values.iter().flat_map(|&v| [v, v, v]).collect(),
    }
}

fn channel0(image: &Image) -> Vec<u8> {
    image.rgb.iter().step_by(3).copied().collect()
}

#[test]
fn frozen_constants() {
    assert_eq!(BLUR_PROBABILITY, 0.1);
    assert_eq!(BLUR_WIDE_PROBABILITY, 0.5);
    assert_eq!(BLUR_DOMAIN, 0x7464_3262_3262_6c72);
    assert_eq!(BlurKernel::Binomial3.taps(), &[1, 2, 1]);
    assert_eq!(BlurKernel::Binomial5.taps(), &[1, 4, 6, 4, 1]);
}

#[test]
fn none_and_uniform_images_are_unchanged() {
    let img = gray(3, 2, &[0, 10, 20, 30, 40, 255]);
    assert_eq!(blur(&img, None).unwrap(), img);
    let flat = gray(4, 3, &[77; 12]);
    for k in [BlurKernel::Binomial3, BlurKernel::Binomial5] {
        assert_eq!(blur(&flat, Some(k)).unwrap(), flat);
    }
}

#[test]
fn binomial3_impulse_hand_computed() {
    // 3x3, centre 255: weights 4/16, 2/16, 1/16 → 63.75, 31.875, 15.9375.
    let img = gray(3, 3, &[0, 0, 0, 0, 255, 0, 0, 0, 0]);
    let out = blur(&img, Some(BlurKernel::Binomial3)).unwrap();
    assert_eq!(channel0(&out), vec![16, 32, 16, 32, 64, 32, 16, 32, 16]);
    // All three channels are blurred alike.
    assert!(
        out.rgb
            .chunks_exact(3)
            .all(|p| p[0] == p[1] && p[1] == p[2])
    );
}

#[test]
fn border_is_replicated_and_halves_round_up() {
    // One column [0, 255, 0]: horizontal taps all hit the same pixel, the
    // vertical pass gives (0 + 2·255 + 0)/4 = 127.5 → 128 in the middle and
    // (3·0 + 255)/4 = 63.75 → 64 at the replicated ends.
    let img = gray(1, 3, &[0, 255, 0]);
    let out = blur(&img, Some(BlurKernel::Binomial3)).unwrap();
    assert_eq!(channel0(&out), vec![64, 128, 64]);
    // 5-tap on one row [255, 0, 0, 0, 0]: replicated left edge.
    // x0: (1+4+6)·255/16 = 175.3125 → 175; x1: (1+4)·255/16 = 79.6875 → 80;
    // x2: 255/16 = 15.9375 → 16; x3, x4: 0.
    let row = gray(5, 1, &[255, 0, 0, 0, 0]);
    let out = blur(&row, Some(BlurKernel::Binomial5)).unwrap();
    assert_eq!(channel0(&out), vec![175, 80, 16, 0, 0]);
}

#[test]
fn two_dimensional_vectors_cross_checked_independently() {
    // 4x3 image; 5-tap values recomputed by an independent Python convolution
    // with replicated borders and half-up rounding. Pins the 5-tap vertical axis.
    let img = gray(4, 3, &[10, 200, 30, 255, 0, 90, 128, 64, 7, 250, 3, 100]);
    let out = blur(&img, Some(BlurKernel::Binomial5)).unwrap();
    assert_eq!(
        channel0(&out),
        vec![52, 94, 126, 160, 51, 90, 108, 118, 58, 95, 97, 91]
    );
    // One column impulse, 5-tap vertical only: (6, 4, 1)·255/16 by distance.
    let col = gray(1, 5, &[0, 0, 255, 0, 0]);
    let out = blur(&col, Some(BlurKernel::Binomial5)).unwrap();
    assert_eq!(channel0(&out), vec![16, 64, 96, 64, 16]);
}

#[test]
fn rejects_overflowing_dimensions() {
    let huge = Image {
        width: usize::MAX,
        height: 2,
        rgb: vec![],
    };
    assert_eq!(
        blur(&huge, Some(BlurKernel::Binomial3)),
        Err(AugmentError::Shape)
    );
}

#[test]
fn rejects_bad_shapes() {
    let bad = Image {
        width: 2,
        height: 2,
        rgb: vec![0; 11],
    };
    assert_eq!(blur(&bad, None), Err(AugmentError::Shape));
    let empty = Image {
        width: 0,
        height: 3,
        rgb: vec![],
    };
    assert_eq!(
        blur(&empty, Some(BlurKernel::Binomial3)),
        Err(AugmentError::Shape)
    );
}

#[test]
fn sampled_vectors_match_an_independent_splitmix64() {
    // Recomputed outside Rust with a separate Python SplitMix64 on
    // seed ^ BLUR_DOMAIN (first draw on/off, second draw width).
    assert_eq!(sample_blur(0), None);
    assert_eq!(sample_blur(1), None);
    assert_eq!(sample_blur(10), Some(BlurKernel::Binomial3));
    assert_eq!(sample_blur(15), Some(BlurKernel::Binomial5));
    assert_eq!(sample_blur(63), Some(BlurKernel::Binomial5));
    assert_eq!(sample_blur(84), Some(BlurKernel::Binomial3));
    let on: Vec<u64> = (0..100).filter(|&s| sample_blur(s).is_some()).collect();
    assert_eq!(on, vec![10, 15, 24, 27, 30, 36, 49, 63, 77, 84, 90]);
}

#[test]
fn blur_draw_frequencies_match() {
    // The D2b2b1 draws are pinned by the frozen vectors in tests/augment.rs,
    // which this change leaves untouched and which still pass.
    let n = 20_000u64;
    let mut on = 0u64;
    let mut wide = 0u64;
    for seed in 0..n {
        if let Some(k) = sample_blur(seed) {
            on += 1;
            wide += u64::from(k == BlurKernel::Binomial5);
        }
    }
    let within = |count: u64, trials: u64, p: f64| {
        let mean = trials as f64 * p;
        let sd = (trials as f64 * p * (1.0 - p)).sqrt();
        (count as f64 - mean).abs() < 4.5 * sd
    };
    assert!(within(on, n, BLUR_PROBABILITY), "{on}");
    assert!(within(wide, on, BLUR_WIDE_PROBABILITY), "{wide}/{on}");
}
