// SPDX-License-Identifier: Apache-2.0
//! Tests for rectified plate crops, the recognizer's input.

use td2_plates::{RenderError, Sample, Scene, crop::rectify, render};

fn scene(max_perspective: f64) -> Scene {
    Scene {
        width: 192,
        height: 96,
        max_perspective,
        max_blur: 0,
        noise_amplitude: 0,
        min_contrast: 0.4,
    }
}

fn distance2(a: [u8; 3], b: [u8; 3]) -> i32 {
    (0..3)
        .map(|i| {
            let d = i32::from(a[i]) - i32::from(b[i]);
            d * d
        })
        .sum()
}

/// Pixels nearer the text colour than halfway to the plate colour.
fn ink_mask(rgb: &[u8], sample: &Sample) -> Vec<bool> {
    let (ink, paper) = (sample.text_rgb, sample.plate_rgb);
    rgb.chunks_exact(3)
        .map(|p| distance2([p[0], p[1], p[2]], ink) * 4 < distance2(ink, paper))
        .collect()
}

#[test]
fn flat_crop_matches_the_plate_pixels() {
    let sample = render(&scene(0.0), "KM 42", 3).expect("renders");
    let [left, top] = sample.plate_quad[0];
    let [right, bottom] = sample.plate_quad[2];
    let (width, height) = (
        (right - left).round() as usize,
        (bottom - top).round() as usize,
    );
    let crop = rectify(&sample, width, height).expect("crops");
    assert_eq!(crop.rgb.len(), width * height * 3);
    // Interior pixels, away from the plate edge, follow the frame closely
    // (1.72 levels on average here; a half-pixel shift gives 2.94).
    let mut total = 0_u64;
    let mut count = 0_u64;
    for y in 2..height - 2 {
        for x in 2..width - 2 {
            let fx = (left + x as f64 + 0.5).floor() as usize;
            let fy = (top + y as f64 + 0.5).floor() as usize;
            for c in 0..3 {
                let a = i64::from(crop.rgb[(y * width + x) * 3 + c]);
                let b = i64::from(sample.rgb[(fy * sample.width + fx) * 3 + c]);
                total += a.abs_diff(b);
                count += 1;
            }
        }
    }
    let mean = total as f64 / count as f64;
    assert!(mean < 2.4, "mean absolute difference {mean}");
}

#[test]
fn crop_labels_are_level_and_hold_the_ink() {
    let (width, height) = (160, 48);
    for seed in 0..24 {
        let sample = render(&scene(0.25), "B7-XQ 05", seed).expect("renders");
        let crop = rectify(&sample, width, height).expect("crops");
        assert_eq!(crop.chars.len(), sample.chars.len());
        let ink = ink_mask(&crop.rgb, &sample);
        let mut previous_right = 0.0;
        for bbox in &crop.chars {
            let [left, top, right, bottom] = *bbox;
            // Cells sit on the same text line after rectification.
            assert!(
                (top - 0.3 / 1.6 * height as f64).abs() < 1e-6,
                "seed {seed}: top {top}"
            );
            assert!(
                (bottom - 1.3 / 1.6 * height as f64).abs() < 1e-6,
                "seed {seed}: bottom {bottom}"
            );
            assert!(left >= previous_right, "seed {seed}: cells overlap");
            previous_right = right;
            let mut inked = 0;
            for y in top as usize..bottom.ceil() as usize {
                for x in left as usize..(right.ceil() as usize).min(width) {
                    inked += usize::from(ink[y * width + x]);
                }
            }
            assert!(inked >= 6, "seed {seed}: {inked} ink pixels in {bbox:?}");
        }
        // Ink inside the crop, away from its border, lies in a cell.
        for y in 2..height - 2 {
            for x in 2..width - 2 {
                if !ink[y * width + x] {
                    continue;
                }
                let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                let covered = crop.chars.iter().any(|b| {
                    cx >= b[0] - 1.0 && cx <= b[2] + 1.0 && cy >= b[1] - 1.0 && cy <= b[3] + 1.0
                });
                assert!(covered, "seed {seed}: ink at ({x}, {y}) outside every cell");
            }
        }
    }
}

#[test]
fn skewed_and_flat_crops_agree() {
    let (width, height) = (128, 40);
    let flat = render(&scene(0.0), "AB 12", 1).expect("renders");
    let flat_ink = ink_mask(&rectify(&flat, width, height).expect("crops").rgb, &flat);
    for seed in 100..116 {
        let skewed = render(&scene(0.3), "AB 12", seed).expect("renders");
        let ink = ink_mask(
            &rectify(&skewed, width, height).expect("crops").rgb,
            &skewed,
        );
        let both = ink
            .iter()
            .zip(&flat_ink)
            .filter(|(a, b)| **a && **b)
            .count();
        let either = ink
            .iter()
            .zip(&flat_ink)
            .filter(|(a, b)| **a || **b)
            .count();
        let iou = both as f64 / either as f64;
        assert!(iou > 0.6, "seed {seed}: ink IoU {iou}");
    }
}

#[test]
fn crops_are_deterministic_and_sizes_are_checked() {
    let sample = render(&scene(0.2), "ZZ9", 9).expect("renders");
    assert_eq!(rectify(&sample, 64, 24), rectify(&sample, 64, 24));
    for (width, height) in [(0, 24), (64, 0), (4097, 24), (64, 4097)] {
        assert!(
            matches!(
                rectify(&sample, width, height),
                Err(RenderError::InvalidCrop(_))
            ),
            "{width}×{height}"
        );
    }
    let mut broken = sample.clone();
    broken.rgb.pop();
    assert!(matches!(
        rectify(&broken, 64, 24),
        Err(RenderError::InvalidCrop(_))
    ));
}
