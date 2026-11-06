// SPDX-License-Identifier: Apache-2.0
//! Behaviour tests for the synthetic plate renderer.

use td2_plates::{Format, GLYPHS, RenderError, Sample, Scene, glyph_mask, render};

fn clean_scene() -> Scene {
    Scene {
        width: 192,
        height: 96,
        max_perspective: 0.0,
        max_blur: 0,
        noise_amplitude: 0,
        min_contrast: 0.4,
    }
}

fn rough_scene() -> Scene {
    Scene {
        width: 160,
        height: 96,
        max_perspective: 0.25,
        max_blur: 2,
        noise_amplitude: 24,
        min_contrast: 0.3,
    }
}

fn fnv1a(sample: &Sample) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for &byte in &sample.rgb {
        eat(byte);
    }
    for byte in sample.text.bytes() {
        eat(byte);
    }
    let labels = sample
        .chars
        .iter()
        .flat_map(|label| label.quad.into_iter().flatten().chain(label.bbox));
    for value in sample.plate_quad.into_iter().flatten().chain(labels) {
        for byte in value.to_bits().to_le_bytes() {
            eat(byte);
        }
    }
    for byte in sample.plate_rgb.into_iter().chain(sample.text_rgb) {
        eat(byte);
    }
    hash
}

fn luminance(rgb: [u8; 3]) -> f64 {
    (0.2126 * f64::from(rgb[0]) + 0.7152 * f64::from(rgb[1]) + 0.0722 * f64::from(rgb[2])) / 255.0
}

fn pixel(sample: &Sample, x: usize, y: usize) -> [u8; 3] {
    let at = (y * sample.width + x) * 3;
    [sample.rgb[at], sample.rgb[at + 1], sample.rgb[at + 2]]
}

fn distance2(a: [u8; 3], b: [u8; 3]) -> i32 {
    (0..3)
        .map(|i| {
            let d = i32::from(a[i]) - i32::from(b[i]);
            d * d
        })
        .sum()
}

/// Point strictly inside a convex quad given in clockwise image order.
fn inside_quad(quad: &[[f64; 2]; 4], x: f64, y: f64, margin: f64) -> bool {
    (0..4).all(|i| {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
        let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        cross / length > margin
    })
}

#[test]
fn every_glyph_is_drawn_and_distinct() {
    let height = 36;
    let masks: Vec<(char, Vec<bool>)> = GLYPHS
        .chars()
        .map(|glyph| (glyph, glyph_mask(glyph, height).expect("glyph in set")))
        .collect();
    assert_eq!(masks.len(), 37);
    let cells = masks[0].1.len();
    assert_eq!(cells, height * (height * 2 / 3));
    for (glyph, mask) in &masks {
        let ink = mask.iter().filter(|&&on| on).count();
        let least = if *glyph == '-' { 50 } else { 20 };
        assert!(
            ink * least > cells,
            "glyph {glyph} has too little ink: {ink}/{cells}"
        );
        // Bold strokes, as on plates, but every glyph keeps open counters.
        assert!(
            ink * 10 < cells * 7,
            "glyph {glyph} is mostly ink: {ink}/{cells}"
        );
    }
    let mut closest = (usize::MAX, ' ', ' ');
    for (i, (a, mask_a)) in masks.iter().enumerate() {
        for (b, mask_b) in &masks[i + 1..] {
            let differ = mask_a.iter().zip(mask_b).filter(|(x, y)| x != y).count();
            if differ < closest.0 {
                closest = (differ, *a, *b);
            }
        }
    }
    // The closest pair (slashed zero against O) still differs by a full stroke.
    assert!(
        closest.0 * 25 > cells,
        "glyphs {} and {} differ in only {} of {cells} cells",
        closest.1,
        closest.2,
        closest.0
    );
    assert_eq!(glyph_mask('a', height), None);
    assert_eq!(glyph_mask('!', height), None);
    assert_eq!(glyph_mask(' ', height), None);
    assert_eq!(glyph_mask('A', 2), None);
    assert_eq!(glyph_mask('A', 4097), None);
    assert!(glyph_mask('A', 3).is_some());
}

#[test]
fn same_seed_is_bit_identical_and_seeds_differ() {
    let scene = rough_scene();
    let first = render(&scene, "AB 12 CD", 7).expect("renders");
    let again = render(&scene, "AB 12 CD", 7).expect("renders");
    assert_eq!(first, again);
    let other = render(&scene, "AB 12 CD", 8).expect("renders");
    assert_ne!(first.rgb, other.rgb);
    assert_ne!(first.plate_quad, other.plate_quad);
    assert_eq!(first.rgb.len(), 160 * 96 * 3);
}

/// Frozen outputs. Rendering uses only IEEE-754 add, subtract, multiply,
/// divide and square root plus integer arithmetic, so these hashes must hold
/// on every platform and Rust version. A change here is a data change and
/// needs a reason in the pull request.
#[test]
fn frozen_render_hashes() {
    let format = Format::new("@@@ ###", "ABCDEFGHJKLMNPRSTUVWXYZ").expect("valid format");
    let cases: [(u64, &str, u64); 4] = [
        (1, "ABC 123", 0x09b6_b6d1_9511_1669),
        (2, "XYZ 789", 0xf249_d42a_f634_eba8),
        (3, "QO0 1I5", 0xec4e_08aa_f48e_9745),
        (4, &format.generate(4), 0x7fbd_69a9_0734_4fc7),
    ];
    let mut actual = Vec::new();
    for (seed, text, _) in &cases {
        actual.push(fnv1a(
            &render(&rough_scene(), text, *seed).expect("renders"),
        ));
    }
    let expected: Vec<u64> = cases.iter().map(|case| case.2).collect();
    assert_eq!(actual, expected, "frozen hashes changed: {actual:#018x?}");
}

#[test]
fn generated_text_follows_the_format() {
    let letters = "ABCDEFGHJKLMNPRSTUVWXYZ";
    let format = Format::new("1@@ ##*-", letters).expect("valid format");
    let mut seen_letters = std::collections::BTreeSet::new();
    for seed in 0..2000 {
        let text = format.generate(seed);
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(chars.len(), 8, "{text}");
        assert_eq!(chars[0], '1');
        assert!(
            letters.contains(chars[1]) && letters.contains(chars[2]),
            "{text}"
        );
        assert_eq!(chars[3], ' ');
        assert!(
            chars[4].is_ascii_digit() && chars[5].is_ascii_digit(),
            "{text}"
        );
        assert!(
            chars[6].is_ascii_digit() || letters.contains(chars[6]),
            "{text}"
        );
        assert_eq!(chars[7], '-');
        seen_letters.extend([chars[1], chars[2]]);
        assert_eq!(format.generate(seed), text, "generation is deterministic");
    }
    assert_eq!(
        seen_letters.len(),
        letters.len(),
        "every pool letter is reachable"
    );
    assert!(render(&clean_scene(), &format.generate(5), 5).is_ok());
}

#[test]
fn labels_cover_exactly_the_drawn_characters() {
    for (scene, seeds) in [
        (clean_scene(), 0..12_u64),
        (
            Scene {
                max_perspective: 0.25,
                ..clean_scene()
            },
            12..24,
        ),
    ] {
        for seed in seeds {
            let text = "K7 W-M2";
            let sample = render(&scene, text, seed).expect("renders");
            let glyphs: Vec<char> = text.chars().filter(|&c| c != ' ').collect();
            assert_eq!(sample.chars.len(), glyphs.len());
            let ink = sample.text_rgb;
            let paper = sample.plate_rgb;
            assert!((luminance(ink) - luminance(paper)).abs() >= scene.min_contrast);
            let mut previous_left = f64::NEG_INFINITY;
            for (label, &glyph) in sample.chars.iter().zip(&glyphs) {
                assert_eq!(label.glyph, glyph);
                let [left, top, right, bottom] = label.bbox;
                assert!(left >= 0.0 && top >= 0.0 && right <= 192.0 && bottom <= 96.0);
                assert!(left < right && top < bottom);
                assert!(left > previous_left, "labels run left to right");
                previous_left = left;
                for corner in label.quad {
                    assert!(inside_quad(&sample.plate_quad, corner[0], corner[1], 0.0));
                }
                // Each character box holds ink.
                let mut ink_pixels = 0;
                for y in top.floor() as usize..bottom.ceil() as usize {
                    for x in left.floor() as usize..right.ceil() as usize {
                        if distance2(pixel(&sample, x, y), ink) * 4 < distance2(ink, paper) {
                            ink_pixels += 1;
                        }
                    }
                }
                assert!(
                    ink_pixels >= 6,
                    "seed {seed} glyph {glyph}: {ink_pixels} ink pixels"
                );
            }
            if scene.max_perspective == 0.0 {
                for pair in sample.chars.windows(2) {
                    assert!(
                        pair[0].bbox[2] < pair[1].bbox[0],
                        "seed {seed}: boxes overlap"
                    );
                }
                // Glyphs that touch all four cell sides fill their boxes.
                for label in sample
                    .chars
                    .iter()
                    .filter(|label| "K7WM2".contains(label.glyph))
                {
                    let [left, top, right, bottom] = label.bbox;
                    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, 0.0_f64, 0.0_f64);
                    for y in top.floor() as usize..bottom.ceil() as usize {
                        for x in left.floor() as usize..right.ceil() as usize {
                            if distance2(pixel(&sample, x, y), ink) * 4 < distance2(ink, paper) {
                                (x0, y0) = (x0.min(x as f64), y0.min(y as f64));
                                (x1, y1) = (x1.max(x as f64 + 1.0), y1.max(y as f64 + 1.0));
                            }
                        }
                    }
                    for (drawn, labelled) in [(x0, left), (y0, top), (x1, right), (y1, bottom)] {
                        assert!(
                            (drawn - labelled).abs() <= 1.5,
                            "seed {seed} glyph {}: ink {drawn} vs box {labelled}",
                            label.glyph
                        );
                    }
                }
            }
            // Every inked plate pixel belongs to some character box.
            for y in 0..sample.height {
                for x in 0..sample.width {
                    let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                    if !inside_quad(&sample.plate_quad, cx, cy, 1.5) {
                        continue;
                    }
                    if distance2(pixel(&sample, x, y), ink) * 4 >= distance2(ink, paper) {
                        continue;
                    }
                    let covered = sample.chars.iter().any(|label| {
                        let [left, top, right, bottom] = label.bbox;
                        cx >= left - 1.0
                            && cx <= right + 1.0
                            && cy >= top - 1.0
                            && cy <= bottom + 1.0
                    });
                    assert!(
                        covered,
                        "seed {seed}: ink at ({x}, {y}) outside every character box"
                    );
                }
            }
        }
    }
}

#[test]
fn perspective_moves_corners_but_keeps_the_plate_in_frame() {
    let scene = rough_scene();
    let mut skewed = 0;
    for seed in 0..64 {
        let sample = render(&scene, "ABC123", seed).expect("renders");
        let q = sample.plate_quad;
        for corner in q {
            assert!(
                corner[0] >= 0.0 && corner[0] <= 160.0 && corner[1] >= 0.0 && corner[1] <= 96.0
            );
        }
        // Clockwise from top-left.
        assert!(q[0][0] < q[1][0] && q[3][0] < q[2][0] && q[0][1] < q[3][1] && q[1][1] < q[2][1]);
        if (q[0][1] - q[1][1]).abs() > 1.0 || (q[0][0] - q[3][0]).abs() > 1.0 {
            skewed += 1;
        }
    }
    assert!(skewed > 32, "only {skewed}/64 plates were skewed");
    let flat = render(&clean_scene(), "ABC123", 3).expect("renders");
    let q = flat.plate_quad;
    assert_eq!(q[0][1], q[1][1]);
    assert_eq!(q[0][0], q[3][0]);
}

#[test]
fn invalid_text_is_rejected() {
    let scene = clean_scene();
    assert_eq!(render(&scene, "", 0), Err(RenderError::EmptyText));
    assert_eq!(render(&scene, "   ", 0), Err(RenderError::EmptyText));
    assert_eq!(
        render(&scene, "AB!1", 0),
        Err(RenderError::UnknownGlyph('!'))
    );
    assert_eq!(
        render(&scene, "ab12", 0),
        Err(RenderError::UnknownGlyph('a'))
    );
    assert_eq!(
        render(&scene, &"A".repeat(40), 0),
        Err(RenderError::TextTooLong {
            glyphs: 40,
            max: 12
        })
    );
    assert!(render(&scene, &"A".repeat(12), 0).is_ok());
    // `max` counts a text without spaces, even when spaces made it too long.
    assert_eq!(
        render(&scene, &["A"; 13].join(" "), 0),
        Err(RenderError::TextTooLong {
            glyphs: 13,
            max: 12
        })
    );
}

#[test]
fn invalid_scenes_are_rejected() {
    let base = clean_scene();
    let cases = [
        Scene { width: 15, ..base },
        Scene {
            height: 4097,
            ..base
        },
        Scene {
            width: 64,
            height: 128,
            ..base
        },
        Scene {
            max_perspective: f64::NAN,
            ..base
        },
        Scene {
            max_perspective: -0.01,
            ..base
        },
        Scene {
            max_perspective: 0.31,
            ..base
        },
        Scene {
            max_blur: 4,
            ..base
        },
        Scene {
            min_contrast: 0.91,
            ..base
        },
        Scene {
            min_contrast: f64::INFINITY,
            ..base
        },
        // Too small for one character, flat or with perspective.
        Scene {
            width: 26,
            height: 26,
            ..base
        },
        Scene {
            width: 4096,
            height: 26,
            ..base
        },
        Scene {
            width: 40,
            height: 40,
            max_perspective: 0.3,
            ..base
        },
    ];
    for scene in cases {
        assert!(
            matches!(render(&scene, "AB12", 0), Err(RenderError::InvalidScene(_))),
            "{scene:?} was accepted"
        );
    }
}

#[test]
fn invalid_formats_are_rejected() {
    let pool = "ABC";
    assert!(matches!(
        Format::new("", pool),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("   ", pool),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("@#?", pool),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("@a#", pool),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("@##", ""),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("@##", "AB1"),
        Err(RenderError::InvalidFormat(_))
    ));
    assert!(matches!(
        Format::new("@##", "ABA"),
        Err(RenderError::InvalidFormat(_))
    ));
    // A digit-only pattern does not need letters, but the pool is still validated.
    assert!(Format::new("###", "ABC").is_ok());
    assert!(Format::new("ABC-12", "Z").is_ok());
}

fn strictly_convex(quad: &[[f64; 2]; 4]) -> bool {
    (0..4).all(|i| {
        let [a, b, c] = [quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]];
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0.0
    })
}

#[test]
fn jittered_plates_stay_convex() {
    let scene = Scene {
        width: 64,
        height: 64,
        max_perspective: 0.3,
        max_blur: 0,
        noise_amplitude: 0,
        min_contrast: 0.4,
    };
    // Seed 558 produced a concave quad before corner jitter was rejected.
    for seed in (0..2000).chain([558]) {
        let sample = render(&scene, "A", seed).expect("renders");
        assert!(
            strictly_convex(&sample.plate_quad),
            "seed {seed}: {:?}",
            sample.plate_quad
        );
        for label in &sample.chars {
            assert!(
                strictly_convex(&label.quad),
                "seed {seed}: cell {:?}",
                label.quad
            );
        }
    }
}

#[test]
fn colours_vary_at_high_contrast() {
    for contrast in [0.6, 0.9] {
        let scene = Scene {
            min_contrast: contrast,
            ..clean_scene()
        };
        let mut pairs = std::collections::BTreeSet::new();
        let mut darker_text = 0;
        for seed in 0..200 {
            let sample = render(&scene, "AB12", seed).expect("renders");
            let (paper, ink) = (luminance(sample.plate_rgb), luminance(sample.text_rgb));
            assert!(
                (paper - ink).abs() >= contrast,
                "seed {seed}: {paper} vs {ink}"
            );
            pairs.insert((sample.plate_rgb, sample.text_rgb));
            darker_text += usize::from(ink < paper);
        }
        assert!(
            pairs.len() > 150,
            "contrast {contrast}: only {} colour pairs",
            pairs.len()
        );
        assert!(
            (20..=180).contains(&darker_text),
            "contrast {contrast}: {darker_text}/200 dark text"
        );
    }
}

#[test]
fn noise_stays_within_its_amplitude() {
    for seed in 0..8 {
        let quiet = render(&clean_scene(), "AB12", seed).expect("renders");
        let noisy = render(
            &Scene {
                noise_amplitude: 20,
                ..clean_scene()
            },
            "AB12",
            seed,
        )
        .expect("renders");
        let largest = quiet
            .rgb
            .iter()
            .zip(&noisy.rgb)
            .map(|(a, b)| a.abs_diff(*b))
            .max();
        assert!(
            largest.is_some_and(|d| (1..=20).contains(&d)),
            "seed {seed}: {largest:?}"
        );
        assert_eq!(quiet.chars, noisy.chars, "noise leaves labels alone");
    }
}
