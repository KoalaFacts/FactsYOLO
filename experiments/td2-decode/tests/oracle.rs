// SPDX-License-Identifier: Apache-2.0
//! Randomized cross-check of `decode` against a deliberately naive,
//! independently structured restatement of the TD2 section 6 decoding
//! contract: explicit (level, row, column, class) loops, per-class NMS run on
//! separate lists, and a merge sorted with the ABI comparator.
//!
//! The scenes come from a fixed-seed generator so failures are reproducible.
//! They concentrate candidates on a few overlapping boxes, repeat scores to
//! exercise ties, and use small top-k and caps so every cut is reached.

use td2_decode::{BoxEdges, DecodeOptions, Letterbox, LevelOutput, decode};

/// SplitMix64; deterministic and dependency free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

struct Scene {
    size: u32,
    width: u32,
    height: u32,
    classes: usize,
    strides: Vec<u32>,
    objectness: Vec<Vec<f32>>,
    class_logits: Vec<Vec<f32>>,
    distances: Vec<Vec<f32>>,
    options: DecodeOptions,
}

fn scene(rng: &mut Rng) -> Scene {
    let size = rng.pick(&[32u32, 64, 96]);
    let width = 8 + rng.below(200) as u32;
    let height = 8 + rng.below(200) as u32;
    let classes = 1 + rng.below(3) as usize;
    let strides: Vec<u32> = [8u32, 16, 32]
        .into_iter()
        .filter(|s| size % s == 0 && rng.below(4) != 0)
        .collect();
    let strides = if strides.is_empty() {
        vec![16]
    } else {
        strides
    };
    // A small logit palette makes exact score ties common.
    let logits = [-30.0f32, -2.0, -0.5, 0.0, 0.5, 1.5, 3.0];
    let dists = [0.0f32, 0.25, 0.5, 1.0, 1.5, 3.0];
    let (mut objectness, mut class_logits, mut distances) = (vec![], vec![], vec![]);
    for &s in &strides {
        let cells = ((size / s) * (size / s)) as usize;
        objectness.push((0..cells).map(|_| rng.pick(&logits)).collect());
        class_logits.push((0..cells * classes).map(|_| rng.pick(&logits)).collect());
        distances.push((0..cells * 4).map(|_| rng.pick(&dists)).collect());
    }
    let options = DecodeOptions {
        score_threshold: rng.pick(&[0.0f32, 0.3, 0.5, 0.7]),
        pre_nms_top_k: rng.pick(&[1usize, 3, 10, 1000]),
        nms_iou: rng.pick(&[0.3f32, 0.5, 0.65, 1.0]),
        max_detections: rng.pick(&[1usize, 2, 5, 100]),
    };
    Scene {
        size,
        width,
        height,
        classes,
        strides,
        objectness,
        class_logits,
        distances,
        options,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Expected {
    class_id: u32,
    score: f32,
    edges: BoxEdges,
}

fn iou(a: &BoxEdges, b: &BoxEdges) -> f32 {
    let area = |e: &BoxEdges| (e.right - e.left).max(0.0) * (e.bottom - e.top).max(0.0);
    let w = (a.right.min(b.right) - a.left.max(b.left)).max(0.0);
    let h = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0);
    let inter = w * h;
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The contract restated without reusing any of the crate's code paths.
fn naive(s: &Scene) -> (Vec<Expected>, bool) {
    let (w, h, size) = (f64::from(s.width), f64::from(s.height), f64::from(s.size));
    let scale = (size / w).min(size / h);
    let pad_x = (size - w * scale) / 2.0;
    let pad_y = (size - h * scale) / 2.0;

    // (score, class, level, row, column, model box)
    let mut all = vec![];
    for (li, &stride) in s.strides.iter().enumerate() {
        let grid = (s.size / stride) as usize;
        for row in 0..grid {
            for col in 0..grid {
                let cell = row * grid + col;
                let p_obj = 1.0 / (1.0 + (-f64::from(s.objectness[li][cell])).exp());
                for class in 0..s.classes {
                    let logit = s.class_logits[li][cell * s.classes + class];
                    let p_cls = 1.0 / (1.0 + (-f64::from(logit)).exp());
                    let score = (p_obj * p_cls).sqrt() as f32;
                    if score < s.options.score_threshold {
                        continue;
                    }
                    let st = stride as f32;
                    let cx = (col as f32 + 0.5) * st;
                    let cy = (row as f32 + 0.5) * st;
                    let d = &s.distances[li][cell * 4..cell * 4 + 4];
                    let model = [
                        cx - d[0] * st,
                        cy - d[1] * st,
                        cx + d[2] * st,
                        cy + d[3] * st,
                    ];
                    all.push((score, class as u32, li, row, col, model));
                }
            }
        }
    }
    // Rank: score descending, class, level, then row-major location.
    all.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(a.4.cmp(&b.4))
    });
    all.truncate(s.options.pre_nms_top_k);

    let mut merged: Vec<(usize, Expected)> = vec![];
    for class in 0..s.classes as u32 {
        let mut kept: Vec<BoxEdges> = vec![];
        for (rank, c) in all.iter().enumerate().filter(|(_, c)| c.1 == class) {
            let map =
                |v: f32, pad: f64, lim: f64| ((f64::from(v) - pad) / scale).clamp(0.0, lim) as f32;
            let e = BoxEdges {
                left: map(c.5[0], pad_x, w),
                top: map(c.5[1], pad_y, h),
                right: map(c.5[2], pad_x, w),
                bottom: map(c.5[3], pad_y, h),
            };
            if e.right <= e.left || e.bottom <= e.top {
                continue;
            }
            if kept.iter().all(|k| iou(k, &e) <= s.options.nms_iou) {
                kept.push(e);
                merged.push((
                    rank,
                    Expected {
                        class_id: class,
                        score: c.0,
                        edges: e,
                    },
                ));
            }
        }
    }
    merged.sort_by(|(ra, a), (rb, b)| {
        b.score
            .total_cmp(&a.score)
            .then(a.class_id.cmp(&b.class_id))
            .then(a.edges.left.total_cmp(&b.edges.left))
            .then(a.edges.top.total_cmp(&b.edges.top))
            .then(a.edges.right.total_cmp(&b.edges.right))
            .then(a.edges.bottom.total_cmp(&b.edges.bottom))
            .then(ra.cmp(rb))
    });
    let truncated = merged.len() > s.options.max_detections;
    merged.truncate(s.options.max_detections);
    (merged.into_iter().map(|(_, e)| e).collect(), truncated)
}

#[test]
fn decode_matches_a_naive_restatement_on_random_scenes() {
    let mut rng = Rng(20_261_106);
    let (mut nonempty, mut truncated_seen, mut suppressed_seen) = (0, 0, 0);
    for case in 0..3000 {
        let s = scene(&mut rng);
        let levels: Vec<LevelOutput<'_>> = (0..s.strides.len())
            .map(|i| LevelOutput {
                stride: s.strides[i],
                objectness: &s.objectness[i],
                class_logits: &s.class_logits[i],
                distances: &s.distances[i],
            })
            .collect();
        let letterbox = Letterbox::new(s.width, s.height, s.size).expect("valid letterbox");
        let got = decode(&levels, &letterbox, &s.options).expect("valid scene decodes");
        let (want, truncated) = naive(&s);
        let got_flat: Vec<Expected> = got
            .detections
            .iter()
            .map(|d| Expected {
                class_id: d.class_id,
                score: d.score,
                edges: d.edges,
            })
            .collect();
        assert_eq!(got_flat, want, "case {case}: {:?}", s.options);
        assert_eq!(got.truncated, truncated, "case {case}");
        nonempty += usize::from(!want.is_empty());
        truncated_seen += usize::from(truncated);
        // A scene where the uncapped NMS output is smaller than the top-k cut
        // means at least one candidate was suppressed or fell in padding.
        let mut uncapped = s.options;
        uncapped.max_detections = usize::MAX;
        let survivors = naive(&Scene {
            options: uncapped,
            ..s
        })
        .0
        .len();
        suppressed_seen +=
            usize::from(survivors > 0 && survivors < uncapped.pre_nms_top_k.min(1000));
    }
    // The generator must actually reach every branch it is meant to test.
    assert!(nonempty > 2000, "nonempty {nonempty}");
    assert!(truncated_seen > 300, "truncated {truncated_seen}");
    assert!(suppressed_seen > 300, "suppressed {suppressed_seen}");
}
