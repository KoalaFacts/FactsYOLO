use td2_decode::{BoxEdges, DecodeError, DecodeOptions, Detection, Letterbox, LevelOutput, decode};

/// Logit whose sigmoid is `p`.
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

const LOW: f32 = -30.0;

/// One level with every location silent; callers switch locations on.
struct Level {
    stride: u32,
    grid: usize,
    classes: usize,
    objectness: Vec<f32>,
    class_logits: Vec<f32>,
    distances: Vec<f32>,
}

impl Level {
    fn new(size: u32, stride: u32, classes: usize) -> Self {
        let grid = (size / stride) as usize;
        let cells = grid * grid;
        Self {
            stride,
            grid,
            classes,
            objectness: vec![LOW; cells],
            class_logits: vec![LOW; cells * classes],
            distances: vec![1.0; cells * 4],
        }
    }

    /// Switch on location (x, y) for one class with distances in stride units.
    fn set(&mut self, x: usize, y: usize, class: usize, obj: f32, cls: f32, ltrb: [f32; 4]) {
        let cell = y * self.grid + x;
        self.objectness[cell] = logit(obj);
        self.class_logits[cell * self.classes + class] = logit(cls);
        self.distances[cell * 4..cell * 4 + 4].copy_from_slice(&ltrb);
    }

    fn view(&self) -> LevelOutput<'_> {
        LevelOutput {
            stride: self.stride,
            objectness: &self.objectness,
            class_logits: &self.class_logits,
            distances: &self.distances,
        }
    }
}

fn opts(threshold: f32) -> DecodeOptions {
    DecodeOptions {
        score_threshold: threshold,
        ..DecodeOptions::default()
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * b.abs().max(1.0)
}

fn assert_box(d: &Detection, l: f32, t: f32, r: f32, b: f32) {
    let e = d.edges;
    assert!(
        close(e.left, l) && close(e.top, t) && close(e.right, r) && close(e.bottom, b),
        "got {e:?}, want ({l}, {t}, {r}, {b})"
    );
}

#[test]
fn defaults_match_the_spec() {
    let d = DecodeOptions::default();
    assert_eq!(d.pre_nms_top_k, 1000);
    assert_eq!(d.nms_iou, 0.65);
}

#[test]
fn letterbox_centres_a_wide_frame() {
    let lb = Letterbox::new(640, 320, 320).unwrap();
    assert_eq!(lb.scale(), 0.5);
    assert_eq!(lb.pad_x(), 0.0);
    assert_eq!(lb.pad_y(), 80.0);
    let tall = Letterbox::new(100, 400, 320).unwrap();
    assert_eq!(tall.scale(), 0.8);
    assert_eq!(tall.pad_x(), 120.0);
    assert_eq!(tall.pad_y(), 0.0);
}

#[test]
fn letterbox_rejects_invalid_sizes() {
    for (w, h, s) in [
        (0, 10, 320),
        (10, 0, 320),
        (10, 10, 0),
        (10, 10, 300),
        (10, 10, 31),
    ] {
        assert!(
            matches!(Letterbox::new(w, h, s), Err(DecodeError::InvalidSize)),
            "{w}x{h} S={s}"
        );
    }
}

#[test]
fn letterbox_maps_model_boxes_back_and_clips() {
    let lb = Letterbox::new(640, 320, 320).unwrap();
    let inside = lb
        .to_source(BoxEdges {
            left: 10.0,
            top: 90.0,
            right: 20.0,
            bottom: 100.0,
        })
        .unwrap();
    assert_eq!(
        inside,
        BoxEdges {
            left: 20.0,
            top: 20.0,
            right: 40.0,
            bottom: 40.0
        }
    );
    // Reaching into the top padding is clipped to the frame edge.
    let clipped = lb
        .to_source(BoxEdges {
            left: 0.0,
            top: 60.0,
            right: 10.0,
            bottom: 90.0,
        })
        .unwrap();
    assert_eq!(clipped.top, 0.0);
    assert_eq!(clipped.bottom, 20.0);
    // Entirely inside the padding: nothing left.
    assert!(
        lb.to_source(BoxEdges {
            left: 0.0,
            top: 0.0,
            right: 10.0,
            bottom: 70.0
        })
        .is_none()
    );
}

#[test]
fn single_location_decodes_with_fused_score() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 8, 3);
    // Location (4, 5) at stride 8 has centre (36, 44).
    level.set(4, 5, 2, 0.9, 0.4, [1.0, 2.0, 3.0, 0.5]);
    let out = decode(&[level.view()], &lb, &opts(0.1)).unwrap();
    assert!(!out.truncated);
    assert_eq!(out.detections.len(), 1);
    let d = &out.detections[0];
    assert_eq!(d.class_id, 2);
    assert!(close(d.score, (0.9f32 * 0.4).sqrt()), "score {}", d.score);
    assert_box(d, 28.0, 28.0, 60.0, 48.0);
}

#[test]
fn scores_below_threshold_are_dropped() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 32, 1);
    level.set(1, 1, 0, 0.5, 0.5, [1.0; 4]); // score 0.5
    level.set(6, 6, 0, 0.9, 0.9, [1.0; 4]); // score 0.9
    let out = decode(&[level.view()], &lb, &opts(0.6)).unwrap();
    assert_eq!(out.detections.len(), 1);
    assert!(close(out.detections[0].score, 0.9));
}

#[test]
fn nms_is_per_class_at_iou_065() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 8, 2);
    // Same box for two classes: both kept. A same-class box shifted 6 px:
    // IoU = 26 / 38 ≈ 0.68, suppressed.
    level.set(10, 10, 0, 0.95, 0.95, [2.0; 4]);
    level.set(10, 10, 1, 0.95, 0.9, [2.0; 4]);
    level.set(11, 10, 0, 0.9, 0.9, [2.25, 2.0, 1.75, 2.0]);
    // Same class, IoU = 16 / 48 ≈ 0.33: kept.
    level.set(20, 20, 0, 0.9, 0.9, [2.0; 4]);
    level.set(21, 20, 0, 0.8, 0.8, [1.0, 2.0, 3.0, 2.0]);
    let out = decode(&[level.view()], &lb, &opts(0.5)).unwrap();
    let mut got: Vec<(u32, i32)> = out
        .detections
        .iter()
        .map(|d| (d.class_id, d.edges.left as i32))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![(0, 68), (0, 148), (0, 164), (1, 68)],
        "{:?}",
        out.detections
    );
}

#[test]
fn iou_exactly_at_threshold_is_kept() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 32, 1);
    // A: x 16..216, y 16..48. B: x 16..146, y 16..48, inside A.
    // IoU = 130 / 200 = 0.65 exactly (all values exact in f32): not above the
    // threshold, so B survives.
    level.set(0, 0, 0, 0.9, 0.9, [0.0, 0.0, 200.0 / 32.0, 1.0]);
    level.set(0, 1, 0, 0.8, 0.8, [0.0, 1.0, 130.0 / 32.0, 0.0]);
    let out = decode(&[level.view()], &lb, &opts(0.5)).unwrap();
    assert_eq!(out.detections.len(), 2, "{:?}", out.detections);
}

#[test]
fn output_is_sorted_by_score_then_class_and_capped() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 32, 3);
    level.set(0, 0, 2, 0.81, 1.0 - 1e-6, [0.4; 4]);
    level.set(3, 3, 1, 0.81, 1.0 - 1e-6, [0.4; 4]);
    level.set(6, 6, 0, 0.99, 0.99, [0.4; 4]);
    let out = decode(&[level.view()], &lb, &opts(0.1)).unwrap();
    let classes: Vec<u32> = out.detections.iter().map(|d| d.class_id).collect();
    assert_eq!(classes, vec![0, 1, 2]);
    assert!(!out.truncated);

    let capped = decode(
        &[level.view()],
        &lb,
        &DecodeOptions {
            max_detections: 2,
            ..opts(0.1)
        },
    )
    .unwrap();
    assert_eq!(capped.detections.len(), 2);
    assert!(capped.truncated);
}

#[test]
fn pre_nms_top_k_keeps_the_best_candidates() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 32, 1);
    for i in 0..10 {
        level.set(i, i, 0, 0.5 + 0.04 * i as f32, 0.99, [0.4; 4]);
    }
    let out = decode(
        &[level.view()],
        &lb,
        &DecodeOptions {
            pre_nms_top_k: 3,
            ..opts(0.1)
        },
    )
    .unwrap();
    let cells: Vec<i32> = out
        .detections
        .iter()
        .map(|d| ((d.edges.left + 12.8) / 32.0) as i32)
        .collect();
    assert_eq!(cells, vec![9, 8, 7]);
}

#[test]
fn levels_combine_and_candidates_in_padding_are_dropped() {
    let lb = Letterbox::new(640, 320, 320).unwrap(); // pad_y = 80
    let mut p3 = Level::new(320, 8, 1);
    let mut p5 = Level::new(320, 32, 1);
    p3.set(0, 0, 0, 0.9, 0.9, [0.5; 4]); // y 0..8 in the padding
    p5.set(4, 4, 0, 0.9, 0.8, [1.0; 4]); // centre (144,144): model 112..176
    let out = decode(&[p3.view(), p5.view()], &lb, &opts(0.1)).unwrap();
    assert_eq!(out.detections.len(), 1);
    assert_box(&out.detections[0], 224.0, 64.0, 352.0, 192.0);
}

#[test]
fn malformed_outputs_are_rejected() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let base = Level::new(320, 32, 2);

    let mut neg = Level::new(320, 32, 2);
    neg.distances[7] = -0.1;
    let mut nan_obj = Level::new(320, 32, 2);
    nan_obj.objectness[3] = f32::NAN;
    let mut inf_cls = Level::new(320, 32, 2);
    inf_cls.class_logits[5] = f32::INFINITY;
    let mut inf_dist = Level::new(320, 32, 2);
    inf_dist.distances[0] = f32::INFINITY;
    for bad in [&neg, &nan_obj, &inf_cls, &inf_dist] {
        assert_eq!(
            decode(&[bad.view()], &lb, &opts(0.5)).unwrap_err(),
            DecodeError::OutputInvalid
        );
    }

    let short = LevelOutput {
        distances: &base.distances[1..],
        ..base.view()
    };
    let bad_stride = LevelOutput {
        stride: 24,
        ..base.view()
    };
    let zero_stride = LevelOutput {
        stride: 0,
        ..base.view()
    };
    for bad in [short, bad_stride, zero_stride] {
        assert_eq!(
            decode(&[bad], &lb, &opts(0.5)).unwrap_err(),
            DecodeError::ShapeMismatch
        );
    }
    // Class count is derived from the class logits and must agree across levels.
    let ragged = LevelOutput {
        class_logits: &base.class_logits[1..],
        ..base.view()
    };
    let no_classes = LevelOutput {
        class_logits: &[],
        ..base.view()
    };
    let three = Level::new(320, 16, 3);
    for levels in [
        vec![ragged],
        vec![no_classes],
        vec![base.view(), three.view()],
    ] {
        assert_eq!(
            decode(&levels, &lb, &opts(0.5)).unwrap_err(),
            DecodeError::ShapeMismatch
        );
    }
    assert_eq!(
        decode(&[], &lb, &opts(0.5)).unwrap_err(),
        DecodeError::ShapeMismatch
    );
}

#[test]
fn invalid_options_are_rejected() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let level = Level::new(320, 32, 1);
    let bad = [
        opts(f32::NAN),
        opts(-0.1),
        opts(1.1),
        DecodeOptions {
            nms_iou: 0.0,
            ..opts(0.5)
        },
        DecodeOptions {
            nms_iou: f32::NAN,
            ..opts(0.5)
        },
        DecodeOptions {
            pre_nms_top_k: 0,
            ..opts(0.5)
        },
        DecodeOptions {
            max_detections: 0,
            ..opts(0.5)
        },
    ];
    for o in bad {
        assert_eq!(
            decode(&[level.view()], &lb, &o).unwrap_err(),
            DecodeError::InvalidOptions,
            "{o:?}"
        );
    }
}

#[test]
fn decoding_is_deterministic_under_ties() {
    let lb = Letterbox::new(320, 320, 320).unwrap();
    let mut level = Level::new(320, 32, 1);
    // Identical scores at disjoint locations: order follows location index.
    for (x, y) in [(7, 1), (2, 5), (5, 2)] {
        level.set(x, y, 0, 0.9, 0.9, [0.4; 4]);
    }
    let a = decode(&[level.view()], &lb, &opts(0.1)).unwrap();
    let b = decode(&[level.view()], &lb, &opts(0.1)).unwrap();
    assert_eq!(a, b);
    let tops: Vec<i32> = a.detections.iter().map(|d| d.edges.top as i32).collect();
    assert_eq!(tops, vec![35, 67, 163]);
}
