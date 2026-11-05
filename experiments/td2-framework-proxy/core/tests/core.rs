// SPDX-License-Identifier: Apache-2.0
//! Contract tests for the framework-independent proxy core.
use fyp_core::data::{batch, Sample, IMAGE, SIDE};
use fyp_core::loss::{loss_and_gradient, HEAD_CHANNELS, HEAD_SIDE, STRIDE};
use fyp_core::net::{initial_parameters, tensors, PARAMETER_COUNT, SEED};
use fyp_core::reference::{
    batch_norm_train, conv2d, finite_difference, reference_head, reference_loss, upsample2, Conv,
};
use fyp_core::sample_plan::gradient_samples;
use fyp_core::sgd::Sgd;
use fyp_core::SplitMix64;

#[test]
fn splitmix64_matches_published_reference_sequence() {
    // First outputs of SplitMix64 seeded with 0 (Steele, Lea, Flood 2014).
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(r.next_u64(), 0x06c4_5d18_8009_454f);
}

#[test]
fn batches_are_deterministic_and_boxes_lie_inside_the_image() {
    let a = batch(SEED, 0, 16);
    let b = batch(SEED, 0, 16);
    assert_eq!(a, b);
    assert_ne!(a, batch(SEED, 1, 16));
    for s in &a {
        assert_eq!(s.pixels.len(), IMAGE);
        assert!(s.pixels.iter().all(|v| (0.0..=1.0).contains(v)));
        let [x0, y0, x1, y1] = s.bbox;
        assert!(x0 < x1 && y0 < y1 && x1 <= SIDE && y1 <= SIDE);
        assert!((8..=40).contains(&(x1 - x0)) && (8..=40).contains(&(y1 - y0)));
    }
}

#[test]
fn rectangle_pixels_carry_the_rectangle_colour() {
    let s: &Sample = &batch(SEED, 3, 1)[0];
    let [x0, y0, _, _] = s.bbox;
    for c in 0..3 {
        assert_eq!(s.pixels[c * SIDE * SIDE + y0 * SIDE + x0], s.colour[c]);
    }
}

#[test]
fn parameter_layout_is_the_predeclared_fyp1_table() {
    let t = tensors();
    let names: Vec<&str> = t.iter().map(|t| t.name).collect();
    assert_eq!(
        names,
        [
            "stem.w",
            "stem.bn.g",
            "stem.bn.b",
            "dw.w",
            "dw.bn.g",
            "dw.bn.b",
            "pw.w",
            "pw.bn.g",
            "pw.bn.b",
            "down.w",
            "down.bn.g",
            "down.bn.b",
            "lateral.w",
            "lateral.b",
            "head.w",
            "head.b",
        ]
    );
    assert_eq!(PARAMETER_COUNT, 22_021);
    assert_eq!(t.last().map(|t| t.offset + t.len), Some(PARAMETER_COUNT));
    let p = initial_parameters();
    assert_eq!(p.len(), PARAMETER_COUNT);
    assert_eq!(p, initial_parameters());
    for t in &t {
        let v = &p[t.offset..t.offset + t.len];
        match t.name.rsplit('.').next() {
            Some("g") => assert!(v.iter().all(|x| *x == 1.0)),
            Some("b") => assert!(v.iter().all(|x| *x == 0.0)),
            _ => {
                let bound = (6.0f32 / t.fan_in as f32).sqrt();
                assert!(v.iter().all(|x| x.abs() <= bound));
                assert!(v.iter().any(|x| *x != 0.0));
            }
        }
    }
}

#[test]
fn convolution_handles_stride_padding_and_groups() {
    // 1x1x3x3 ones, 3x3 ones kernel, stride 2, padding 1 -> 2x2 corner sums of 4.
    let x = vec![1.0f64; 9];
    let w = vec![1.0f64; 9];
    let c = Conv {
        c_in: 1,
        c_out: 1,
        k: 3,
        stride: 2,
        groups: 1,
    };
    let (y, h) = conv2d(&x, 1, 3, &w, None, c);
    assert_eq!(h, 2);
    assert_eq!(y, vec![4.0; 4]);

    // A depthwise conv equals independent single-channel convs.
    let x: Vec<f64> = (0..2 * 16).map(|i| i as f64 * 0.1).collect();
    let w: Vec<f64> = (0..2 * 9).map(|i| 1.0 - i as f64 * 0.05).collect();
    let dw = Conv {
        c_in: 2,
        c_out: 2,
        k: 3,
        stride: 1,
        groups: 2,
    };
    let (y, _) = conv2d(&x, 1, 4, &w, None, dw);
    let single = Conv {
        c_in: 1,
        c_out: 1,
        k: 3,
        stride: 1,
        groups: 1,
    };
    for ch in 0..2 {
        let (yc, _) = conv2d(
            &x[ch * 16..ch * 16 + 16],
            1,
            4,
            &w[ch * 9..ch * 9 + 9],
            None,
            single,
        );
        assert_eq!(&y[ch * 16..ch * 16 + 16], &yc[..]);
    }
}

#[test]
fn batch_norm_normalises_each_channel_over_the_batch() {
    let x: Vec<f64> = (0..2 * 2 * 4).map(|i| ((i * 7) % 5) as f64).collect();
    let y = batch_norm_train(&x, 2, 2, 4, &[1.0, 1.0], &[0.0, 0.0]);
    for c in 0..2 {
        let v: Vec<f64> = (0..2)
            .flat_map(|n| y[(n * 2 + c) * 4..(n * 2 + c) * 4 + 4].to_vec())
            .collect();
        let mean = v.iter().sum::<f64>() / 8.0;
        let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / 8.0;
        assert!(mean.abs() < 1e-12);
        assert!((var - 1.0).abs() < 1e-4);
    }
}

#[test]
fn nearest_upsample_repeats_each_value_in_a_2x2_block() {
    let y = upsample2(&[1.0, 2.0, 3.0, 4.0], 1, 2);
    assert_eq!(
        y,
        [1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]
    );
}

#[test]
fn loss_gradient_matches_finite_differences_of_the_loss() {
    let samples = batch(SEED, 0, 2);
    let n = HEAD_CHANNELS * HEAD_SIDE * HEAD_SIDE;
    let mut head: Vec<f64> = (0..2 * n)
        .map(|i| ((i * 37 % 101) as f64 / 50.0) - 1.0)
        .collect();
    let (_, g) = loss_and_gradient(&head, &samples);
    assert_eq!(STRIDE, 4);
    let h = 1e-6;
    for i in (0..2 * n).step_by(97).chain([0, 1, 2, 3, 4]) {
        let keep = head[i];
        head[i] = keep + h;
        let (up, _) = loss_and_gradient(&head, &samples);
        head[i] = keep - h;
        let (down, _) = loss_and_gradient(&head, &samples);
        head[i] = keep;
        let fd = (up - down) / (2.0 * h);
        assert!(
            (fd - g[i]).abs() <= 1e-7 + 1e-5 * fd.abs(),
            "index {i}: fd {fd} vs {}",
            g[i]
        );
    }
}

#[test]
fn head_bias_finite_difference_equals_summed_head_gradient() {
    // Ties the F64 reference forward and finite differences to the loss gradient:
    // dL/d(head.b[c]) = sum over batch and cells of dL/dY[c].
    let samples = batch(SEED, 0, 2);
    let p: Vec<f64> = initial_parameters().iter().map(|v| *v as f64).collect();
    let head = reference_head(&p, &samples);
    let (_, g) = loss_and_gradient(&head, &samples);
    let cells = HEAD_SIDE * HEAD_SIDE;
    let bias = tensors()
        .iter()
        .find(|t| t.name == "head.b")
        .map(|t| t.offset)
        .unwrap();
    for c in 0..HEAD_CHANNELS {
        let analytic: f64 = (0..2)
            .map(|n| {
                g[(n * HEAD_CHANNELS + c) * cells..][..cells]
                    .iter()
                    .sum::<f64>()
            })
            .sum();
        let fd = finite_difference(&p, &samples, bias + c);
        assert!(
            (fd - analytic).abs() <= 1e-8 + 1e-5 * analytic.abs(),
            "channel {c}: {fd} vs {analytic}"
        );
    }
    assert!(reference_loss(&p, &samples).is_finite());
}

#[test]
fn gradient_sample_plan_is_the_predeclared_125_parameters() {
    let plan = gradient_samples();
    assert_eq!(plan, gradient_samples());
    assert_eq!(plan.len(), 125);
    let mut sorted = plan.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 125);
    for t in tensors() {
        let k = plan
            .iter()
            .filter(|i| (t.offset..t.offset + t.len).contains(i))
            .count();
        assert_eq!(k, t.len.min(8), "{}", t.name);
    }
}

#[test]
fn sgd_with_momentum_follows_the_predeclared_update() {
    let mut p = vec![1.0f32, -2.0];
    let mut opt = Sgd::new(2);
    opt.step(&mut p, &[0.5, -1.0]);
    assert_eq!(p, [1.0 - 0.05 * 0.5, -2.0 + 0.05]);
    opt.step(&mut p, &[0.5, -1.0]);
    // v = 0.9 * 0.5 + 0.5 = 0.95
    assert!((p[0] - (0.975 - 0.05 * 0.95)).abs() < 1e-7);
}
