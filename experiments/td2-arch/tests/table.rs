// SPDX-License-Identifier: Apache-2.0
use std::collections::HashSet;
use td2_arch::*;

fn frozen(classes: u32, size: u32) -> Report {
    analyze(&table(classes).unwrap(), size, classes, &HEADS).unwrap()
}

#[test]
fn frozen_totals_at_baseline_with_max_classes() {
    let r = frozen(MAX_CLASSES, BASELINE_SIZE);
    // Frozen: any change to the table changes these numbers and must be
    // recorded in the spec before training.
    assert_eq!(r.parameters, 1_757_463);
    assert_eq!(r.flops, 1_454_700_900);
    assert_eq!(r.layers.len(), 91);
}

#[test]
fn budgets_hold_for_every_class_count() {
    for classes in 1..=MAX_CLASSES {
        let r = frozen(classes, BASELINE_SIZE);
        assert!(
            r.parameters <= MAX_PARAMETERS,
            "{classes}: {}",
            r.parameters
        );
        assert!(r.flops <= MAX_FLOPS_AT_BASELINE, "{classes}: {}", r.flops);
        // F32 weights within the 12 MB budget of section 6.
        assert!(r.parameters * 4 <= 12_000_000);
    }
}

#[test]
fn accounting_matches_hand_computed_layers() {
    let r = frozen(MAX_CLASSES, BASELINE_SIZE);
    let get = |n: &str| r.layers.iter().find(|l| l.name == n).unwrap().clone();
    // stem.0: 3x3 conv, 3 -> 24, stride 2, 160x160 output.
    let s = get("stem.0");
    assert_eq!(s.parameters, 3 * 3 * 3 * 24 + 24);
    assert_eq!(s.flops, 2 * 27 * 24 * 160 * 160 + 24 * 160 * 160);
    // c3.0.dw: depthwise 3x3 on 192 channels at 40x40.
    let d = get("c3.0.dw");
    assert_eq!(d.parameters, 192 * 9 + 192);
    assert_eq!(d.flops, 2 * 9 * 192 * 40 * 40 + 192 * 40 * 40);
    // c3.0: residual add at 96x40x40, no parameters.
    let a = get("c3.0");
    assert_eq!((a.parameters, a.flops), (0, 96 * 40 * 40));
    // Upsample is free.
    let u = get("td4.up");
    assert_eq!((u.parameters, u.flops), (0, 0));
    assert_eq!(
        u.shape,
        Shape {
            c: 96,
            h: 20,
            w: 20
        }
    );
    // Head output: 1x1, 96 -> 121 at 40x40.
    let h = get("p8.cls.out");
    assert_eq!(h.parameters, 96 * 121 + 121);
    let sum_p: u64 = r.layers.iter().map(|l| l.parameters).sum();
    let sum_f: u64 = r.layers.iter().map(|l| l.flops).sum();
    assert_eq!((sum_p, sum_f), (r.parameters, r.flops));
}

#[test]
fn heads_have_expected_shapes_at_every_evaluation_size() {
    for size in [320, 416, 640] {
        for classes in [1, 82, MAX_CLASSES] {
            let r = frozen(classes, size);
            for head in HEADS {
                let side = size / head.stride;
                assert_eq!(
                    r.shape(head.class_layer),
                    Some(Shape {
                        c: classes + 1,
                        h: side,
                        w: side
                    })
                );
                assert_eq!(
                    r.shape(head.box_layer),
                    Some(Shape {
                        c: 4,
                        h: side,
                        w: side
                    })
                );
            }
            assert_eq!(HEADS.map(|h| h.stride), HEAD_STRIDES);
        }
    }
}

#[test]
fn parameters_do_not_depend_on_size_and_flops_scale_with_area() {
    let base = frozen(MAX_CLASSES, 320);
    let big = frozen(MAX_CLASSES, 640);
    assert_eq!(base.parameters, big.parameters);
    assert_eq!(big.flops, 4 * base.flops);
}

#[test]
fn every_non_output_layer_is_consumed_and_operators_are_limited() {
    let layers = table(MAX_CLASSES).unwrap();
    let mut used = HashSet::new();
    for l in &layers {
        match l.op {
            Op::Conv {
                input,
                kernel,
                depthwise,
                ..
            } => {
                used.insert(input);
                assert!(kernel == 1 || kernel == 3, "{}", l.name);
                if depthwise {
                    assert_eq!(kernel, 3, "{}", l.name);
                }
            }
            Op::Upsample2 { input } => {
                used.insert(input);
            }
            Op::Add { a, b } => {
                used.insert(a);
                used.insert(b);
            }
        }
    }
    let outputs: HashSet<_> = HEADS
        .iter()
        .flat_map(|h| [h.class_layer, h.box_layer])
        .collect();
    for l in &layers {
        assert_eq!(
            used.contains(l.name),
            !outputs.contains(l.name),
            "{} dead or output consumed",
            l.name
        );
    }
}

#[test]
fn rejects_class_counts_and_input_sizes_out_of_range() {
    assert_eq!(table(0), Err(ArchError::ClassCount(0)));
    assert_eq!(table(121), Err(ArchError::ClassCount(121)));
    let t = table(10).unwrap();
    assert_eq!(analyze(&t, 320, 0, &HEADS), Err(ArchError::ClassCount(0)));
    assert_eq!(
        analyze(&t, 320, 121, &HEADS),
        Err(ArchError::ClassCount(121))
    );
    for size in [0, 100, 330, 336] {
        assert_eq!(
            analyze(&t, size, 10, &HEADS),
            Err(ArchError::InputSize(size))
        );
    }
    // Wrong class count for the table: head shape mismatch.
    assert!(matches!(
        analyze(&t, 320, 11, &HEADS),
        Err(ArchError::HeadShape {
            layer: "p8.cls.out",
            ..
        })
    ));
}

fn conv(
    name: &'static str,
    input: &'static str,
    out: u32,
    kernel: u32,
    stride: u32,
    depthwise: bool,
) -> Layer {
    Layer {
        name,
        op: Op::Conv {
            input,
            out,
            kernel,
            stride,
            depthwise,
            act: Act::Silu,
        },
    }
}

#[test]
fn rejects_malformed_tables() {
    let ok = conv("a", INPUT, 8, 3, 2, false);
    let mut cases: Vec<(Vec<Layer>, ArchError)> = vec![
        (
            vec![ok.clone(), conv("a", "a", 8, 1, 1, false)],
            ArchError::DuplicateName("a"),
        ),
        (
            vec![conv("a", "missing", 8, 1, 1, false)],
            ArchError::UnknownInput {
                layer: "a",
                input: "missing",
            },
        ),
        (
            vec![conv("a", INPUT, 8, 2, 1, false)],
            ArchError::BadKernel {
                layer: "a",
                kernel: 2,
            },
        ),
        (
            vec![conv("a", INPUT, 8, 0, 1, false)],
            ArchError::BadKernel {
                layer: "a",
                kernel: 0,
            },
        ),
        (
            vec![conv("a", INPUT, 8, 3, 0, false)],
            ArchError::BadStride {
                layer: "a",
                stride: 0,
                size: (64, 64),
            },
        ),
        (
            vec![conv("a", INPUT, 8, 3, 3, false)],
            ArchError::BadStride {
                layer: "a",
                stride: 3,
                size: (64, 64),
            },
        ),
        (
            vec![conv("a", INPUT, 0, 1, 1, false)],
            ArchError::ZeroChannels("a"),
        ),
        (
            vec![conv("a", INPUT, 8, 3, 1, true)],
            ArchError::DepthwiseWidth {
                layer: "a",
                input: 3,
                output: 8,
            },
        ),
        (
            vec![
                ok.clone(),
                Layer {
                    name: "s",
                    op: Op::Add { a: "a", b: INPUT },
                },
            ],
            ArchError::AddShape {
                layer: "s",
                a: Shape { c: 8, h: 32, w: 32 },
                b: Shape { c: 3, h: 64, w: 64 },
            },
        ),
        (
            vec![Layer {
                name: "u",
                op: Op::Upsample2 { input: "nope" },
            }],
            ArchError::UnknownInput {
                layer: "u",
                input: "nope",
            },
        ),
    ];
    // Same channels, different spatial size.
    cases.push((
        vec![
            ok.clone(),
            conv("c", INPUT, 8, 1, 1, false),
            Layer {
                name: "s",
                op: Op::Add { a: "a", b: "c" },
            },
        ],
        ArchError::AddShape {
            layer: "s",
            a: Shape { c: 8, h: 32, w: 32 },
            b: Shape { c: 8, h: 64, w: 64 },
        },
    ));
    for (layers, expected) in cases {
        assert_eq!(analyze(&layers, 64, 1, &[]), Err(expected));
    }
    // A table without the declared heads is rejected, not accepted silently.
    assert_eq!(
        analyze(&[ok], 64, 1, &HEADS),
        Err(ArchError::UnknownInput {
            layer: "heads",
            input: "p8.cls.out"
        })
    );
}

/// Canonical text of the table: every layer's name, kind, inputs and
/// settings, independent of `Debug` formatting.
fn canonical(layers: &[Layer]) -> String {
    let mut out = String::new();
    for l in layers {
        let entry = match l.op {
            Op::Conv {
                input,
                out,
                kernel,
                stride,
                depthwise,
                act,
            } => {
                let act = match act {
                    Act::Linear => "linear",
                    Act::Silu => "silu",
                };
                format!(
                    "{}=conv({input},{out},k{kernel},s{stride},dw{depthwise},{act});",
                    l.name
                )
            }
            Op::Upsample2 { input } => format!("{}=up2({input});", l.name),
            Op::Add { a, b } => format!("{}=add({a},{b});", l.name),
        };
        out.push_str(&entry);
    }
    out
}

/// FNV-1a 64 (fingerprint only, not security).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn wiring_fingerprint_is_frozen() {
    let text = canonical(&table(MAX_CLASSES).unwrap());
    assert!(text.starts_with("stem.0=conv(input,24,k3,s2,dwfalse,silu);"));
    assert!(text.contains("bu4.sum=add(bu4.down,td4);"));
    assert_eq!(fnv1a(text.as_bytes()), 10_822_089_495_293_790_261);
}
