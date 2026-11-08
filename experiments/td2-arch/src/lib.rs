// SPDX-License-Identifier: Apache-2.0
//! Frozen TD2 detector layer table (stage D2b1) with shape, parameter and
//! FLOP accounting.
//!
//! Original network written for section 6 of the TD2 design, fixed before any
//! training so it is not tuned against results. Ideas only from published
//! papers: depthwise-separable convolutions
//! ([MobileNets](https://arxiv.org/abs/1704.04861)), inverted residual blocks
//! ([MobileNetV2](https://arxiv.org/abs/1801.04381)), a top-down plus
//! bottom-up feature pyramid ([FPN](https://arxiv.org/abs/1612.03144),
//! [PANet](https://arxiv.org/abs/1803.01534)) and decoupled heads
//! ([YOLOX](https://arxiv.org/abs/2107.08430)). No code was ported.
//! Development tooling only: never linked into the deployed library, no I/O,
//! no dependencies.
//!
//! The table is a list of named layers; each layer reads earlier layers by
//! name. Batch norm is folded into convolution weight and bias at export, so
//! every convolution here has a bias. Kinds are limited to the stage B
//! operator list (convolution, depthwise convolution, SiLU, nearest 2×
//! upsample; section 10 of the design) plus element-wise add.
//!
//! Accounting rules (budgets in section 6: ≤ 3.0 M parameters, ≤ 2.5 GFLOPs
//! at S = 320):
//! - parameters: convolution weights plus one bias per output channel;
//! - FLOPs: 2 per convolution multiply-accumulate, plus 1 per output element
//!   for the bias and 1 per element of an add; activations and upsampling are
//!   not counted.

use std::collections::HashMap;
use std::fmt;

/// Largest class count the table must fit (section 6).
pub const MAX_CLASSES: u32 = 120;
/// Input image channels after colour conversion.
pub const INPUT_CHANNELS: u32 = 3;
/// Pyramid strides of the three heads, finest first.
pub const HEAD_STRIDES: [u32; 3] = [8, 16, 32];
/// Parameter budget from section 6.
pub const MAX_PARAMETERS: u64 = 3_000_000;
/// FLOP budget from section 6, at `BASELINE_SIZE`.
pub const MAX_FLOPS_AT_BASELINE: u64 = 2_500_000_000;
/// Baseline input size S.
pub const BASELINE_SIZE: u32 = 320;
/// Largest input size accepted by [`analyze`]; bounds every count and equals
/// the largest S an FYTDv002 header can carry (spec section 7).
pub const MAX_INPUT_SIZE: u32 = 1024;
/// Width of every pyramid and head feature.
pub const PYRAMID_WIDTH: u32 = 96;
/// Name of the implicit input layer.
pub const INPUT: &str = "input";

/// Activation applied after a convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// Identity.
    Linear,
    /// `x · sigmoid(x)`.
    Silu,
}

/// One layer operation. Inputs name earlier layers (or [`INPUT`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Square convolution, zero padding `kernel / 2`, with bias; stride 1 or 2.
    Conv {
        input: &'static str,
        out: u32,
        kernel: u32,
        stride: u32,
        depthwise: bool,
        act: Act,
    },
    /// Nearest-neighbour 2× upsample.
    Upsample2 { input: &'static str },
    /// Element-wise sum of two equally shaped tensors.
    Add { a: &'static str, b: &'static str },
}

/// A named layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub name: &'static str,
    pub op: Op,
}

/// Tensor shape, channels first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

/// One detection head output: class logits plus objectness, and box distances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Head {
    pub stride: u32,
    pub class_layer: &'static str,
    pub box_layer: &'static str,
}

/// Why a table or input size was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArchError {
    ClassCount(u32),
    InputSize(u32),
    DuplicateName(&'static str),
    UnknownInput {
        layer: &'static str,
        input: &'static str,
    },
    BadKernel {
        layer: &'static str,
        kernel: u32,
    },
    BadStride {
        layer: &'static str,
        stride: u32,
        size: (u32, u32),
    },
    DepthwiseWidth {
        layer: &'static str,
        input: u32,
        output: u32,
    },
    ZeroChannels(&'static str),
    AddShape {
        layer: &'static str,
        a: Shape,
        b: Shape,
    },
    HeadShape {
        layer: &'static str,
        expected: Shape,
        actual: Shape,
    },
    /// A head's stride is not one of [`HEAD_STRIDES`].
    HeadStride(u32),
    /// A head output is not a convolution without activation.
    HeadActivation(&'static str),
    /// A parameter, FLOP or shape count does not fit its integer type.
    Overflow,
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ArchError {}

/// Per-layer accounting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerReport {
    pub name: &'static str,
    pub shape: Shape,
    pub parameters: u64,
    pub flops: u64,
}

/// Whole-network accounting for one input size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub size: u32,
    pub layers: Vec<LayerReport>,
    pub parameters: u64,
    pub flops: u64,
}

impl Report {
    /// Shape of a named layer.
    pub fn shape(&self, name: &str) -> Option<Shape> {
        self.layers.iter().find(|l| l.name == name).map(|l| l.shape)
    }
}

/// The three heads of the frozen table.
pub const HEADS: [Head; 3] = [
    Head {
        stride: 8,
        class_layer: "p8.cls.out",
        box_layer: "p8.box.out",
    },
    Head {
        stride: 16,
        class_layer: "p16.cls.out",
        box_layer: "p16.box.out",
    },
    Head {
        stride: 32,
        class_layer: "p32.cls.out",
        box_layer: "p32.box.out",
    },
];

fn conv(
    name: &'static str,
    input: &'static str,
    out: u32,
    kernel: u32,
    stride: u32,
    act: Act,
) -> Layer {
    Layer {
        name,
        op: Op::Conv {
            input,
            out,
            kernel,
            stride,
            depthwise: false,
            act,
        },
    }
}

fn dw(name: &'static str, input: &'static str, width: u32, stride: u32) -> Layer {
    Layer {
        name,
        op: Op::Conv {
            input,
            out: width,
            kernel: 3,
            stride,
            depthwise: true,
            act: Act::Silu,
        },
    }
}

fn add(name: &'static str, a: &'static str, b: &'static str) -> Layer {
    Layer {
        name,
        op: Op::Add { a, b },
    }
}

fn up(name: &'static str, input: &'static str) -> Layer {
    Layer {
        name,
        op: Op::Upsample2 { input },
    }
}

/// Inverted residual block: 1×1 expand (×2, SiLU), 3×3 depthwise (SiLU),
/// 1×1 project (linear), then add the block input. Names are `<p>.expand`,
/// `<p>.dw`, `<p>.project`, `<p>` (the sum).
macro_rules! block {
    ($v:ident, $p:literal, $input:expr, $width:expr) => {
        $v.push(conv(
            concat!($p, ".expand"),
            $input,
            $width * 2,
            1,
            1,
            Act::Silu,
        ));
        $v.push(dw(
            concat!($p, ".dw"),
            concat!($p, ".expand"),
            $width * 2,
            1,
        ));
        $v.push(conv(
            concat!($p, ".project"),
            concat!($p, ".dw"),
            $width,
            1,
            1,
            Act::Linear,
        ));
        $v.push(add($p, concat!($p, ".project"), $input));
    };
}

/// Depthwise-separable unit: 3×3 depthwise (SiLU, given stride) then 1×1
/// pointwise (SiLU). Names `<p>.dw` and `<p>`.
macro_rules! sep {
    ($v:ident, $p:expr, $input:expr, $in_w:expr, $out_w:expr, $stride:expr) => {
        $v.push(dw(concat!($p, ".dw"), $input, $in_w, $stride));
        $v.push(conv($p, concat!($p, ".dw"), $out_w, 1, 1, Act::Silu));
    };
}

/// Decoupled head on one pyramid level: two separate branches of
/// depthwise-separable units, then a 1×1 linear output.
macro_rules! head {
    ($v:ident, $lvl:expr, $classes:expr) => {
        sep!(
            $v,
            concat!($lvl, ".cls.a"),
            $lvl,
            PYRAMID_WIDTH,
            PYRAMID_WIDTH,
            1
        );
        sep!(
            $v,
            concat!($lvl, ".cls.b"),
            concat!($lvl, ".cls.a"),
            PYRAMID_WIDTH,
            PYRAMID_WIDTH,
            1
        );
        $v.push(conv(
            concat!($lvl, ".cls.out"),
            concat!($lvl, ".cls.b"),
            $classes + 1,
            1,
            1,
            Act::Linear,
        ));
        sep!(
            $v,
            concat!($lvl, ".box.a"),
            $lvl,
            PYRAMID_WIDTH,
            PYRAMID_WIDTH,
            1
        );
        sep!(
            $v,
            concat!($lvl, ".box.b"),
            concat!($lvl, ".box.a"),
            PYRAMID_WIDTH,
            PYRAMID_WIDTH,
            1
        );
        $v.push(conv(
            concat!($lvl, ".box.out"),
            concat!($lvl, ".box.b"),
            4,
            1,
            1,
            Act::Linear,
        ));
    };
}

/// The frozen TD2 T1 layer table for `classes` classes (1..=[`MAX_CLASSES`]).
///
/// Class output channels are `classes + 1` (class logits, then objectness);
/// box outputs are the four `l, t, r, b` distances in stride units.
pub fn table(classes: u32) -> Result<Vec<Layer>, ArchError> {
    if classes == 0 || classes > MAX_CLASSES {
        return Err(ArchError::ClassCount(classes));
    }
    // Backbone. Stem to stride 4.
    let mut v = vec![
        conv("stem.0", INPUT, 24, 3, 2, Act::Silu),
        conv("stem.1", "stem.0", 48, 3, 2, Act::Silu),
    ];
    block!(v, "stem.2", "stem.1", 48);
    // Stride 8.
    sep!(v, "c3.down", "stem.2", 48, 96, 2);
    block!(v, "c3.0", "c3.down", 96);
    block!(v, "c3.1", "c3.0", 96);
    // Stride 16.
    sep!(v, "c4.down", "c3.1", 96, 192, 2);
    block!(v, "c4.0", "c4.down", 192);
    block!(v, "c4.1", "c4.0", 192);
    block!(v, "c4.2", "c4.1", 192);
    // Stride 32.
    sep!(v, "c5.down", "c4.2", 192, 320, 2);
    block!(v, "c5.0", "c5.down", 320);
    block!(v, "c5.1", "c5.0", 320);
    // Pyramid: lateral 1×1 to the common width.
    v.push(conv("lat5", "c5.1", PYRAMID_WIDTH, 1, 1, Act::Silu));
    v.push(conv("lat4", "c4.2", PYRAMID_WIDTH, 1, 1, Act::Silu));
    v.push(conv("lat3", "c3.1", PYRAMID_WIDTH, 1, 1, Act::Silu));
    // Top-down.
    v.push(up("td4.up", "lat5"));
    v.push(add("td4.sum", "td4.up", "lat4"));
    sep!(v, "td4", "td4.sum", PYRAMID_WIDTH, PYRAMID_WIDTH, 1);
    v.push(up("td3.up", "td4"));
    v.push(add("td3.sum", "td3.up", "lat3"));
    sep!(v, "p8", "td3.sum", PYRAMID_WIDTH, PYRAMID_WIDTH, 1);
    // Bottom-up.
    sep!(v, "bu4.down", "p8", PYRAMID_WIDTH, PYRAMID_WIDTH, 2);
    v.push(add("bu4.sum", "bu4.down", "td4"));
    sep!(v, "p16", "bu4.sum", PYRAMID_WIDTH, PYRAMID_WIDTH, 1);
    sep!(v, "bu5.down", "p16", PYRAMID_WIDTH, PYRAMID_WIDTH, 2);
    v.push(add("bu5.sum", "bu5.down", "lat5"));
    sep!(v, "p32", "bu5.sum", PYRAMID_WIDTH, PYRAMID_WIDTH, 1);
    // Decoupled heads.
    head!(v, "p8", classes);
    head!(v, "p16", classes);
    head!(v, "p32", classes);
    Ok(v)
}

/// Shapes, parameters and FLOPs of `layers` for an `size`×`size` input, with
/// the head outputs checked against `heads` and `classes`.
pub fn analyze(
    layers: &[Layer],
    size: u32,
    classes: u32,
    heads: &[Head],
) -> Result<Report, ArchError> {
    if classes == 0 || classes > MAX_CLASSES {
        return Err(ArchError::ClassCount(classes));
    }
    if size == 0 || size % 32 != 0 || size > MAX_INPUT_SIZE {
        return Err(ArchError::InputSize(size));
    }
    if let Some(head) = heads.iter().find(|h| !HEAD_STRIDES.contains(&h.stride)) {
        return Err(ArchError::HeadStride(head.stride));
    }
    let mul = |values: &[u64]| {
        values
            .iter()
            .try_fold(1u64, |acc, &v| acc.checked_mul(v))
            .ok_or(ArchError::Overflow)
    };
    let mut linear: HashMap<&'static str, bool> = HashMap::new();
    let mut shapes: HashMap<&'static str, Shape> = HashMap::new();
    shapes.insert(
        INPUT,
        Shape {
            c: INPUT_CHANNELS,
            h: size,
            w: size,
        },
    );
    let mut reports = Vec::with_capacity(layers.len());
    let (mut total_p, mut total_f) = (0u64, 0u64);
    for layer in layers {
        if shapes.contains_key(layer.name) {
            return Err(ArchError::DuplicateName(layer.name));
        }
        let get = |input: &'static str| {
            shapes.get(input).copied().ok_or(ArchError::UnknownInput {
                layer: layer.name,
                input,
            })
        };
        let (shape, parameters, flops) = match layer.op {
            Op::Conv {
                input,
                out,
                kernel,
                stride,
                depthwise,
                act: _,
            } => {
                let x = get(input)?;
                // The frozen detector uses only 1x1/3x3 dense convolutions
                // and 3x3 depthwise convolutions. Reject unsupported kinds
                // before counting a malformed table as a valid architecture.
                if !matches!(kernel, 1 | 3) || (depthwise && kernel != 3) {
                    return Err(ArchError::BadKernel {
                        layer: layer.name,
                        kernel,
                    });
                }
                if !matches!(stride, 1 | 2) || x.h % stride != 0 || x.w % stride != 0 {
                    return Err(ArchError::BadStride {
                        layer: layer.name,
                        stride,
                        size: (x.h, x.w),
                    });
                }
                if out == 0 {
                    return Err(ArchError::ZeroChannels(layer.name));
                }
                if depthwise && out != x.c {
                    return Err(ArchError::DepthwiseWidth {
                        layer: layer.name,
                        input: x.c,
                        output: out,
                    });
                }
                let per_out = mul(&[
                    u64::from(if depthwise { 1 } else { x.c }),
                    u64::from(kernel),
                    u64::from(kernel),
                ])?;
                let y = Shape {
                    c: out,
                    h: x.h / stride,
                    w: x.w / stride,
                };
                let elements = mul(&[u64::from(y.c), u64::from(y.h), u64::from(y.w)])?;
                let params = mul(&[per_out, u64::from(out)])?
                    .checked_add(u64::from(out))
                    .ok_or(ArchError::Overflow)?;
                let flops = mul(&[2, per_out, elements])?
                    .checked_add(elements)
                    .ok_or(ArchError::Overflow)?;
                (y, params, flops)
            }
            Op::Upsample2 { input } => {
                let x = get(input)?;
                (
                    Shape {
                        c: x.c,
                        h: x.h.checked_mul(2).ok_or(ArchError::Overflow)?,
                        w: x.w.checked_mul(2).ok_or(ArchError::Overflow)?,
                    },
                    0,
                    0,
                )
            }
            Op::Add { a, b } => {
                let (sa, sb) = (get(a)?, get(b)?);
                if sa != sb {
                    return Err(ArchError::AddShape {
                        layer: layer.name,
                        a: sa,
                        b: sb,
                    });
                }
                (
                    sa,
                    0,
                    mul(&[u64::from(sa.c), u64::from(sa.h), u64::from(sa.w)])?,
                )
            }
        };
        shapes.insert(layer.name, shape);
        let is_linear = matches!(
            layer.op,
            Op::Conv {
                act: Act::Linear,
                ..
            }
        );
        linear.insert(layer.name, is_linear);
        total_p = total_p.checked_add(parameters).ok_or(ArchError::Overflow)?;
        total_f = total_f.checked_add(flops).ok_or(ArchError::Overflow)?;
        reports.push(LayerReport {
            name: layer.name,
            shape,
            parameters,
            flops,
        });
    }
    for head in heads {
        let side = size / head.stride;
        for (name, c) in [(head.class_layer, classes + 1), (head.box_layer, 4)] {
            let expected = Shape {
                c,
                h: side,
                w: side,
            };
            let actual = shapes.get(name).copied().ok_or(ArchError::UnknownInput {
                layer: "heads",
                input: name,
            })?;
            if actual != expected {
                return Err(ArchError::HeadShape {
                    layer: name,
                    expected,
                    actual,
                });
            }
            // Logits and raw distances must not pass through an activation.
            if linear.get(name) != Some(&true) {
                return Err(ArchError::HeadActivation(name));
            }
        }
    }
    Ok(Report {
        size,
        layers: reports,
        parameters: total_p,
        flops: total_f,
    })
}
