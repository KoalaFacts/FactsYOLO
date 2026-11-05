// SPDX-License-Identifier: Apache-2.0
//! The predeclared `fyp1` parameter table and its seeded initialization.
use crate::SplitMix64;

pub const SEED: u64 = 20_261_105;
pub const PARAMETER_COUNT: usize = 22_021;

/// One parameter tensor in the flat F32 parameter vector.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: &'static str,
    pub offset: usize,
    pub len: usize,
    /// Input fan of a kernel; 0 for BN and bias tensors.
    pub fan_in: usize,
}

/// `(name, len, fan_in)` in layout order; kernels are `[c_out, c_in/groups, k, k]`.
const TABLE: [(&str, usize, usize); 16] = [
    ("stem.w", 16 * 3 * 9, 3 * 9),
    ("stem.bn.g", 16, 0),
    ("stem.bn.b", 16, 0),
    ("dw.w", 16 * 9, 9),
    ("dw.bn.g", 16, 0),
    ("dw.bn.b", 16, 0),
    ("pw.w", 32 * 16, 16),
    ("pw.bn.g", 32, 0),
    ("pw.bn.b", 32, 0),
    ("down.w", 64 * 32 * 9, 32 * 9),
    ("down.bn.g", 64, 0),
    ("down.bn.b", 64, 0),
    ("lateral.w", 32 * 64, 64),
    ("lateral.b", 32, 0),
    ("head.w", 5 * 32, 32),
    ("head.b", 5, 0),
];

pub fn tensors() -> Vec<TensorSpec> {
    let mut offset = 0;
    TABLE
        .iter()
        .map(|&(name, len, fan_in)| {
            let t = TensorSpec {
                name,
                offset,
                len,
                fan_in,
            };
            offset += len;
            t
        })
        .collect()
}

/// Offset of a named tensor.
pub fn offset(name: &str) -> usize {
    tensors()
        .into_iter()
        .find(|t| t.name == name)
        .map(|t| t.offset)
        .unwrap_or_else(|| panic!("unknown tensor {name}"))
}

/// Kaiming-uniform kernels (bound √(6/fan_in)), BN γ = 1, β = 0, biases 0.
pub fn initial_parameters() -> Vec<f32> {
    let mut p = Vec::with_capacity(PARAMETER_COUNT);
    for (i, t) in tensors().iter().enumerate() {
        if t.name.ends_with(".g") {
            p.extend(std::iter::repeat_n(1.0f32, t.len));
        } else if t.fan_in == 0 {
            p.extend(std::iter::repeat_n(0.0f32, t.len));
        } else {
            let bound = (6.0f32 / t.fan_in as f32).sqrt();
            let mut r = SplitMix64::new(SEED ^ ((i as u64 + 1) << 48));
            p.extend((0..t.len).map(|_| (2.0 * r.unit_f32() - 1.0) * bound));
        }
    }
    p
}

/// Little-endian bytes of a parameter vector, for fingerprinting.
pub fn parameter_bytes(p: &[f32]) -> Vec<u8> {
    p.iter().flat_map(|v| v.to_le_bytes()).collect()
}
