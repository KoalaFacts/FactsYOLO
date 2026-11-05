// SPDX-License-Identifier: Apache-2.0
//! Original synthetic data: one solid rectangle on uniform noise.
use crate::SplitMix64;

pub const SIDE: usize = 64;
pub const IMAGE: usize = 3 * SIDE * SIDE;
pub const MIN_BOX: usize = 8;
pub const MAX_BOX: usize = 40;

/// A CHW image in [0, 1] with its half-open pixel box `[x0, y0, x1, y1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub pixels: Vec<f32>,
    pub bbox: [usize; 4],
    pub colour: [f32; 3],
}

fn sample(seed: u64, index: u64, k: u64) -> Sample {
    let mut r = SplitMix64::new(
        seed ^ index.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ k.wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    let mut pixels: Vec<f32> = (0..IMAGE).map(|_| r.unit_f32()).collect();
    let span = (MAX_BOX - MIN_BOX + 1) as u64;
    let w = MIN_BOX + r.below(span) as usize;
    let h = MIN_BOX + r.below(span) as usize;
    let x0 = r.below((SIDE - w + 1) as u64) as usize;
    let y0 = r.below((SIDE - h + 1) as u64) as usize;
    let colour = [r.unit_f32(), r.unit_f32(), r.unit_f32()];
    for (c, value) in colour.iter().enumerate() {
        for y in y0..y0 + h {
            let row = c * SIDE * SIDE + y * SIDE;
            pixels[row + x0..row + x0 + w].fill(*value);
        }
    }
    Sample {
        pixels,
        bbox: [x0, y0, x0 + w, y0 + h],
        colour,
    }
}

/// Batch `index` of `n` samples; identical for identical arguments.
pub fn batch(seed: u64, index: u64, n: usize) -> Vec<Sample> {
    (0..n as u64).map(|k| sample(seed, index, k)).collect()
}

/// Concatenated NCHW pixels of a batch.
pub fn pixels(samples: &[Sample]) -> Vec<f32> {
    samples
        .iter()
        .flat_map(|s| s.pixels.iter().copied())
        .collect()
}
