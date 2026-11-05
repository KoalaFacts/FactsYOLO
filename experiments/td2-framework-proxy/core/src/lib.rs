// SPDX-License-Identifier: Apache-2.0
//! Framework-independent core of the stage B proxy (`fyp1`).
pub mod data;
pub mod loss;
pub mod net;
pub mod reference;
pub mod sample_plan;
pub mod sgd;

/// SplitMix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number
/// generators", OOPSLA 2014). Reproducibility only, not secrecy.
pub struct SplitMix64(u64);
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1) with 24 bits, exactly representable as F32.
    pub fn unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform integer in `0..bound` (bound > 0); modulo bias is irrelevant here.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}
