// SPDX-License-Identifier: Apache-2.0
mod artifact;
mod data;
mod loss;
mod network;
pub use data::{dataset, features, Sample};
pub use loss::loss_gradient;
pub use network::{decode, sigmoid, Detection, Model};
pub const INPUTS: usize = 192;
pub const HIDDEN: usize = 64;
pub const OUTPUTS: usize = 7;
pub const PARAMETERS: usize = 12_807;
pub const W1: usize = INPUTS * HIDDEN;
pub const B1: usize = W1 + HIDDEN;
pub const W2: usize = B1 + HIDDEN * OUTPUTS;

// Deterministic 64-bit LCG; this experiment needs reproducibility, not secrecy.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}
