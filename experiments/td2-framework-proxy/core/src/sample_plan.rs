// SPDX-License-Identifier: Apache-2.0
//! The predeclared finite-difference sample: up to 8 distinct parameters per tensor.
use crate::net::{tensors, SEED};
use crate::SplitMix64;

pub const PER_TENSOR: usize = 8;

pub fn gradient_samples() -> Vec<usize> {
    let mut plan = Vec::new();
    for (i, t) in tensors().iter().enumerate() {
        let want = t.len.min(PER_TENSOR);
        let mut r = SplitMix64::new(SEED ^ 0x5eed_0000 ^ i as u64);
        let mut picked: Vec<usize> = Vec::with_capacity(want);
        while picked.len() < want {
            let j = r.below(t.len as u64) as usize;
            if !picked.contains(&j) {
                picked.push(j);
            }
        }
        plan.extend(picked.into_iter().map(|j| t.offset + j));
    }
    plan
}
