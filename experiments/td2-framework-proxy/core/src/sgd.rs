// SPDX-License-Identifier: Apache-2.0
//! Shared optimizer: SGD with momentum, applied to every framework's gradients.
pub const LEARNING_RATE: f32 = 0.05;
pub const MOMENTUM: f32 = 0.9;

pub struct Sgd {
    velocity: Vec<f32>,
}

impl Sgd {
    pub fn new(parameters: usize) -> Self {
        Self {
            velocity: vec![0.0; parameters],
        }
    }
    /// `v = μ·v + g; p = p − η·v`.
    pub fn step(&mut self, parameters: &mut [f32], gradient: &[f32]) {
        assert_eq!(parameters.len(), self.velocity.len(), "parameter count");
        assert_eq!(gradient.len(), self.velocity.len(), "gradient count");
        for ((p, v), g) in parameters.iter_mut().zip(&mut self.velocity).zip(gradient) {
            *v = MOMENTUM * *v + g;
            *p -= LEARNING_RATE * *v;
        }
    }
}
