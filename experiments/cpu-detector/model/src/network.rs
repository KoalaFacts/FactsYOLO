// SPDX-License-Identifier: Apache-2.0
use crate::{data::features_from_rgb, Rng, B1, HIDDEN, INPUTS, PARAMETERS, W1, W2};
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class: usize,
    pub score: f32,
    pub corners: [f32; 4],
}
#[derive(Clone)]
pub struct Model {
    pub weights: Vec<f32>,
}

/// A validated, immutable artifact-backed TD1 model for repeated inference.
///
/// Unlike the public training/reference Model, this type has no mutable
/// weight access. The artifact parser checks header, checksum, parameter
/// count, bounds and finiteness once before creating this state; subsequent
/// forward calls continue to check each input and the decoded outputs.
pub struct InferenceModel {
    model: Model,
}

impl InferenceModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            model: Model::from_bytes(bytes)?,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<[f32; 7], String> {
        self.model.forward_validated(input)
    }

    pub fn detect_with_rgb(
        &self,
        rgb_at: impl FnMut(usize, usize) -> Option<[u8; 3]>,
    ) -> Result<Option<Detection>, String> {
        decode(&self.forward(&features_from_rgb(rgb_at)?)?)
    }
}
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
pub fn decode(out: &[f32; 7]) -> Result<Option<Detection>, String> {
    if out.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite network output".into());
    }
    let max = out[..3].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = [
        (out[0] - max).exp(),
        (out[1] - max).exp(),
        (out[2] - max).exp(),
    ];
    let mut winner = 0;
    for i in 1..3 {
        if exp[i] > exp[winner] {
            winner = i;
        }
    }
    let score = exp[winner] / exp.iter().sum::<f32>();
    if winner == 0 || score < 0.5 {
        return Ok(None);
    }
    let [cx, cy, w, h] = [
        sigmoid(out[3]),
        sigmoid(out[4]),
        sigmoid(out[5]),
        sigmoid(out[6]),
    ];
    let corners = [
        ((cx - w / 2.0) * 64.0).clamp(0.0, 64.0),
        ((cy - h / 2.0) * 64.0).clamp(0.0, 64.0),
        ((cx + w / 2.0) * 64.0).clamp(0.0, 64.0),
        ((cy + h / 2.0) * 64.0).clamp(0.0, 64.0),
    ];
    if corners[2] <= corners[0] || corners[3] <= corners[1] {
        return Ok(None);
    }
    Ok(Some(Detection {
        class: winner - 1,
        score,
        corners,
    }))
}
impl Model {
    pub fn initialized(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weights = vec![0.0; PARAMETERS];
        let a = (6.0_f32 / (INPUTS + HIDDEN) as f32).sqrt();
        let b = (6.0_f32 / (HIDDEN + 7) as f32).sqrt();
        for w in &mut weights[..W1] {
            *w = (rng.unit() * 2.0 - 1.0) * a;
        }
        for w in &mut weights[B1..W2] {
            *w = (rng.unit() * 2.0 - 1.0) * b;
        }
        Self { weights }
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.weights.len() != PARAMETERS
            || self
                .weights
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 10_000.0)
        {
            return Err("invalid TD1 parameters".into());
        }
        Ok(())
    }
    pub fn forward(&self, input: &[f32]) -> Result<[f32; 7], String> {
        // Training/reference weights are externally mutable; never skip this.
        self.validate()?;
        self.forward_validated(input)
    }

    // Only an immutable, artifact-validated model may call this without first
    // checking weights. The feature validation and network math are shared
    // with the mutable reference to preserve bit-for-bit inference results.
    fn forward_validated(&self, input: &[f32]) -> Result<[f32; 7], String> {
        if input.len() != INPUTS
            || input
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("invalid TD1 features".into());
        }
        let mut hidden = [0.0; HIDDEN];
        for (j, h) in hidden.iter_mut().enumerate() {
            *h = self.weights[W1 + j];
            for (i, x) in input.iter().enumerate() {
                *h += x * self.weights[i * HIDDEN + j];
            }
            *h = h.max(0.0);
        }
        let mut out = [0.0; 7];
        for (k, y) in out.iter_mut().enumerate() {
            *y = self.weights[W2 + k];
            for (j, h) in hidden.iter().enumerate() {
                *y += h * self.weights[B1 + j * 7 + k];
            }
        }
        Ok(out)
    }
    pub fn detect(&self, pixels: &[u8]) -> Result<Option<Detection>, String> {
        if pixels.len() != 64 * 64 * 3 {
            return Err("TD1 requires exactly 64x64 RGB bytes".into());
        }
        // Reuse the borrowed preprocessing path: the previous compact path
        // allocated a 192-float Vec before every portable detection.
        self.detect_with_rgb(|x, y| {
            let at = (y * 64 + x) * 3;
            Some([pixels[at], pixels[at + 1], pixels[at + 2]])
        })
    }

    /// Execute the same TD1 preprocessing/forward/decode without copying a
    /// checked 64x64 image into an intermediate compact RGB allocation.
    /// The accessor must yield straight RGB for every pixel, regardless of
    /// the caller's source format or row stride.
    pub fn detect_with_rgb(
        &self,
        rgb_at: impl FnMut(usize, usize) -> Option<[u8; 3]>,
    ) -> Result<Option<Detection>, String> {
        decode(&self.forward(&features_from_rgb(rgb_at)?)?)
    }
}
