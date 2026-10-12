// SPDX-License-Identifier: Apache-2.0
//! Original-image result normalization; no model-specific suppression.
use crate::{Error, ffi::types::FyDetection};

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    pub score: f32,
    pub class_id: u32,
    pub source_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageTransform {
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

#[derive(Debug)]
pub struct Finalized {
    pub detections: Vec<FyDetection>,
    pub truncated: bool,
}

/// Normalize post-suppression candidates. RED checkpoint, not implemented yet.
pub fn finalize(
    _candidates_after_suppression: &[Candidate],
    _transform: ImageTransform,
    _source: (u32, u32),
    _class_count: u32,
    _threshold: f32,
    _max_detections: usize,
) -> Result<Finalized, Error> {
    Err(Error {
        status: crate::ffi::types::FY_INTERNAL_ERROR,
        message: "result finalization is not implemented",
    })
}
