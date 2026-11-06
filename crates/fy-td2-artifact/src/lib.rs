// SPDX-License-Identifier: Apache-2.0
//! FYTDv002 artifact container (TD2 stage E1a).
//!
//! The byte layout is fixed in section 7 of
//! `docs/superpowers/specs/2026-11-05-td2-general-detector-design.md`. This
//! crate validates the container only: fixed header, layer-table records,
//! contiguous F32 parameters and the SHA-256 trailer. The meaning of each layer
//! kind and its kind-specific fields is defined in stage D2b and checked by the
//! backend that executes the table, not here.
//!
//! The crate performs no I/O, has no networking and stores nothing; callers pass
//! a borrowed byte slice and receive owned, validated data or an error.
#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use sha2::{Digest, Sha256};

/// Eight-byte magic at offset 0.
pub const MAGIC: [u8; 8] = *b"FYTDv002";
/// The only accepted format version.
pub const FORMAT_VERSION: u32 = 2;
/// Task kind of T1 general detection; T2–T4 define their own in their stages.
pub const TASK_T1_DETECTION: u32 = 1;
/// Fixed header length in bytes.
pub const HEADER_BYTES: usize = 64;
/// Length of one layer-table record in bytes.
pub const LAYER_RECORD_BYTES: usize = 64;
/// Alignment of the parameter region from the start of the artifact. The
/// header and every record are 64 bytes, so the region, which starts right
/// after the table, is always 64-byte aligned and no padding exists.
pub const PARAMETER_ALIGNMENT: usize = 64;
/// Length of the SHA-256 trailer.
pub const DIGEST_BYTES: usize = 32;
/// Upper bound on layer records, independent of caller limits.
pub const MAX_LAYERS: u32 = 1024;
/// Upper bound on the class count of a T1 model (spec section 6).
pub const MAX_T1_CLASSES: u32 = 120;
/// Smallest and largest accepted input size S (a multiple of 32).
pub const MIN_INPUT_SIZE: u32 = 32;
pub const MAX_INPUT_SIZE: u32 = 1024;
/// Number of kind-specific `u32` fields in a layer record (bytes 8..48).
pub const LAYER_FIELDS: usize = 10;

/// Why an artifact was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Longer than the caller's `max_artifact_bytes`.
    TooLarge,
    /// The caller's parser-step budget is smaller than the work required.
    StepLimit,
    /// Shorter than the fixed parts or the declared total length.
    Truncated,
    BadMagic,
    UnsupportedVersion,
    UnsupportedTask,
    BadInputSize,
    BadClassCount,
    BadLayerCount,
    /// Declared total length differs from the slice length or the layout sum.
    LengthMismatch,
    /// A reserved byte is not zero.
    ReservedNonZero,
    /// A layer record has kind 0.
    BadLayerKind,
    /// Layer parameter ranges are not contiguous, in order, covering every parameter.
    ParameterLayout,
    NonFiniteParameter,
    DigestMismatch,
    /// A fallible allocation failed.
    OutOfMemory,
}

/// One validated layer-table record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerRecord {
    /// Non-zero layer kind; meaning defined in stage D2b.
    pub kind: u32,
    /// Activation code; meaning defined in stage D2b.
    pub activation: u32,
    /// Kind-specific fields (shapes, strides, padding, groups, inputs).
    pub fields: [u32; LAYER_FIELDS],
    /// First parameter of this layer, in F32 elements from the region start.
    pub parameter_offset: u64,
    /// Number of F32 parameters of this layer.
    pub parameter_count: u64,
}

/// Fixed header fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub task_kind: u32,
    pub input_size: u32,
    pub class_count: u32,
    pub layer_count: u32,
    pub parameter_count: u64,
    pub total_bytes: u64,
}

/// A fully validated FYTDv002 artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct Artifact {
    pub header: Header,
    pub layers: Vec<LayerRecord>,
    pub parameters: Vec<f32>,
    /// SHA-256 of every byte before the trailer; equals the trailer.
    pub model_id: [u8; 32],
}

/// Caller-supplied limits, mirroring the ABI's `max_artifact_bytes` and
/// `max_parser_steps` options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_artifact_bytes: usize,
    /// One step per input byte; parsing needs at least `bytes.len()` steps.
    pub max_parser_steps: usize,
}

/// Byte offset of the parameter region for `layer_count` records.
pub fn parameter_region_offset(layer_count: u32) -> usize {
    HEADER_BYTES + layer_count as usize * LAYER_RECORD_BYTES
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

/// Validate `bytes` as an FYTDv002 artifact.
///
/// Checks run cheapest first: limits, header fields and the exact layout
/// length before any allocation or hashing, so a hostile header cannot make
/// the parser allocate or hash more than the slice it was given.
pub fn parse(bytes: &[u8], limits: Limits) -> Result<Artifact, Error> {
    if bytes.len() > limits.max_artifact_bytes {
        return Err(Error::TooLarge);
    }
    if limits.max_parser_steps < bytes.len() {
        return Err(Error::StepLimit);
    }
    if bytes.len() < HEADER_BYTES + DIGEST_BYTES {
        return Err(Error::Truncated);
    }
    if bytes[..8] != MAGIC {
        return Err(Error::BadMagic);
    }
    if u32_at(bytes, 8) != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion);
    }
    let header = Header {
        task_kind: u32_at(bytes, 12),
        input_size: u32_at(bytes, 16),
        class_count: u32_at(bytes, 20),
        layer_count: u32_at(bytes, 24),
        parameter_count: u64_at(bytes, 32),
        total_bytes: u64_at(bytes, 40),
    };
    if header.task_kind != TASK_T1_DETECTION {
        return Err(Error::UnsupportedTask);
    }
    if !(MIN_INPUT_SIZE..=MAX_INPUT_SIZE).contains(&header.input_size)
        || header.input_size % 32 != 0
    {
        return Err(Error::BadInputSize);
    }
    if !(1..=MAX_T1_CLASSES).contains(&header.class_count) {
        return Err(Error::BadClassCount);
    }
    if !(1..=MAX_LAYERS).contains(&header.layer_count) {
        return Err(Error::BadLayerCount);
    }
    if u32_at(bytes, 28) != 0 || bytes[48..HEADER_BYTES].iter().any(|&b| b != 0) {
        return Err(Error::ReservedNonZero);
    }
    if header.total_bytes != bytes.len() as u64 {
        return Err(Error::LengthMismatch);
    }
    let region = parameter_region_offset(header.layer_count);
    let expected = header
        .parameter_count
        .checked_mul(4)
        .and_then(|p| p.checked_add((region + DIGEST_BYTES) as u64));
    if expected != Some(bytes.len() as u64) {
        return Err(Error::LengthMismatch);
    }
    // From here every offset is inside `bytes`.
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(header.layer_count as usize)
        .map_err(|_| Error::OutOfMemory)?;
    let mut next = 0u64;
    for at in (HEADER_BYTES..region).step_by(LAYER_RECORD_BYTES) {
        let kind = u32_at(bytes, at);
        if kind == 0 {
            return Err(Error::BadLayerKind);
        }
        let mut fields = [0; LAYER_FIELDS];
        for (i, f) in fields.iter_mut().enumerate() {
            *f = u32_at(bytes, at + 8 + 4 * i);
        }
        let record = LayerRecord {
            kind,
            activation: u32_at(bytes, at + 4),
            fields,
            parameter_offset: u64_at(bytes, at + 48),
            parameter_count: u64_at(bytes, at + 56),
        };
        if record.parameter_offset != next {
            return Err(Error::ParameterLayout);
        }
        next = next
            .checked_add(record.parameter_count)
            .ok_or(Error::ParameterLayout)?;
        layers.push(record);
    }
    if next != header.parameter_count {
        return Err(Error::ParameterLayout);
    }
    let digest_at = bytes.len() - DIGEST_BYTES;
    let mut parameters = Vec::new();
    parameters
        .try_reserve_exact(header.parameter_count as usize)
        .map_err(|_| Error::OutOfMemory)?;
    for chunk in bytes[region..digest_at].chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(Error::NonFiniteParameter);
        }
        parameters.push(value);
    }
    let digest: [u8; 32] = Sha256::digest(&bytes[..digest_at]).into();
    if digest[..] != bytes[digest_at..] {
        return Err(Error::DigestMismatch);
    }
    Ok(Artifact {
        header,
        layers,
        parameters,
        model_id: digest,
    })
}

/// Encode an artifact from its parts (used by exporters and tests).
///
/// Layer `parameter_offset`/`parameter_count` are written as given, so a
/// caller can also construct deliberately invalid artifacts for tests; the
/// header's `layer_count`, `parameter_count` and `total_bytes` are derived.
pub fn encode(
    task_kind: u32,
    input_size: u32,
    class_count: u32,
    layers: &[LayerRecord],
    parameters: &[f32],
) -> Vec<u8> {
    let layer_count = layers.len() as u32;
    let region = parameter_region_offset(layer_count);
    let total = region + parameters.len() * 4 + DIGEST_BYTES;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&task_kind.to_le_bytes());
    out.extend_from_slice(&input_size.to_le_bytes());
    out.extend_from_slice(&class_count.to_le_bytes());
    out.extend_from_slice(&layer_count.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(parameters.len() as u64).to_le_bytes());
    out.extend_from_slice(&(total as u64).to_le_bytes());
    out.resize(HEADER_BYTES, 0);
    for layer in layers {
        out.extend_from_slice(&layer.kind.to_le_bytes());
        out.extend_from_slice(&layer.activation.to_le_bytes());
        for f in layer.fields {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&layer.parameter_offset.to_le_bytes());
        out.extend_from_slice(&layer.parameter_count.to_le_bytes());
    }
    for p in parameters {
        out.extend_from_slice(&p.to_le_bytes());
    }
    let digest: [u8; 32] = Sha256::digest(&out).into();
    out.extend_from_slice(&digest);
    out
}
