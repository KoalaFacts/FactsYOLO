// SPDX-License-Identifier: Apache-2.0
use super::types::*;
use crate::error::{Error, FLAGS, INVALID, VERSION};

// Private whitelist: only repr(C) records starting with the three u32 fields.
// Do not implement this for arbitrary Rust types or fixed array elements.
pub(crate) trait Descriptor {}
impl Descriptor for FyAbiInfo {}
impl Descriptor for FyDiagnostic {}
impl Descriptor for FyModelOptions {}
impl Descriptor for FySessionOptions {}
impl Descriptor for FyImage {}
impl Descriptor for FyDetectOptions {}
impl Descriptor for FySessionInfo {}
impl Descriptor for FyResultInfo {}

/// Validate a prefix without reading uninitialized outputs or forming a reference.
///
/// # Safety
/// Except for null/misaligned pointers, the caller provides an initialized,
/// readable two-u32 prefix. A sufficiently large advertised record must really
/// exist, with initialized header fields. Storage remains live during the call.
pub(crate) unsafe fn validate<T: Descriptor>(raw: *const T) -> Result<(), Error> {
    if raw.is_null() || !raw.is_aligned() {
        return Err(INVALID);
    }
    let header = raw.cast::<u32>();
    // SAFETY: only the first initialized word is read. In particular, no &T
    // or full T value is formed from a potentially prefix-only allocation.
    let advertised = unsafe { header.read() } as usize;
    if advertised < size_of::<T>() {
        return Err(INVALID);
    }
    // SAFETY: the preceding size check and the caller's allocation contract
    // permit the complete header, but not reads of arbitrary output fields.
    if unsafe { header.add(1).read() } != FY_STRUCT_VERSION {
        return Err(VERSION);
    }
    if unsafe { header.add(2).read() } != 0 {
        return Err(FLAGS);
    }
    Ok(())
}
