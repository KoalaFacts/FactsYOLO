// SPDX-License-Identifier: Apache-2.0
use super::{descriptors, handles::entry, types::*};
use crate::{Error, error::BUILD_ID};
use std::ptr::addr_of_mut;

/// Query experimental ABI 1.0. This does not claim a trained detector exists.
///
/// # Safety
/// Non-null aligned descriptors must have a readable initialized two-u32
/// prefix. If struct_size advertises a full record, that allocation must exist
/// and its input header fields must be initialized. Diagnostic data/capacity
/// are also inputs; required_bytes and the ABI payload may be uninitialized.
/// All accepted outputs/buffers are writable, live, properly aligned, mutually
/// disjoint, and externally synchronized for this call. Invalid foreign memory
/// is not made safe by these checks. No input storage is retained after return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_abi_query(out: *mut FyAbiInfo, diag: *mut FyDiagnostic) -> i32 {
    // All exported operations share diagnostic precedence and unwind handling.
    unsafe { entry(diag, || query(out)) }
}

unsafe fn query(out: *mut FyAbiInfo) -> Result<(), Error> {
    // SAFETY: only the verified prefix is read; no full output value is read.
    unsafe { descriptors::validate(out)? };
    let id = option_env!("FACTSYOLO_BUILD_ID").unwrap_or("local-unversioned");
    if id.is_empty() || id.len() > 63 || !id.is_ascii() || id.as_bytes().contains(&0) {
        return Err(BUILD_ID);
    }
    let mut build_id = [0_u8; 64];
    build_id[..id.len()].copy_from_slice(id.as_bytes());
    // SAFETY: all validation is complete; write payload fields individually,
    // leaving the caller's header, padding and optional extension tail intact.
    unsafe {
        addr_of_mut!((*out).abi_major).write(FY_ABI_MAJOR);
        addr_of_mut!((*out).abi_minor).write(FY_ABI_MINOR);
        addr_of_mut!((*out).build_id).write(build_id);
    }
    Ok(())
}
