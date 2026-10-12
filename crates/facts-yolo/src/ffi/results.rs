// SPDX-License-Identifier: Apache-2.0
//! Independently owned results and bounded host copies.
use super::{
    descriptors,
    handles::{FySession, SHORT, SessionHandle, empty_slot, entry, release},
    types::*,
};
use crate::{Error, error::INVALID, image::ImageView, result::owned::OwnedResult};
use std::ptr::addr_of_mut;

/// Opaque owning result marker. Never allocate or dereference this marker.
#[repr(C)]
pub struct FyResult {
    _private: [u8; 0],
}

/// Synchronously borrow host pixels and produce an independently owned result.
///
/// # Safety
/// session is a live correctly typed handle from this library, not concurrently
/// released. Inputs provide readable initialized prefixes and fully initialized
/// accepted records; pixel storage is one live initialized immutable allocation
/// covering the declared range until return. out is an initialized writable
/// empty pointer slot. All outputs, diagnostics, records and pixel storage obey
/// the documented alignment, disjointness and synchronization requirements.
/// Validation does not make forged pointers, false lengths or stale handles safe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_detect(
    session: *mut FySession,
    image: *const FyImage,
    options: *const FyDetectOptions,
    out: *mut *mut FyResult,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            empty_slot(out)?;
            descriptors::validate(options)?;
            descriptors::validate(image)?;
            let raw = session.cast::<SessionHandle>();
            if raw.is_null() || !raw.is_aligned() {
                return Err(INVALID);
            }
            // Only shared outer access; no exclusive reference is formed here.
            let session = &(*raw).session;
            let options = options.read();
            session.validate_detect(options)?;
            let image = image.read();
            let view = ImageView::from_ffi(&image, session.image_limits())?;
            let result = Box::new(session.run(view, options)?);
            // No host pointer, session borrow or fallible work survives transfer.
            out.write(Box::into_raw(result).cast());
            Ok(())
        })
    }
}

/// # Safety
/// A non-null aligned handle points to a live OwnedResult from this library for
/// the entire returned borrow; no concurrent release is permitted.
unsafe fn borrowed<'a>(result: *const FyResult) -> Result<&'a OwnedResult, Error> {
    let raw = result.cast::<OwnedResult>();
    if raw.is_null() || !raw.is_aligned() {
        return Err(INVALID);
    }
    // SAFETY: callers supply the correct live resource, not arbitrary storage.
    Ok(unsafe { &*raw })
}

/// Describe immutable result metadata without executing a model.
///
/// # Safety
/// result is a live correctly typed handle not concurrently released. out has
/// an initialized readable descriptor header and full writable accepted record;
/// payload may be uninitialized. All mutable outputs/diagnostics are disjoint,
/// live, aligned and synchronized. Header, padding and extension tail survive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_describe(
    result: *const FyResult,
    out: *mut FyResultInfo,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            descriptors::validate(out)?;
            let info = &borrowed(result)?.info;
            addr_of_mut!((*out).source_width).write(info.source_width);
            addr_of_mut!((*out).source_height).write(info.source_height);
            addr_of_mut!((*out).count).write(info.count);
            addr_of_mut!((*out).result_flags).write(info.result_flags);
            addr_of_mut!((*out).backend_id).write(info.backend_id);
            addr_of_mut!((*out).device_kind).write(info.device_kind);
            addr_of_mut!((*out).device_index).write(info.device_index);
            addr_of_mut!((*out).precision).write(info.precision);
            addr_of_mut!((*out).execution_flags).write(info.execution_flags);
            addr_of_mut!((*out).model_id).write(info.model_id);
            Ok(())
        })
    }
}

/// Query/copy an immutable completed result. Short buffers receive no records.
///
/// # Safety
/// result is a live correctly typed handle. required is writable aligned usize
/// storage. A used destination covers capacity aligned writable records. All
/// outputs/diagnostics are live, disjoint and synchronized; none overlaps result
/// storage. No concurrent release is permitted. required need not be initialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_copy(
    result: *const FyResult,
    dst: *mut FyDetection,
    capacity: usize,
    required: *mut usize,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            if required.is_null() || !required.is_aligned() {
                return Err(INVALID);
            }
            let bytes = capacity
                .checked_mul(size_of::<FyDetection>())
                .ok_or(INVALID)?;
            if bytes > isize::MAX as usize
                || (dst.is_null() && capacity != 0)
                || (!dst.is_null() && !dst.is_aligned())
                || (dst as usize).checked_add(bytes).is_none()
            {
                return Err(INVALID);
            }
            let result = borrowed(result)?;
            let count = result.detections.len();
            if dst.is_null() && capacity == 0 {
                required.write(count);
                return Ok(());
            }
            if capacity < count {
                required.write(count);
                return Err(SHORT);
            }
            if count != 0 {
                std::ptr::copy_nonoverlapping(result.detections.as_ptr(), dst, count);
            }
            required.write(count);
            Ok(())
        })
    }
}

/// Free this library's result and clear its owning pointer slot.
///
/// # Safety
/// slot is live aligned initialized writable pointer storage. Its non-null value
/// is a correct live result handle from this library with no concurrent users.
/// Diagnostics and slot are disjoint. Copied stale handles are not release tokens.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_result_release(
    slot: *mut *mut FyResult,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe { entry(diag, || release::<OwnedResult, FyResult>(slot)) }
}
