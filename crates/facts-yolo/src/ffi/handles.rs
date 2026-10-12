// SPDX-License-Identifier: Apache-2.0
//! Owned resource entry points. Raw callers must obey allocation and lifetime rules.
use super::{descriptors, diagnostics::Diagnostic, types::*};
use crate::{Error, error::INVALID, model::Model, session::Session};
use std::ptr::{addr_of_mut, null_mut};
use std::sync::Arc;

/// Opaque ABI marker, never a by-value record or a Rust handle layout.
#[repr(C)]
pub struct FyModel {
    _private: [u8; 0],
}

/// Opaque ABI marker, never construct or dereference it as a concrete Rust value.
#[repr(C)]
pub struct FySession {
    _private: [u8; 0],
}

struct ModelHandle {
    model: Arc<Model>,
    #[cfg(feature = "contract-fixtures")]
    _live: crate::fixtures::Live,
}

struct SessionHandle {
    session: Session,
}

const ABI: Error = Error {
    status: FY_UNSUPPORTED_ABI,
    message: "unsupported required ABI major",
};
const MODEL: Error = Error {
    status: FY_UNSUPPORTED_MODEL,
    message: "no supported model artifact format is installed",
};
const LIMIT: Error = Error {
    status: FY_LIMIT_EXCEEDED,
    message: "model artifact byte limit exceeded",
};
const SHORT: Error = Error {
    status: FY_BUFFER_TOO_SMALL,
    message: "destination capacity is too small",
};
const INTERNAL: Error = Error {
    status: FY_INTERNAL_ERROR,
    message: "internal native operation failed",
};

#[cfg(not(feature = "contract-fixtures"))]
const CAPABILITIES: &[FyCapability] = &[];
#[cfg(feature = "contract-fixtures")]
const CAPABILITIES: &[FyCapability] = &[FyCapability {
    backend_id: crate::fixtures::FY_BACKEND_TEST_FIXTURE,
    device_kind: FY_DEVICE_CPU,
    device_index: 0,
    flags: FY_CAP_COMPILED | FY_CAP_AVAILABLE | FY_CAP_TEST_ONLY,
    precision_mask: 1,
}];

fn require_abi(major: u32) -> Result<(), Error> {
    if major == FY_ABI_MAJOR {
        Ok(())
    } else {
        Err(ABI)
    }
}

/// # Safety
/// Non-null aligned storage really contains an initialized, writable pointer slot.
unsafe fn slot_value<T>(slot: *mut *mut T) -> Result<*mut T, Error> {
    if slot.is_null() || !slot.is_aligned() {
        return Err(INVALID);
    }
    // SAFETY: caller supplies an initialized pointer slot; T is not accessed.
    Ok(unsafe { slot.read() })
}

/// # Safety
/// The slot satisfies slot_value's contract. A rejected non-null value is not read.
unsafe fn empty_slot<T>(slot: *mut *mut T) -> Result<(), Error> {
    if unsafe { slot_value(slot)? }.is_null() {
        Ok(())
    } else {
        Err(INVALID)
    }
}

/// # Safety
/// T is the private allocation type corresponding to opaque O, and a non-null
/// value is a unique live Box<T> from this library with no concurrent users.
unsafe fn release<T, O>(slot: *mut *mut O) -> Result<(), Error> {
    let raw = unsafe { slot_value(slot)? }.cast::<T>();
    if raw.is_null() {
        return Ok(());
    }
    if !raw.is_aligned() {
        return Err(INVALID);
    }
    // SAFETY: the caller transfers the one live allocation back to its owner.
    // Clear the slot before destruction. Our resource destructors do not panic.
    unsafe {
        let owned = Box::from_raw(raw);
        slot.write(null_mut());
        drop(owned);
    }
    Ok(())
}

/// # Safety
/// Descriptor prefixes and accepted diagnostic storage follow the C contract.
/// operation must leave outputs unchanged on failure, except documented counts.
unsafe fn entry(raw_diag: *mut FyDiagnostic, operation: impl FnOnce() -> Result<(), Error>) -> i32 {
    // SAFETY: validate diagnostics before reading or changing operation outputs.
    let diag = match unsafe { Diagnostic::validate(raw_diag) } {
        Ok(diag) => diag,
        Err(error) => return error.status,
    };
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)));
            if let Err(secondary) = cleanup {
                // Only an exceptional second panic payload is retained.
                // Abort and invalid foreign memory are not recoverable promises.
                std::mem::forget(secondary);
            }
            Err(INTERNAL)
        }
    };
    match result {
        Ok(()) => {
            unsafe { diag.write("") };
            FY_OK
        }
        Err(error) => {
            unsafe { diag.write(error.message) };
            error.status
        }
    }
}

/// Validate a model-open request. Task 3 installs no artifact parser.
///
/// # Safety
/// out is a live aligned initialized pointer slot, initially null. Descriptor
/// prefixes are readable; accepted input records are fully initialized. bytes
/// with accepted nonzero length denotes a live readable allocation. All mutable
/// outputs/diagnostics are writable, disjoint and synchronized. Storage is not
/// retained. False pointer provenance/lengths are not made safe by validation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_model_open(
    required_abi_major: u32,
    bytes: *const u8,
    length: usize,
    options: *const FyModelOptions,
    out: *mut *mut FyModel,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            empty_slot(out)?;
            require_abi(required_abi_major)?;
            descriptors::validate(options)?;
            // Only input records are read whole, after prefix validation.
            let options = options.read();
            if options.max_artifact_bytes == 0
                || options.max_parser_steps == 0
                || length == 0
                || bytes.is_null()
                || length > isize::MAX as usize
                || (bytes as usize).checked_add(length).is_none()
            {
                return Err(INVALID);
            }
            if length > options.max_artifact_bytes {
                return Err(LIMIT);
            }
            // Do not parse, copy, execute, download, or accept fixture bytes.
            Err(MODEL)
        })
    }
}

/// Release the owning model handle; sessions keep their own strong model state.
///
/// # Safety
/// slot is a valid initialized writable pointer slot. Its non-null value is the
/// correct live model handle from this library, with no overlapping use/release.
/// Diagnostics and slot are disjoint. Copied stale handles are never safe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_model_release(slot: *mut *mut FyModel, diag: *mut FyDiagnostic) -> i32 {
    unsafe { entry(diag, || release::<ModelHandle, FyModel>(slot)) }
}

/// Create an owned session with copied settings and retained immutable model.
///
/// # Safety
/// model is a correct live handle from this loaded library for the entire call.
/// out is a writable initialized empty slot. options supplies a readable prefix
/// and fully initialized accepted input fields. Outputs, diagnostics and input
/// storage do not overlap; no concurrent release or mutation is permitted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_session_create(
    required_abi_major: u32,
    model: *const FyModel,
    options: *const FySessionOptions,
    out: *mut *mut FySession,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            empty_slot(out)?;
            require_abi(required_abi_major)?;
            descriptors::validate(options)?;
            let options = options.read();
            let model = model.cast::<ModelHandle>();
            if model.is_null() || !model.is_aligned() {
                return Err(INVALID);
            }
            // Shared access only; cloning Arc retains state, not a raw handle.
            let state = Arc::clone(&(*model).model);
            let session = Session::new(state, options)?;
            let owned = Box::new(SessionHandle { session });
            // No fallible work follows transfer into the initially empty slot.
            out.write(Box::into_raw(owned).cast());
            Ok(())
        })
    }
}

/// Describe actual session settings without executing a model.
///
/// # Safety
/// session is a correct live handle not concurrently released. out supplies an
/// initialized readable header and a full writable accepted record; payload
/// fields need not be initialized. Headers, padding and extension tails are
/// preserved. All mutable outputs/diagnostics are disjoint and synchronized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_session_describe(
    session: *const FySession,
    out: *mut FySessionInfo,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            descriptors::validate(out)?;
            let session = session.cast::<SessionHandle>();
            if session.is_null() || !session.is_aligned() {
                return Err(INVALID);
            }
            let values = (*session).session.description();
            addr_of_mut!((*out).backend_id).write(values[0]);
            addr_of_mut!((*out).device_kind).write(values[1]);
            addr_of_mut!((*out).device_index).write(values[2]);
            addr_of_mut!((*out).precision).write(values[3]);
            addr_of_mut!((*out).execution_flags).write(values[4]);
            Ok(())
        })
    }
}

/// Release a session and its strong reference to model state.
///
/// # Safety
/// slot is initialized, writable and aligned; non-null values are correct live
/// session handles from this library. No call/release may overlap destruction.
/// Diagnostic storage and slot are disjoint. This is not cancellation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_session_release(
    slot: *mut *mut FySession,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe { entry(diag, || release::<SessionHandle, FySession>(slot)) }
}

/// Copy compiled capabilities, or query their count with null dst and capacity 0.
///
/// # Safety
/// required is aligned writable usize storage; dst, when used, is an aligned
/// allocation of capacity writable records. All output/diagnostic regions are
/// live, disjoint and synchronized. Invalid foreign allocations are not checked.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_capabilities_copy(
    dst: *mut FyCapability,
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
                .checked_mul(size_of::<FyCapability>())
                .ok_or(INVALID)?;
            if bytes > isize::MAX as usize
                || (dst.is_null() && capacity != 0)
                || (!dst.is_null() && !dst.is_aligned())
                || (dst as usize).checked_add(bytes).is_none()
            {
                return Err(INVALID);
            }
            let count = CAPABILITIES.len();
            if dst.is_null() && capacity == 0 {
                required.write(count);
                return Ok(());
            }
            if capacity < count {
                required.write(count);
                return Err(SHORT);
            }
            if count != 0 {
                std::ptr::copy_nonoverlapping(CAPABILITIES.as_ptr(), dst, count);
            }
            required.write(count);
            Ok(())
        })
    }
}

/// Create test-only model state. Never accepts serialized model bytes.
///
/// # Safety
/// out is an initialized empty aligned writable slot. All diagnostic/output
/// storage obeys the same allocation, synchronization and disjointness contract
/// as model_open. Returned handles must be released through this loaded library.
#[cfg(feature = "contract-fixtures")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fy_test_model_create(
    required_abi_major: u32,
    fixture_id: u32,
    out: *mut *mut FyModel,
    diag: *mut FyDiagnostic,
) -> i32 {
    unsafe {
        entry(diag, || {
            empty_slot(out)?;
            require_abi(required_abi_major)?;
            let model = Arc::new(Model::fixture(fixture_id)?);
            let owned = Box::new(ModelHandle {
                model,
                _live: crate::fixtures::Live::new(0),
            });
            out.write(Box::into_raw(owned).cast());
            Ok(())
        })
    }
}
