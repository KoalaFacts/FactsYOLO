// SPDX-License-Identifier: Apache-2.0
use super::{descriptors, types::FyDiagnostic};
use crate::error::{Error, INVALID};
use std::num::NonZeroUsize;
use std::ptr::{NonNull, addr_of, addr_of_mut};

/// Per-call validated destinations. Disabled diagnostics contain no pointers.
/// Never stored globally or returned across the ABI.
pub(super) struct Diagnostic {
    storage: Option<Storage>,
}

struct Storage {
    required_bytes: NonNull<usize>,
    buffer: Option<(NonNull<u8>, NonZeroUsize)>,
}

impl Diagnostic {
    /// # Safety
    /// The raw descriptor/buffer satisfy the public allocation and lifetime
    /// preconditions. Output-only required_bytes need not be initialized.
    pub(super) unsafe fn validate(raw: *mut FyDiagnostic) -> Result<Self, Error> {
        let Some(raw) = NonNull::new(raw) else {
            return Ok(Self { storage: None });
        };
        // SAFETY: forwarded caller prefix contract; no full-record read.
        unsafe { descriptors::validate(raw.as_ptr())? };
        // SAFETY: accepted input fields are initialized. Output-only fields and
        // padding are not read, and no reference to the whole record is formed.
        let data = unsafe { addr_of!((*raw.as_ptr()).data).read() }.cast::<u8>();
        let capacity = unsafe { addr_of!((*raw.as_ptr()).capacity).read() };
        if capacity > isize::MAX as usize || (data as usize).checked_add(capacity).is_none() {
            return Err(INVALID);
        }
        let buffer = match NonZeroUsize::new(capacity) {
            Some(capacity) => Some((NonNull::new(data).ok_or(INVALID)?, capacity)),
            None => None,
        };
        // SAFETY: a valid non-null full descriptor supplies this writable field.
        // Retain its destination directly, not a nullable descriptor to later
        // dereference. NonNull is not a substitute for the caller's lifetime,
        // allocation, initialization and disjoint-storage obligations.
        let required_bytes = unsafe { addr_of_mut!((*raw.as_ptr()).required_bytes) };
        Ok(Self {
            storage: Some(Storage {
                required_bytes: NonNull::new(required_bytes).ok_or(INVALID)?,
                buffer,
            }),
        })
    }

    /// # Safety
    /// The validated destinations remain live, writable and disjoint from other
    /// call storage and message. No concurrent access to them is permitted.
    pub(super) unsafe fn write(&self, message: &str) {
        let Some(storage) = &self.storage else {
            return;
        };
        // SAFETY: the validated non-null destination is the output field only.
        unsafe { storage.required_bytes.as_ptr().write(message.len()) };
        let Some((data, capacity)) = storage.buffer else {
            return;
        };
        let count = prefix_len(message, capacity.get());
        // SAFETY: count <= capacity - 1; a present buffer is non-null and has
        // positive capacity. The message and caller's writable range are disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), data.as_ptr(), count);
            data.as_ptr().add(count).write(0);
        }
    }
}

fn prefix_len(message: &str, capacity: usize) -> usize {
    let mut count = message.len().min(capacity.saturating_sub(1));
    // At most three decrements: a UTF-8 scalar occupies at most four bytes.
    while !message.is_char_boundary(count) {
        count -= 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{Diagnostic, prefix_len};
    use crate::ffi::types::FyDiagnostic;
    use std::ptr::null_mut;

    #[test]
    fn disabled_diagnostics_have_no_writable_destinations() {
        let diag = unsafe { Diagnostic::validate(null_mut()) }.unwrap();
        assert!(diag.storage.is_none());
        unsafe { diag.write("no caller diagnostic storage") };
    }

    #[test]
    fn zero_capacity_has_only_a_required_count_destination() {
        let mut raw = FyDiagnostic {
            struct_size: size_of::<FyDiagnostic>() as u32,
            struct_version: 1,
            required_flags: 0,
            data: null_mut(),
            capacity: 0,
            required_bytes: usize::MAX,
        };
        let diag = unsafe { Diagnostic::validate(&mut raw) }.unwrap();
        assert!(diag.storage.as_ref().unwrap().buffer.is_none());
        unsafe { diag.write("éX") };
        assert_eq!(raw.required_bytes, 3);
        assert!(raw.data.is_null());
        assert_eq!(raw.capacity, 0);
    }

    #[test]
    fn unicode_truncation_respects_character_boundaries() {
        assert_eq!(prefix_len("éX", 0), 0);
        assert_eq!(prefix_len("éX", 1), 0);
        assert_eq!(prefix_len("éX", 2), 0);
        assert_eq!(prefix_len("éX", 3), 2);
        assert_eq!(prefix_len("éX", 4), 3);
    }

    #[test]
    fn unicode_writer_preserves_canary_and_required_count() {
        for capacity in 0..=4 {
            let mut bytes = [0xA5; 8];
            let mut raw = FyDiagnostic {
                struct_size: size_of::<FyDiagnostic>() as u32,
                struct_version: 1,
                required_flags: 0,
                data: bytes.as_mut_ptr().cast(),
                capacity,
                required_bytes: usize::MAX,
            };
            let diag = unsafe { Diagnostic::validate(&mut raw) }.unwrap();
            unsafe { diag.write("éX") };
            assert_eq!(raw.required_bytes, 3);
            let count = prefix_len("éX", capacity);
            assert_eq!(&bytes[..count], &"éX".as_bytes()[..count]);
            if capacity > 0 {
                assert_eq!(bytes[count], 0);
            }
            assert!(bytes[capacity..].iter().all(|b| *b == 0xA5));
        }
    }
}
