// SPDX-License-Identifier: Apache-2.0
use super::{descriptors, types::FyDiagnostic};
use crate::error::{Error, INVALID};
use std::ptr::{addr_of, addr_of_mut, null_mut};

/// A per-call raw view. Never stored globally or returned across the ABI.
pub(super) struct Diagnostic {
    raw: *mut FyDiagnostic,
    data: *mut u8,
    capacity: usize,
}

impl Diagnostic {
    /// # Safety
    /// The raw descriptor/buffer satisfy the public allocation and lifetime
    /// preconditions. Output-only required_bytes need not be initialized.
    pub(super) unsafe fn validate(raw: *mut FyDiagnostic) -> Result<Self, Error> {
        if raw.is_null() {
            return Ok(Self { raw, data: null_mut(), capacity: 0 });
        }
        // SAFETY: forwarded caller prefix contract; no full-record read.
        unsafe { descriptors::validate(raw)? };
        // SAFETY: input fields are initialized; required_bytes and padding
        // deliberately are not read.
        let data = unsafe { addr_of!((*raw).data).read() }.cast::<u8>();
        let capacity = unsafe { addr_of!((*raw).capacity).read() };
        if capacity > isize::MAX as usize
            || (capacity > 0 && data.is_null())
            || (data as usize).checked_add(capacity).is_none()
        {
            return Err(INVALID);
        }
        Ok(Self { raw, data, capacity })
    }

    /// # Safety
    /// The validated storage is still live, writable and disjoint from other
    /// call storage and message. No concurrent access to it is permitted.
    pub(super) unsafe fn write(&self, message: &str) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: writes only the output field, not input fields/padding/tails.
        unsafe { addr_of_mut!((*self.raw).required_bytes).write(message.len()) };
        if self.capacity == 0 {
            return;
        }
        let count = prefix_len(message, self.capacity);
        // SAFETY: count <= capacity - 1; caller supplies writable storage.
        // The source is an independent Rust string, and storage is disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(message.as_ptr(), self.data, count);
            self.data.add(count).write(0);
        }
    }
}

// ASCII behavior required by the initial ABI tests; the Unicode regression
// below is deliberately RED until character-boundary truncation is added.
fn prefix_len(message: &str, capacity: usize) -> usize {
    message.len().min(capacity.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::prefix_len;

    #[test]
    fn unicode_truncation_respects_character_boundaries() {
        assert_eq!(prefix_len("éX", 0), 0);
        assert_eq!(prefix_len("éX", 1), 0);
        assert_eq!(prefix_len("éX", 2), 0);
        assert_eq!(prefix_len("éX", 3), 2);
        assert_eq!(prefix_len("éX", 4), 3);
    }
}
