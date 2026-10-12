// SPDX-License-Identifier: Apache-2.0
use crate::ffi::types::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Error {
    pub status: i32,
    pub message: &'static str,
}

pub(crate) const INVALID: Error = Error {
    status: FY_INVALID_ARGUMENT,
    message: "invalid descriptor or buffer",
};
pub(crate) const VERSION: Error = Error {
    status: FY_UNSUPPORTED_ABI,
    message: "unsupported descriptor version",
};
pub(crate) const FLAGS: Error = Error {
    status: FY_UNSUPPORTED_FEATURE,
    message: "unsupported required flags",
};
pub(crate) const BUILD_ID: Error = Error {
    status: FY_INTERNAL_ERROR,
    message: "invalid compiled build identifier",
};
