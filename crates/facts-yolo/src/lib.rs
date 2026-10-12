// SPDX-License-Identifier: Apache-2.0
//! Experimental native infrastructure. No detector or backend is implemented.
mod error;
pub mod ffi;
pub use ffi::exports::fy_abi_query;
