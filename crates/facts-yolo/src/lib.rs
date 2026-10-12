// SPDX-License-Identifier: Apache-2.0
//! Experimental native infrastructure. No trained detector is implemented.
mod error;
pub mod ffi;
#[cfg(feature = "contract-fixtures")]
pub mod fixtures;
pub mod image;
pub use error::Error;
pub use ffi::exports::fy_abi_query;
