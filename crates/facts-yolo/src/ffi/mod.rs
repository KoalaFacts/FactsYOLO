// SPDX-License-Identifier: Apache-2.0
//! C-compatible boundary records and entry points.
pub(crate) mod descriptors;
mod diagnostics;
pub mod exports;
pub mod handles;
pub mod results;
#[cfg(feature = "contract-fixtures")]
pub mod testing;
pub mod types;
