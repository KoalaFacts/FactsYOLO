// SPDX-License-Identifier: Apache-2.0
//! Immutable session configuration and strong model retention.
use crate::{Error, error::INVALID, ffi::types::*, model::Model};
use std::sync::Arc;

pub(crate) const UNSUPPORTED: Error = Error {
    status: FY_UNSUPPORTED_FEATURE,
    message: "unsupported backend or session setting",
};
const DEVICE: Error = Error {
    status: FY_DEVICE_UNAVAILABLE,
    message: "required device is unavailable",
};

pub(crate) struct Session {
    model: Arc<Model>,
    options: FySessionOptions,
    #[cfg(feature = "contract-fixtures")]
    _live: crate::fixtures::Live,
}

impl Session {
    pub(crate) fn new(model: Arc<Model>, options: FySessionOptions) -> Result<Self, Error> {
        if options.cpu_threads == 0
            || options.max_image_bytes == 0
            || options.max_pixels == 0
            || options.max_detections == 0
        {
            return Err(INVALID);
        }
        if !matches!(options.device_kind, FY_DEVICE_CPU | FY_DEVICE_GPU) {
            return Err(INVALID);
        }
        if options.backend_id == 0
            || options.backend_id != model.backend_id()
            || options.precision != FY_PRECISION_F32
            || options.fallback_policy != FY_FALLBACK_NONE
            || options.cpu_threads != 1
        {
            return Err(UNSUPPORTED);
        }
        if options.device_kind != FY_DEVICE_CPU || options.device_index != 0 {
            return Err(DEVICE);
        }
        // No mutable execution state or background work exists in task 3.
        // The only constructible model is test-only; normal model_open rejects.
        Ok(Self {
            model,
            options,
            #[cfg(feature = "contract-fixtures")]
            _live: crate::fixtures::Live::new(2),
        })
    }

    pub(crate) fn description(&self) -> [u32; 5] {
        [
            self.model.backend_id(),
            self.options.device_kind,
            self.options.device_index,
            self.options.precision,
            0,
        ]
    }
}
