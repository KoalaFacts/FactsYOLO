// SPDX-License-Identifier: Apache-2.0
//! Immutable configuration, retained model state and synchronous owned results.
use crate::{
    Error,
    error::INVALID,
    ffi::types::*,
    image::{ImageLimits, ImageView},
    model::Model,
    result::{finalize, owned::OwnedResult},
};
use std::sync::{Arc, Mutex, TryLockError};

pub(crate) const UNSUPPORTED: Error = Error {
    status: FY_UNSUPPORTED_FEATURE,
    message: "unsupported backend or session setting",
};
const DEVICE: Error = Error {
    status: FY_DEVICE_UNAVAILABLE,
    message: "required device is unavailable",
};
const LIMIT: Error = Error {
    status: FY_LIMIT_EXCEEDED,
    message: "requested result count exceeds session limit",
};
const BUSY: Error = Error {
    status: FY_BUSY,
    message: "session is already executing",
};
const POISONED: Error = Error {
    status: FY_POISONED,
    message: "session execution state is poisoned",
};

pub(crate) struct Session {
    model: Arc<Model>,
    options: FySessionOptions,
    execution: Mutex<()>,
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
        Ok(Self {
            model,
            options,
            execution: Mutex::new(()),
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

    pub(crate) fn image_limits(&self) -> ImageLimits {
        ImageLimits {
            max_bytes: self.options.max_image_bytes,
            max_pixels: self.options.max_pixels,
        }
    }

    pub(crate) fn validate_detect(&self, options: FyDetectOptions) -> Result<(), Error> {
        if !options.score_threshold.is_finite()
            || !(0.0..=1.0).contains(&options.score_threshold)
            || options.max_detections == 0
        {
            return Err(INVALID);
        }
        if options.max_detections > self.options.max_detections {
            return Err(LIMIT);
        }
        Ok(())
    }

    pub(crate) fn run(
        &self,
        image: ImageView<'_>,
        options: FyDetectOptions,
    ) -> Result<OwnedResult, Error> {
        self.validate_detect(options)?;
        // Admission covers execution AND finalization. Only shared outer
        // references exist before try_lock; no unbounded queue is introduced.
        let _admission = self.execution.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => BUSY,
            TryLockError::Poisoned(_) => POISONED,
        })?;
        let (candidates, transform) = self.model.execute()?;
        let source = (image.width(), image.height());
        let finalized = finalize(
            candidates,
            transform,
            source,
            2,
            options.score_threshold,
            options.max_detections,
        )?;
        Ok(OwnedResult::new(finalized, source, self.description(), self.model.identity()))
    }
}
